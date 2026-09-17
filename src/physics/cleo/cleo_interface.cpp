#include "cleo_interface.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <array>
#include <iostream>
#include <memory>
#include <utility>
#include <stdexcept>

#include "cartesiandomain/cartesian_decomposition.hpp"
#include "cartesiandomain/createcartesianmaps.hpp"
#include "cartesiandomain/movement/cartesian_motion.hpp"
#include "configuration/communicator.hpp"
// TODO: update CLEO constants to be consistent with VVMex
#include "cartesiandomain/movement/cartesian_movement.hpp"
#include "cleoconstants.hpp"
#include "initialise/init_all_supers_from_binary.hpp"
#include "initialise/initgbxsnull.hpp"
#include "initialise/timesteps.hpp"
#include "cartesiandomain/collect_data_for_collective_dataset.hpp"
#include "observers/gbxindex_observer.hpp"
#include "observers/massmoments_observer.hpp"
#include "observers/nsupers_observer.hpp"
#include "observers/observers.hpp"
#include "observers/sdmmonitor/monitor_condensation_observer.hpp"
#include "observers/sdmmonitor/monitor_massmoments_change_observer.hpp"
#include "observers/sdmmonitor/monitor_precipitation_observer.hpp"
#include "observers/state_observer.hpp"
#include "observers/streamout_observer.hpp"
#include "observers/superdrops_observer.hpp"
#include "observers/time_observer.hpp"
#include "observers/totnsupers_observer.hpp"
#include "runcleo/creategbxs.hpp"
#include "runcleo/createsupers.hpp"
#include "runcleo/sdmmethods.hpp"
#include "superdrops/collisions/breakup_nfrags.hpp"
#include "superdrops/collisions/coalbure.hpp"
#include "superdrops/collisions/coalbure_flag.hpp"
#include "superdrops/collisions/collisions.hpp"
#include "superdrops/collisions/longhydroprob.hpp"
#include "superdrops/condensation.hpp"
#include "zarr/fsstore.hpp"
#include "zarr/collective_dataset.hpp"
#include "core/Field.hpp"

namespace VVM {
namespace Physics {

namespace {

using Store = FSStore;
/* CollectiveDataset, not SimpleDataset.

SimpleDataset has no notion of MPI: every rank would open the same FSStore path
and write its own gridboxes starting at index 0 of arrays that are dimensioned
for the WHOLE domain (physics.cleo.ngbxs is global). With two ranks that means
each overwrites the other's half and the far half of every row stays empty, while
the ragged superdroplet arrays interleave into nonsense. Nothing reports it --
the zarr is simply wrong. CollectiveDataset takes the decomposition and places
each rank's slice at the right offset. */
using Dataset = CollectiveDataset<Store, CartesianDecomposition>;

static SuppliedDecomposition
supplied_from_grid(const Core::Grid& grid) {
    init_communicator::set_communicator(grid.get_cart_comm());

    const MPI_Comm cart = grid.get_cart_comm();
    int nprocs = 0;
    MPI_Comm_size(cart, &nprocs);

    const std::array<size_t, 3> my_origin{0,
        static_cast<size_t>(grid.get_local_physical_start_x()),
        static_cast<size_t>(grid.get_local_physical_start_y())};
    const std::array<size_t, 3> my_size{static_cast<size_t>(grid.get_global_points_z()),
        static_cast<size_t>(grid.get_local_physical_points_x()),
        static_cast<size_t>(grid.get_local_physical_points_y())};

    SuppliedDecomposition supplied;
    supplied.origins.resize(nprocs);
    supplied.sizes.resize(nprocs);

    MPI_Allgather(my_origin.data(),
        3,
        MPI_UNSIGNED_LONG,
        supplied.origins.data(),
        3,
        MPI_UNSIGNED_LONG,
        cart);
    MPI_Allgather(my_size.data(),
        3,
        MPI_UNSIGNED_LONG,
        supplied.sizes.data(),
        3,
        MPI_UNSIGNED_LONG,
        cart);

    for (int d = 0; d < 3; ++d) {
        std::vector<size_t> distinct;
        for (const auto& o : supplied.origins) {
            distinct.push_back(o[d]);
        }
        std::sort(distinct.begin(), distinct.end());
        distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
        supplied.decomposition[d] = distinct.size();
    }
    return supplied;
}

inline MicrophysicalProcess auto
create_microphysics(const Timesteps& ts, const Utils::ConfigurationManager& config) {
    // === condensation ====
    OptionalConfigParams::CondensationParams c;
    c.do_alter_thermo =
        config.get_value("physics.cleo.microphysics.condensation.do_alter_thermo", true);
    c.maxniters = config.get_value("physics.cleo.microphysics.condensation.maxniters", 100);
    c.MINSUBTSTEP = config.get_value("physics.cleo.microphysics.condensation.MINSUBTSTEP", 1e-5);
    c.rtol = config.get_value("physics.cleo.microphysics.condensation.rtol", 0.0);
    c.atol = config.get_value("physics.cleo.microphysics.condensation.atol", 0.01);
    // TODO: fill this params
    auto cond = Condensation(ts.get_condstep(),
        &step2dimlesstime,
        c.do_alter_thermo,
        c.maxniters,
        c.rtol,
        c.atol,
        c.MINSUBTSTEP,
        &realtime2dimless);

    // === collision ====
    const PairProbability auto collprob = LongHydroProb();
    const NFragments auto nfrags =
        ConstNFrags(config.get_value<double>("physics.cleo.microphysics.breakup_nfrags", 5.0));
    const CoalBuReFlag auto coalbure_flag = TSCoalBuReFlag(RogersGKTerminalVelocity{});

    // Collision is disabled by giving ConstTstepMicrophysics an interval of
    // LIMITVALUES::uintmax: on_step() special-cases that value and never fires, so
    // neither the Monte-Carlo step nor the Fisher-Yates shuffle runs. Keeping the
    // same type here is what lets this be a runtime switch at all.
    const auto enable_collision =
        config.get_value<bool>("physics.cleo.microphysics.enable_collision", true);
    const auto collint = enable_collision ? ts.get_collstep() : LIMITVALUES::uintmax;

    // Without a seed CLEO seeds the pool from std::random_device, so no two runs
    // agree. A non-zero collision_seed makes a single-rank run reproducible; it does
    // NOT make ranks agree, because the pool's state is per-thread, not per-gridbox.
    const auto collision_seed =
        config.get_value<std::uint64_t>("physics.cleo.microphysics.collision_seed", 0);

    const MicrophysicalProcess auto colls =
        collision_seed
            ? CoalBuRe(collint, &step2realtime, collprob, nfrags, coalbure_flag, collision_seed)
            : CoalBuRe(collint, &step2realtime, collprob, nfrags, coalbure_flag);
    return colls >> cond;
}

template <GridboxMaps GbxMaps>
inline auto
create_movement(const Timesteps& ts, const GbxMaps& gbxmaps) {
    const Motion<CartesianMaps> auto motion =
        CartesianMotion(ts.get_motionstep(), &step2dimlesstime, RogersGKTerminalVelocity{});
    const BoundaryConditions<CartesianMaps> auto boundary_conditions = NullBoundaryConditions();
    const auto movement = cartesian_movement(gbxmaps, motion, boundary_conditions);
    return movement;
}

template <typename Dataset, typename Store>
inline Observer auto
create_superdrops_observer(
    const unsigned int interval, Dataset& dataset, Store& store, const size_t maxchunk) {
    CollectDataForDataset<Dataset> auto sdid = CollectSdId(dataset, maxchunk);
    CollectDataForDataset<Dataset> auto sdgbxindex = CollectSdgbxindex(dataset, maxchunk);
    CollectDataForDataset<Dataset> auto xi = CollectXi(dataset, maxchunk);
    CollectDataForDataset<Dataset> auto radius = CollectRadius(dataset, maxchunk);
    CollectDataForDataset<Dataset> auto msol = CollectMsol(dataset, maxchunk);
    CollectDataForDataset<Dataset> auto coord3 = CollectCoord3(dataset, maxchunk);
    CollectDataForDataset<Dataset> auto coord1 = CollectCoord1(dataset, maxchunk);
    CollectDataForDataset<Dataset> auto coord2 = CollectCoord2(dataset, maxchunk);

    /* sdId FIRST, and the order matters.

    CollectiveDataset::write_to_ragged_array reorders every superdroplet array
    through global_superdroplet_ordering, which it builds from the sdId array as
    a side effect of writing it -- so any array written before sdId indexes that
    table while it still holds its UINT_MAX fill value. CLEO says so in a comment
    on that function; it is not visible from this end. Under SimpleDataset the
    order is irrelevant, which is why the old chain (sdId last) went unnoticed.
    CombinedCollectDataForDataset writes its left operand first, and >> is
    left-associative, so leftmost is written first. */
    const auto collect_sddata =
        sdid >> sdgbxindex >> xi >> radius >> msol >> coord3 >> coord2 >> coord1;
    return SuperdropsObserver(interval, dataset, store, maxchunk, collect_sddata);
}

template <typename Dataset>
inline Observer auto
create_gridboxes_observer(
    const unsigned int interval, Dataset& dataset, const size_t maxchunk, const size_t ngbxs) {
    const CollectDataForDataset<Dataset> auto thermo = CollectThermo(dataset, maxchunk, ngbxs);
    const CollectDataForDataset<Dataset> auto windvel = CollectWindVel(dataset, maxchunk, ngbxs);
    const CollectDataForDataset<Dataset> auto nsupers = CollectNsupers(dataset, maxchunk, ngbxs);

    const CollectDataForDataset<Dataset> auto collect_gbxdata = nsupers >> windvel >> thermo;
    return WriteToDatasetObserver(interval, dataset, collect_gbxdata);
}

template <typename Dataset, typename Store>
inline Observer auto
create_sdmmonitor_observer(const unsigned int interval,
    Dataset& dataset,
    Store& store,
    const size_t maxchunk,
    const size_t ngbxs) {
    const Observer auto obs_cond =
        MonitorCondensationObserver(interval, dataset, store, maxchunk, ngbxs);
    const Observer auto obs_massmoms =
        MonitorMassMomentsChangeObserver(interval, dataset, store, maxchunk, ngbxs);
    const Observer auto obs_rainmassmoms =
        MonitorRainMassMomentsObserver(interval, dataset, store, maxchunk, ngbxs);
    const Observer auto obs_precip =
        MonitorPrecipitationObserver(interval, dataset, store, maxchunk, ngbxs);

    return obs_cond >> obs_massmoms >> obs_rainmassmoms >> obs_precip;
}

template <typename Dataset, typename Store>
inline Observer auto
create_observer(const Timesteps& ts,
    const Utils::ConfigurationManager& config,
    Dataset& dataset,
    Store& store,
    const size_t ngbxs) {
    const auto obsstep = ts.get_obsstep();
    const auto maxchunk = config.get_value<size_t>("physics.cleo.output.maxchunk");

    const Observer auto obs0 = StreamOutObserver(obsstep, &step2realtime);

    const Observer auto obs1 = TimeObserver(obsstep, dataset, store, maxchunk, &step2dimlesstime);

    const Observer auto obs2 = GbxindexObserver(dataset, store, maxchunk, ngbxs);

    const Observer auto obs3 = TotNsupersObserver(obsstep, dataset, store, maxchunk);

    const Observer auto obs4 = MassMomentsObserver(obsstep, dataset, store, maxchunk, ngbxs);

    const Observer auto obs5 =
        MassMomentsRaindropsObserver(obsstep, dataset, store, maxchunk, ngbxs);

    const Observer auto obsgbx = create_gridboxes_observer(obsstep, dataset, maxchunk, ngbxs);

    const Observer auto obssd = create_superdrops_observer(obsstep, dataset, store, maxchunk);

    const Observer auto obsm = create_sdmmonitor_observer(obsstep, dataset, store, maxchunk, ngbxs);

    return obsm >> obssd >> obsgbx >> obs5 >> obs4 >> obs3 >> obs2 >> obs1 >> obs0;
}

inline Timesteps
timesteps_from(const Utils::ConfigurationManager& config) {
    RequiredConfigParams::TimestepsParams p;
    p.CONDTSTEP = config.get_value<double>("physics.cleo.timesteps_s.condensation");
    p.COLLTSTEP = config.get_value<double>("physics.cleo.timesteps_s.collision");
    p.MOTIONTSTEP = config.get_value<double>("physics.cleo.timesteps_s.motion");
    p.COUPLTSTEP = config.get_value<double>("physics.cleo.timesteps_s.coupling");
    p.OBSTSTEP = config.get_value<double>("physics.cleo.timesteps_s.observation");
    p.T_END = config.get_value<double>("simulation.total_time_s");
    return Timesteps(p);
}

struct SdmRunner {
    virtual ~SdmRunner() = default;

    virtual unsigned int get_couplstep() const = 0;
    virtual unsigned int next_couplstep(unsigned int t_mdl) const = 0;
    virtual void prepare_to_timestep(dualview_gbx gbxs, const SupersInDomain& allsupers) const = 0;
    virtual void at_start_step(
        unsigned int t_mdl, dualview_gbx gbxs, const SupersInDomain& allsupers) const = 0;
    virtual void run_step(unsigned int t_mdl,
        unsigned int t_mdl_next,
        viewd_gbx d_gbxs,
        SupersInDomain& allsupers) const = 0;
    virtual void after_timestepping() const = 0;
};

template <typename SDM>
struct SdmRunnerOf final : SdmRunner {
    SDM sdm;

    explicit SdmRunnerOf(SDM sdm) : sdm(std::move(sdm)) {}

    unsigned int
    get_couplstep() const override {
        return sdm.get_couplstep();
    }
    unsigned int
    next_couplstep(unsigned int t_mdl) const override {
        return sdm.next_couplstep(t_mdl);
    }
    void
    prepare_to_timestep(dualview_gbx gbxs, const SupersInDomain& allsupers) const override {
        sdm.prepare_to_timestep(gbxs, allsupers);
    }
    void
    at_start_step(
        unsigned int t_mdl, dualview_gbx gbxs, const SupersInDomain& allsupers) const override {
        sdm.at_start_step(t_mdl, gbxs, allsupers);
    }
    void
    run_step(unsigned int t_mdl,
        unsigned int t_mdl_next,
        viewd_gbx d_gbxs,
        SupersInDomain& allsupers) const override {
        sdm.run_step(t_mdl, t_mdl_next, d_gbxs, allsupers);
    }
    void
    after_timestepping() const override {
        sdm.obs.after_timestepping();
    }
};

template <typename SDM>
inline std::unique_ptr<SdmRunner>
erase_sdm(SDM sdm) {
    return std::make_unique<SdmRunnerOf<SDM>>(std::move(sdm));
}

struct CleoState {
    const Timesteps ts;
    Store store;
    Dataset dataset;
    const CartesianMaps gbxmaps;
    SupersInDomain allsupers;
    dualview_gbx gbxs;
    std::unique_ptr<SdmRunner> sdm;

    const size_t lnz, lnx, lny, halo;

    /* Write the y-mean of the SDM feedback back to every y column.

    For a quasi-2-D run the y cells are not independent physics: they exist only
    because the halo machinery needs a second point, and they are replicate
    samples of the same 2-D column. SDM is stochastic, so their feedback differs
    by a small random amount -- and with ny = 2 and periodic boundaries that
    difference is precisely the Nyquist mode, which this configuration amplifies
    with an e-folding time of about seven timesteps. Averaging removes it by
    construction and is also the better Monte-Carlo estimate, since it uses every
    superdroplet in the column. Leave it off for a genuinely 3-D run, where the y
    structure is physics. */
    const bool average_over_y;

    CleoState(const Utils::ConfigurationManager& config, const Core::Grid& grid)
        : ts(timesteps_from(config)),
          store(config.get_value<std::string>("output.output_dir") +
                config.get_value<std::string>("physics.cleo.output.zarrbasedir", "vvm_sol.zarr")),
          dataset(store),
          gbxmaps(create_cartesian_maps(config.get_value<std::size_t>("physics.cleo.ngbxs"),
              3,
              config.get_value<std::string>("physics.cleo.init_gbx_path"),
              supplied_from_grid(grid))),
          // gbxmaps, not the bare local gridbox count: the binary numbers gridboxes
          // globally and every rank reads all of it, so the indexes have to be
          // mapped onto this rank before anything can tell "not mine" apart from
          // "somewhere else in my partition".
          allsupers(
              create_supers(InitAllSupersFromBinary(config.get_value<std::size_t>(
                                                        "physics.cleo.superdroplets.max_total"),
                                config.get_value<std::string>("physics.cleo.init_supers_path"),
                                3),
                  gbxmaps)),
          gbxs(create_gbxs(
              gbxmaps, InitGbxsNull(gbxmaps.get_local_ngridboxes_hostcopy()), allsupers)),
          sdm(erase_sdm(SDMMethods(ts.get_couplstep(),
              gbxmaps,
              create_microphysics(ts, config),
              create_movement(ts, gbxmaps),
              // The LOCAL gridbox count, not physics.cleo.ngbxs. CollectiveDataset
              // treats each rank's declared dimension as that rank's share and sums
              // them into the global one, so handing every rank the global count
              // makes every gridbox-dimensioned array comm_size times too wide,
              // with each rank's slice written at the wrong offset.
              create_observer(ts, config, dataset, store,
                  gbxmaps.get_local_ngridboxes_hostcopy())))),
          lnz(grid.get_local_physical_points_z()), lnx(grid.get_local_physical_points_x()),
          lny(grid.get_local_physical_points_y()), halo(grid.get_halo_cells()),
          average_over_y(config.get_value("physics.cleo.average_over_y", false)) {
        /* Must follow sdm: the observers are built during its construction and
        hold the dataset by reference, so the decomposition only has to be in
        place before the first write, not before they are created. This is the
        order CLEO's own examples/fromfile/src/main_fromfile.cpp uses. */
        dataset.set_decomposition(gbxmaps.get_domain_decomposition());
        dataset.set_max_superdroplets(
            config.get_value<unsigned int>("physics.cleo.superdroplets.max_total"));

        sdm->prepare_to_timestep(gbxs, allsupers);

        // The coupling loops decode CLEO's gridbox index with these three sizes;
        // if they do not multiply out to the gridbox count CLEO actually built,
        // every field would be mapped to the wrong cell silently.
        if (lnz * lnx * lny != gbxs.extent(0)) {
            throw std::runtime_error("CLEO built " + std::to_string(gbxs.extent(0)) +
                                     " local gridboxes but the grid's "
                                     "local physical points are " +
                                     std::to_string(lnz) + "*" + std::to_string(lnx) + "*" +
                                     std::to_string(lny) + " = " + std::to_string(lnz * lnx * lny));
        }
    }

    void
    send_dynamics(const Core::State& state) {
        const auto u = state.get_field<3>("u").get_device_data();
        const auto v = state.get_field<3>("v").get_device_data();
        const auto w = state.get_field<3>("w").get_device_data();
        const auto th = state.get_field<3>("th").get_device_data();
        const auto pibar = state.get_field<1>("pibar").get_device_data();
        const auto qvap = state.get_field<3>("qv").get_device_data();
        const auto press = state.get_field<1>("pbar").get_device_data();

        const auto nz = lnz, nx = lnx, h = halo;
        gbxs.sync_device();
        const auto d_gbxs = gbxs.view_device();
        Kokkos::parallel_for("cleo_send_dynamics",
            Kokkos::RangePolicy<ExecSpace>(0, d_gbxs.extent(0)),
            KOKKOS_LAMBDA(const size_t idx) {
                const auto k = idx % nz;
                const auto i = (idx / nz) % nx;
                const auto j = idx / (nz * nx);
                const auto ka = h + k, ja = h + j, ia = h + i;

                auto& gb_state = d_gbxs(idx).state;
                gb_state.uvel = {u(ka, ja, ia - 1) / dlc::W0, u(ka, ja, ia) / dlc::W0};
                gb_state.vvel = {v(ka, ja - 1, ia) / dlc::W0, v(ka, ja, ia) / dlc::W0};
                gb_state.wvel = {w(ka - 1, ja, ia) / dlc::W0, w(ka, ja, ia) / dlc::W0};
                gb_state.temp = th(ka, ja, ia) * pibar(ka) / dlc::TEMP0;
                gb_state.qvap = qvap(ka, ja, ia);
                gb_state.press = press(ka) / dlc::P0;
            });
        gbxs.modify_device();
    }

    void
    receive_dynamics(Core::State& state, Core::HaloExchanger& halo_exchanger) {
        auto th = state.get_field<3>("th").get_mutable_device_data();
        auto qvap = state.get_field<3>("qv").get_mutable_device_data();
        auto qcond = state.get_field<3>("qcond").get_mutable_device_data();
        auto qp = state.get_field<3>("qp").get_mutable_device_data();
        const auto pibar = state.get_field<1>("pibar").get_device_data();

        const auto nz = lnz, nx = lnx, ny = lny, h = halo;
        const bool average_y = average_over_y;
        const auto d_gbxs = gbxs.view_device();
        Kokkos::parallel_for("cleo_receive_dynamics",
            Kokkos::RangePolicy<ExecSpace>(0, nz * nx),
            KOKKOS_LAMBDA(const size_t ki) {
                const auto k = ki % nz;
                const auto i = ki / nz;
                const auto ka = h + k, ia = h + i;
                double t_sum = 0.0, qv_sum = 0.0, qc_sum = 0.0;
                if (average_y) {
                    for (size_t j = 0; j < ny; ++j) {
                        const auto& st = d_gbxs(k + nz * (i + nx * j)).state;
                        t_sum += st.temp;
                        qv_sum += st.qvap;
                        qc_sum += st.qcond;
                    }
                }
                const double inv = average_y ? 1.0 / static_cast<double>(ny) : 1.0;
                for (size_t j = 0; j < ny; ++j) {
                    const auto ja = h + j;
                    const auto& st = d_gbxs(k + nz * (i + nx * j)).state;
                    const double temp = average_y ? t_sum * inv : st.temp;
                    const double qv = average_y ? qv_sum * inv : st.qvap;
                    const double qc = average_y ? qc_sum * inv : st.qcond;
                    th(ka, ja, ia) = temp * dlc::TEMP0 / pibar(ka);
                    qvap(ka, ja, ia) = qv;
                    qcond(ka, ja, ia) = qc;
                    qp(ka, ja, ia) = qc;
                }
            });

        // The loop writes physical cells only, so without this the halos still
        // hold the values from before the coupling. The dynamics differentiates
        // across them, and the resulting step at the domain edge is a forcing
        // that is not physics: in a quasi-2-D run (ny = 2, periodic) it drives a
        // v that should be identically zero, and that grows until the wind is
        // large enough to break the superdroplet motion's CFL check -- which
        // reports a timestep problem, several steps after the actual cause.
        halo_exchanger.exchange_halos(state.get_field<3>("th"));
        halo_exchanger.exchange_halos(state.get_field<3>("qv"));
        halo_exchanger.exchange_halos(state.get_field<3>("qcond"));
        halo_exchanger.exchange_halos(state.get_field<3>("qp"));
    }
};

} // namespace

struct CLEO_Interface::Impl {
    const Utils::ConfigurationManager& config;
    const Core::Grid& grid;
    Core::HaloExchanger& halo_exchanger;
    std::unique_ptr<CleoState> cleo;

    double next_couple_s = 0.0;
    unsigned int t_cleo = 0;

    Impl(const Utils::ConfigurationManager& config,
        const Core::Grid& grid,
        Core::HaloExchanger& halo_exchanger)
        : config(config), grid(grid), halo_exchanger(halo_exchanger) {}

    void
    initialize(Core::State& state) {
        /* Before CleoState, not inside it. CollectiveDataset grabs the
        communicator in its constructor, and CleoState builds `dataset` before
        `gbxmaps` -- whose initialiser is where supplied_from_grid() used to set
        it. The dataset then held MPI_COMM_NULL and hung on its first MPI_Gather,
        inside the TimeObserver's coordinate array, before a single step ran. */
        init_communicator::set_communicator(grid.get_cart_comm());

        cleo = std::make_unique<CleoState>(config, grid);

        std::cout << "[CLEO] rank " << grid.get_mpi_rank()
                  << " ready: " << cleo->gbxmaps.get_local_ngridboxes_hostcopy() << " of "
                  << config.get_value<std::size_t>("physics.cleo.ngbxs") << " gridboxes, couplstep "
                  << cleo->sdm->get_couplstep() << " (dimensionless, TIME0=" << dlc::TIME0
                  << "s), zarr at "
                  << config.get_value<std::string>("output.output_dir") +
                         config.get_value<std::string>("physics.cleo.output.zarrbasedir",
                             "vvm_sol.zarr")
                  << std::endl;

        // send initial field to cleo gridbox
        cleo->send_dynamics(state);
    }
};

CLEO_Interface::CLEO_Interface(const Utils::ConfigurationManager& config,
    const Core::Grid& grid,
    Core::HaloExchanger& halo_exchanger,
    Core::State& state)
    : impl_(std::make_unique<Impl>(config, grid, halo_exchanger)) {
    const std::array<int, 3> shape{grid.get_local_total_points_z(),
        grid.get_local_total_points_y(),
        grid.get_local_total_points_x()};

    state.add_field<3>("qcond",
        shape,
        Core::FieldMetadata{Core::GridStaggering::Centered,
            "kg kg-1",
            "total condensation mixing ratio"});

    if (!state.has_field("qp")) {
        state.add_field<3>("qp",
            shape,
            Core::FieldMetadata{Core::GridStaggering::Centered,
                "kg kg-1",
                "total hydrometeor mass mixing ratio"});
    }
}

CLEO_Interface::~CLEO_Interface() = default;

void
CLEO_Interface::initialize(Core::State& state) {
    if (impl_->grid.get_mpi_rank() == 0) {
        std::cout << "[CLEO] initializing..." << std::endl;
    }
    impl_->initialize(state);
}

void
CLEO_Interface::run(Core::State& state, const VVM::Real dt) {
    auto& cleo = *impl_->cleo;
    auto& sdm = *cleo.sdm;

    const double couplstep_s = step2realtime(sdm.get_couplstep());
    if (state.get_time() + 0.5 * dt < impl_->next_couple_s) {
        return;
    }
    impl_->next_couple_s += couplstep_s;

    const auto t_mdl = impl_->t_cleo;
    const auto t_next = t_mdl + sdm.get_couplstep();

    if (t_next != sdm.next_couplstep(t_mdl)) {
        throw std::runtime_error("CLEO is out of sync with the coupling: t_mdl " +
                                 std::to_string(t_mdl) + " + couplstep gives " +
                                 std::to_string(t_next) + " but the next coupling boundary is " +
                                 std::to_string(sdm.next_couplstep(t_mdl)));
    }
    impl_->t_cleo = t_next;

    cleo.send_dynamics(state);
    sdm.at_start_step(t_mdl, cleo.gbxs, cleo.allsupers);
    sdm.run_step(t_mdl, t_next, cleo.gbxs.view_device(), cleo.allsupers);
    cleo.receive_dynamics(state, impl_->halo_exchanger);
}

void
CLEO_Interface::finalize() {
    if (impl_ && impl_->cleo) {
        impl_->cleo->sdm->after_timestepping();
    }
}

} // namespace Physics
} // namespace VVM
