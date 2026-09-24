#include "cleo_interface.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
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
#include "superdrops/sdmmonitor.hpp"
#include "superdrops/thermodynamic_equations.hpp"
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

/* Condensate mass that has fallen through the bottom face of each local gridbox.

CLEO's MonitorPrecipitation measures the same crossing, but it lives inside the
observer chain, is zeroed on every observation step and only ever reaches the
zarr, so the coupling cannot read it. This one the coupling owns: d_mass is a
Kokkos::View, so the copy SDMMethods takes through get_sdmmonitor() writes into
the memory CleoState reads.

It stores sum(condensate_mass * xi) only and leaves the division by area to the
caller. CartesianMaps::get_gbxarea reads a host-side map, so calling it here --
inside the motion kernel, as MonitorPrecipitation does -- is a host read from
device code. */
struct BottomFluxMonitor {
    Kokkos::View<double*> d_mass;

    void
    reset_monitor() const {}

    KOKKOS_FUNCTION
    void
    before_timestepping(const TeamMember&, const viewd_constsupers) const {}

    KOKKOS_FUNCTION
    void
    monitor_condensation(const TeamMember&, const double) const {}

    KOKKOS_FUNCTION
    void
    monitor_microphysics(const TeamMember&, const viewd_constsupers) const {}

    void
    monitor_motion(const auto, const auto) const {}

    // Called after the drop's coordinates are updated and before its gridbox
    // index is, so gbxindex is still the box it is leaving; league_rank() is that
    // box's position in the local gridbox view.
    KOKKOS_FUNCTION
    void
    monitor_precipitation(const TeamMember& team_member,
        const unsigned int gbxindex,
        const auto& gbxmaps,
        Superdrop& drop) const {
        if (drop.get_coord3() < gbxmaps.coord3bounds(gbxindex).first) {
            Kokkos::atomic_add(&d_mass(team_member.league_rank()),
                drop.condensate_mass() * static_cast<double>(drop.get_xi()));
        }
    }
};

// An observer that never observes; it exists only to put BottomFluxMonitor into
// the chain SDMMethods takes its monitor from.
struct BottomFluxObserver {
    BottomFluxMonitor mo;

    void
    before_timestepping(const viewd_constgbx, const subviewd_constsupers) const {}
    void
    after_timestepping() const {}
    unsigned int
    next_obs(const unsigned int) const {
        return LIMITVALUES::uintmax;
    }
    bool
    on_step(const unsigned int) const {
        return false;
    }
    void
    at_start_step(const unsigned int, const viewd_constgbx, const subviewd_constsupers) const {}
    SDMMonitor auto
    get_sdmmonitor() const {
        return mo;
    }
};

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
    // Before sdm: the observer built inside sdm's initialiser holds a copy of it.
    Kokkos::View<double*> bottom_mass;
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

    // Radius bounds [um] splitting superdroplets into haze, cloud and rain for
    // the bulk diagnostics; see diagnose_hydrometeors.
    const double cloud_min_radius_um, rain_min_radius_um;
    const double cell_area_m2;
    // Per local gridbox: qc, qr, nc, then sum(xi r^3) and sum(xi r^2) for cloud
    // and for rain.
    Kokkos::View<double**> moments;

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
          allsupers(create_supers(
              InitAllSupersFromBinary(
                  config.get_value<std::size_t>("physics.cleo.superdroplets.max_total"),
                  config.get_value<std::string>("physics.cleo.init_supers_path"),
                  3),
              gbxmaps,
              // The superdroplet view is sized from what THIS rank starts with, not
              // from the whole domain, so it needs room for superdroplets drifting in
              // from neighbours. Too small and the exchange throws; too large and the
              // per-step host staging and the sort both carry the slack.
              config.get_value<double>("physics.cleo.superdroplets.rank_capacity_factor", 1.5))),
          gbxs(create_gbxs(
              gbxmaps, InitGbxsNull(gbxmaps.get_local_ngridboxes_hostcopy()), allsupers)),
          bottom_mass("cleo_bottom_mass", gbxmaps.get_local_ngridboxes_hostcopy()),
          sdm(erase_sdm(SDMMethods(ts.get_couplstep(),
              gbxmaps,
              create_microphysics(ts, config),
              create_movement(ts, gbxmaps),
              // The LOCAL gridbox count, not physics.cleo.ngbxs. CollectiveDataset
              // treats each rank's declared dimension as that rank's share and sums
              // them into the global one, so handing every rank the global count
              // makes every gridbox-dimensioned array comm_size times too wide,
              // with each rank's slice written at the wrong offset.
              create_observer(
                  ts, config, dataset, store, gbxmaps.get_local_ngridboxes_hostcopy()) >>
                  BottomFluxObserver{BottomFluxMonitor{bottom_mass}}))),
          lnz(grid.get_local_physical_points_z()), lnx(grid.get_local_physical_points_x()),
          lny(grid.get_local_physical_points_y()), halo(grid.get_halo_cells()),
          average_over_y(config.get_value("physics.cleo.average_over_y", false)),
          cloud_min_radius_um(
              config.get_value<double>("physics.cleo.diagnostics.cloud_min_radius_um", 1.0)),
          rain_min_radius_um(
              config.get_value<double>("physics.cleo.diagnostics.rain_min_radius_um", 40.0)),
          cell_area_m2(static_cast<double>(grid.get_dx()) * static_cast<double>(grid.get_dy())),
          moments("cleo_hydrometeor_moments", lnz * lnx * lny, 7) {
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

    /* The bulk fields the rest of the model expects from a microphysics scheme,
    diagnosed from the superdroplets: qc/qr/nc and the effective radii are what
    RRTMGP reads from P3, so with these in place radiation sees CLEO's cloud
    without knowing where it came from.

    Superdroplets are split by radius. Below cloud_min_radius they are haze --
    unactivated aerosol, numerous enough to drag the effective radius down and
    to dominate nc, but with negligible mass -- and are left out. Between that
    and rain_min_radius they are cloud, above it rain (40 um is CLEO's own
    raindrop threshold, see MassMomentsRaindropsObserver). qcond, which drives
    the buoyancy, still counts every superdroplet.

    Diagnostic only: nothing here is fed back into th/qv, and qc/qr are not
    advected, since the superdroplets carry the water themselves. */
    void
    diagnose_hydrometeors(Core::State& state) {
        const auto nz = lnz, nx = lnx, ny = lny, h = halo;
        const bool average_y = average_over_y;
        const double r_cloud = cloud_min_radius_um * 1e-6 / dlc::R0;
        const double r_rain = rain_min_radius_um * 1e-6 / dlc::R0;
        const double um_per_r = dlc::R0 * 1e6;

        const auto d_gbxs = gbxs.view_device();
        const auto domainsupers = allsupers.domain_supers_readonly();
        const auto mom = moments;
        // One thread per gridbox, serial over its superdroplets: this runs once
        // per coupling step, and the per-box counts are small.
        Kokkos::parallel_for("cleo_hydrometeor_moments",
            Kokkos::RangePolicy<ExecSpace>(0, d_gbxs.extent(0)),
            KOKKOS_LAMBDA(const size_t idx) {
                const auto& st = d_gbxs(idx).state;
                const auto supers = d_gbxs(idx).supersingbx.readonly(domainsupers);
                double mc = 0.0, mr = 0.0, nc = 0.0, c3 = 0.0, c2 = 0.0, r3 = 0.0, r2 = 0.0;
                for (size_t kk = 0; kk < supers.extent(0); ++kk) {
                    const auto& drop = supers(kk);
                    const double r = drop.get_radius();
                    const double xi = static_cast<double>(drop.get_xi());
                    const double xr2 = xi * r * r;
                    if (r >= r_rain) {
                        mr += drop.condensate_mass() * xi;
                        r3 += xr2 * r;
                        r2 += xr2;
                    } else if (r >= r_cloud) {
                        mc += drop.condensate_mass() * xi;
                        nc += xi;
                        c3 += xr2 * r;
                        c2 += xr2;
                    }
                }
                // Dry-air mass of the gridbox in kg, as EffectOnQcondFunctor
                // uses for qcond.
                const double air = st.get_volume() * dlc::VOL0 *
                                   dry_air_density(st.press, st.temp, st.qvap) * dlc::RHO0;
                mom(idx, 0) = mc * dlc::MASS0 / air;
                mom(idx, 1) = mr * dlc::MASS0 / air;
                mom(idx, 2) = nc / air;
                mom(idx, 3) = c3;
                mom(idx, 4) = c2;
                mom(idx, 5) = r3;
                mom(idx, 6) = r2;
            });

        auto qc = state.get_field<3>("qc").get_mutable_device_data();
        auto qr = state.get_field<3>("qr").get_mutable_device_data();
        auto ncf = state.get_field<3>("nc").get_mutable_device_data();
        auto re_c = state.get_field<3>("diag_eff_radius_qc").get_mutable_device_data();
        auto re_r = state.get_field<3>("diag_eff_radius_qr").get_mutable_device_data();

        // As receive_dynamics: with average_over_y every y column gets the
        // column mean. Radiation heats th, so replicate columns that see
        // different clouds would reintroduce exactly the y asymmetry that
        // averaging the feedback is there to remove.
        Kokkos::parallel_for("cleo_diagnose_hydrometeors",
            Kokkos::RangePolicy<ExecSpace>(0, nz * nx),
            KOKKOS_LAMBDA(const size_t ki) {
                const auto k = ki % nz;
                const auto i = ki / nz;
                const auto ka = h + k, ia = h + i;
                constexpr int nmom = 7;
                double sum[nmom] = {};
                if (average_y) {
                    for (size_t j = 0; j < ny; ++j) {
                        for (int m = 0; m < nmom; ++m) {
                            sum[m] += mom(k + nz * (i + nx * j), m);
                        }
                    }
                }
                const double inv = average_y ? 1.0 / static_cast<double>(ny) : 1.0;
                for (size_t j = 0; j < ny; ++j) {
                    const auto ja = h + j;
                    double v[nmom];
                    for (int m = 0; m < nmom; ++m) {
                        v[m] = average_y ? sum[m] * inv : mom(k + nz * (i + nx * j), m);
                    }
                    qc(ka, ja, ia) = v[0];
                    qr(ka, ja, ia) = v[1];
                    ncf(ka, ja, ia) = v[2];
                    // P3's values for an empty cell (p3_main_impl.hpp).
                    re_c(ka, ja, ia) = v[4] > 0.0 ? v[3] / v[4] * um_per_r : 10.0;
                    re_r(ka, ja, ia) = v[6] > 0.0 ? v[5] / v[6] * um_per_r : 500.0;
                }
            });
    }

    /* Surface liquid precipitation over the coupling step just run.

    The flux is the mean over that step, and is held until the next one, which
    is what land reads every model step in between. precip_liq_surf_mass is
    accumulated here and turned into a mean rate over the output interval by the
    caller, as P3 does. Topography is honoured by reading the face below the
    first gridbox above topo, the level P3 takes its surface flux from. */
    void
    collect_surface_precip(Core::State& state, const double interval_s) {
        const auto nz = lnz, nx = lnx, ny = lny, h = halo;
        const bool average_y = average_over_y;
        // sum(condensate_mass * xi) is in units of MASS0; per unit horizontal
        // area and per second gives kg m-2 s-1.
        const double to_flux = dlc::MASS0 / (cell_area_m2 * interval_s);

        const auto mass = bottom_mass;
        const auto topo = state.get_field<2>("topo").get_device_data();
        auto flux = state.get_field<2>("precip_liq_surf_flux").get_mutable_device_data();
        auto acc = state.get_field<2>("precip_liq_surf_mass").get_mutable_device_data();

        Kokkos::parallel_for("cleo_surface_precip",
            Kokkos::RangePolicy<ExecSpace>(0, nx),
            KOKKOS_LAMBDA(const size_t i) {
                const auto ia = h + i;
                const auto surface_flux = [&](const size_t j) {
                    const int k = static_cast<int>(topo(h + j, ia)) + 1 - static_cast<int>(h);
                    const auto ks = static_cast<size_t>(
                        Kokkos::max(0, Kokkos::min(k, static_cast<int>(nz) - 1)));
                    return mass(ks + nz * (i + nx * j)) * to_flux;
                };
                double sum = 0.0;
                if (average_y) {
                    for (size_t j = 0; j < ny; ++j) {
                        sum += surface_flux(j);
                    }
                }
                const double inv = average_y ? 1.0 / static_cast<double>(ny) : 1.0;
                for (size_t j = 0; j < ny; ++j) {
                    const double f = average_y ? sum * inv : surface_flux(j);
                    flux(h + j, ia) = f;
                    acc(h + j, ia) += f * interval_s;
                }
            });
        Kokkos::deep_copy(bottom_mass, 0.0);
    }

    // Turns the accumulated precip_liq_surf_mass [kg m-2] into a mean rate.
    void
    average_surface_precip(Core::State& state, const double interval_s) const {
        const double inv = 1.0 / interval_s;
        auto acc = state.get_field<2>("precip_liq_surf_mass").get_mutable_device_data();
        Kokkos::parallel_for("cleo_average_precip",
            Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(
                {0, 0}, {static_cast<long>(acc.extent(0)), static_cast<long>(acc.extent(1))}),
            KOKKOS_LAMBDA(const long j, const long i) { acc(j, i) *= inv; });
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

    // precip_liq_surf_mass follows P3: accumulated in kg m-2, divided on the
    // step before output, zeroed on the step after.
    const double output_interval_s;
    double next_output_s = -1.0;
    // Coupling time summed into the accumulator; see average_precip_if_output_step.
    double precip_window_s = 0.0;
    bool need_reset_precip = false;

    Impl(const Utils::ConfigurationManager& config,
        const Core::Grid& grid,
        Core::HaloExchanger& halo_exchanger)
        : config(config), grid(grid), halo_exchanger(halo_exchanger),
          output_interval_s(config.get_value<double>("simulation.output_interval_s", 600.0)) {}

    void
    couple(Core::State& state) {
        auto& sdm = *cleo->sdm;
        const auto t_mdl = t_cleo;
        const auto t_next = t_mdl + sdm.get_couplstep();

        if (t_next != sdm.next_couplstep(t_mdl)) {
            throw std::runtime_error("CLEO is out of sync with the coupling: t_mdl " +
                                     std::to_string(t_mdl) + " + couplstep gives " +
                                     std::to_string(t_next) +
                                     " but the next coupling boundary is " +
                                     std::to_string(sdm.next_couplstep(t_mdl)));
        }
        t_cleo = t_next;

        cleo->send_dynamics(state);
        sdm.at_start_step(t_mdl, cleo->gbxs, cleo->allsupers);
        sdm.run_step(t_mdl, t_next, cleo->gbxs.view_device(), cleo->allsupers);
        cleo->receive_dynamics(state, halo_exchanger);
        cleo->diagnose_hydrometeors(state);
        const double couplstep_s = step2realtime(sdm.get_couplstep());
        cleo->collect_surface_precip(state, couplstep_s);
        precip_window_s += couplstep_s;
    }

    /* Output is decided in main.cpp by the accumulated model time, and so is
    this, with the same arithmetic. Counting steps instead -- as P3 does -- drifts
    from it: time_ += 0.0125 falls just short of 10 s after 800 steps, output
    comes on step 801, and by then the accumulator has already been averaged and
    zeroed, so the file holds one coupling step's precipitation, undivided.

    The same drift decides which side of an output a coupling on the boundary
    falls, so a window can hold one coupling step more or less than
    output_interval_s. Dividing by the coupling time actually accumulated keeps
    the result a mean rate; dividing by the interval, as P3 does, made one
    window in a test 10% low. */
    void
    average_precip_if_output_step(Core::State& state, const VVM::Real dt) {
        if (next_output_s < 0.0) {
            // As main.cpp, which also covers a restart.
            next_output_s =
                (std::floor(state.get_time() / output_interval_s) + 1.0) * output_interval_s;
        }
        const VVM::Real time_after_step = state.get_time() + dt;
        if (time_after_step < next_output_s) {
            return;
        }
        next_output_s += output_interval_s;
        if (precip_window_s > 0.0) {
            cleo->average_surface_precip(state, precip_window_s);
        }
        precip_window_s = 0.0;
        need_reset_precip = true;
    }

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
        // Radiation runs before CLEO in a step, so without this its first call
        // would see no cloud whatever the initial superdroplets hold.
        cleo->diagnose_hydrometeors(state);
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

    /* The fields P3 would otherwise provide, under P3's names and units, so
    RRTMGP (qc, nc, qi, effective radii) and land (surface precipitation) read
    CLEO's without knowing which scheme is running. Both schemes writing them
    would silently mix two microphysics, hence the check. */
    if (config.get_value<bool>("physics.p3.enable_p3", false)) {
        throw std::runtime_error(
            "physics.cleo.enable_cleo and physics.p3.enable_p3 are both true; they write "
            "the same hydrometeor and precipitation fields, so enable only one.");
    }
    const auto add3 = [&](const char* name, const char* units, const char* long_name) {
        if (!state.has_field(name)) {
            state.add_field<3>(name,
                shape,
                Core::FieldMetadata{Core::GridStaggering::Centered, units, long_name});
        }
    };
    add3("qc", "kg kg-1", "cloud liquid water mass mixing ratio");
    add3("qr", "kg kg-1", "rain water mass mixing ratio");
    add3("nc", "kg-1", "cloud droplet number mixing ratio");
    // No ice in CLEO: stays zero, present because RRTMGP reads it.
    add3("qi", "kg kg-1", "total ice mass mixing ratio");
    add3("diag_eff_radius_qc", "um", "cloud droplet effective radius");
    add3("diag_eff_radius_qr", "um", "rain drop effective radius");
    add3("diag_eff_radius_qi", "um", "ice particle effective radius");

    const std::array<int, 2> sfc{grid.get_local_total_points_y(), grid.get_local_total_points_x()};
    const auto add2 = [&](const char* name, const char* long_name) {
        if (!state.has_field(name)) {
            state.add_field<2>(name,
                sfc,
                Core::FieldMetadata{Core::GridStaggering::Surface, "kg m-2 s-1", long_name});
        }
    };
    add2("precip_liq_surf_mass", "time-averaged surface liquid precipitation flux");
    add2("precip_liq_surf_flux", "surface liquid precipitation flux");
    // Zero, for land, which adds liquid and ice.
    add2("precip_ice_surf_mass", "time-averaged surface ice precipitation flux");
    add2("precip_ice_surf_flux", "surface ice precipitation flux");
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
    auto& impl = *impl_;
    if (impl.need_reset_precip) {
        Kokkos::deep_copy(
            state.get_field<2>("precip_liq_surf_mass").get_mutable_device_data(), 0.0);
        impl.need_reset_precip = false;
    }

    // Every model step, not only coupling ones, so the output step is not missed.
    if (state.get_time() + 0.5 * dt >= impl.next_couple_s) {
        impl.next_couple_s += step2realtime(impl.cleo->sdm->get_couplstep());
        impl.couple(state);
    }
    impl.average_precip_if_output_step(state, dt);
}

void
CLEO_Interface::finalize() {
    if (impl_ && impl_->cleo) {
        impl_->cleo->sdm->after_timestepping();
    }
}

} // namespace Physics
} // namespace VVM
