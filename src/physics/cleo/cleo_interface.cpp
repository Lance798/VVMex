#include "cleo_interface.hpp"

#include <mpi.h>

#include <algorithm>
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
#include "observers/collect_data_for_simple_dataset.hpp"
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
#include "zarr/simple_dataset.hpp"
#include "core/Field.hpp"

namespace VVM {
namespace Physics {

namespace {

using Store = FSStore;
using Dataset = SimpleDataset<Store>;

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
    const NFragments auto nfrags = ConstNFrags(5);
    const CoalBuReFlag auto coalbure_flag = TSCoalBuReFlag(RogersGKTerminalVelocity{});

    const MicrophysicalProcess auto colls =
        CoalBuRe(ts.get_collstep(), &step2realtime, collprob, nfrags, coalbure_flag);
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

    const auto collect_sddata =
        coord1 >> coord2 >> coord3 >> msol >> radius >> xi >> sdgbxindex >> sdid;
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
    Store& store) {
    const auto obsstep = ts.get_obsstep();
    const auto maxchunk = config.get_value<size_t>("physics.cleo.output.maxchunk");
    const auto ngbxs = config.get_value<std::size_t>("physics.cleo.ngbxs");

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

    /* Local gridbox layout, for the coupling loops. CLEO numbers a rank's own
    gridboxes idx = k + nz*(i + nx*j) over its OWN partition (see
    get_index_from_coordinates in cartesian_decomposition.cpp), so these are the
    LOCAL physical sizes -- using the global ones decodes every index wrongly the
    moment there is more than one rank. VVMex's arrays carry n_halo_cells ghost
    cells on each side of every dimension, so a local physical coordinate becomes
    an array index by adding halo. */
    const size_t lnz, lnx, lny, halo;

    CleoState(const Utils::ConfigurationManager& config, const Core::Grid& grid)
        : ts(timesteps_from(config)),
          store(config.get_value<std::string>("physics.cleo.output.zarrbasedir")), dataset(store),
          gbxmaps(create_cartesian_maps(config.get_value<std::size_t>("physics.cleo.ngbxs"),
              3,
              config.get_value<std::string>("physics.cleo.init_gbx_path"),
              supplied_from_grid(grid))),
          allsupers(
              create_supers(InitAllSupersFromBinary(config.get_value<std::size_t>(
                                                        "physics.cleo.superdroplets.max_total"),
                                config.get_value<std::string>("physics.cleo.init_supers_path"),
                                3),
                  gbxmaps.get_local_ngridboxes_hostcopy())),
          gbxs(create_gbxs(
              gbxmaps, InitGbxsNull(gbxmaps.get_local_ngridboxes_hostcopy()), allsupers)),
          sdm(erase_sdm(SDMMethods(ts.get_couplstep(),
              gbxmaps,
              create_microphysics(ts, config),
              create_movement(ts, gbxmaps),
              create_observer(ts, config, dataset, store)))),
          lnz(grid.get_local_physical_points_z()), lnx(grid.get_local_physical_points_x()),
          lny(grid.get_local_physical_points_y()), halo(grid.get_halo_cells()) {
        sdm->prepare_to_timestep(gbxs, allsupers);

        // The coupling loops decode CLEO's gridbox index with these three sizes;
        // if they do not multiply out to the gridbox count CLEO actually built,
        // every field would be mapped to the wrong cell silently.
        if (lnz * lnx * lny != gbxs.extent(0)) {
            throw std::runtime_error(
                "CLEO built " + std::to_string(gbxs.extent(0)) + " local gridboxes but the grid's "
                "local physical points are " + std::to_string(lnz) + "*" + std::to_string(lnx) +
                "*" + std::to_string(lny) + " = " + std::to_string(lnz * lnx * lny));
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
    receive_dynamics(Core::State& state) {
        auto th = state.get_field<3>("th").get_mutable_device_data();
        auto qvap = state.get_field<3>("qv").get_mutable_device_data();
        auto qcond = state.get_field<3>("qcond").get_mutable_device_data();
        const auto pibar = state.get_field<1>("pibar").get_device_data();

        const auto nz = lnz, nx = lnx, h = halo;
        const auto d_gbxs = gbxs.view_device();
        Kokkos::parallel_for("cleo_receive_dynamics",
            Kokkos::RangePolicy<ExecSpace>(0, d_gbxs.extent(0)),
            KOKKOS_LAMBDA(const size_t idx) {
                const auto k = idx % nz;
                const auto i = (idx / nz) % nx;
                const auto j = idx / (nz * nx);
                const auto ka = h + k, ja = h + j, ia = h + i;

                const auto& gb_state = d_gbxs(idx).state;
                th(ka, ja, ia) = gb_state.temp * dlc::TEMP0 / pibar(ka);
                qvap(ka, ja, ia) = gb_state.qvap;
                qcond(ka, ja, ia) = gb_state.qcond;
            });
    }
};

} // namespace

struct CLEO_Interface::Impl {
    const Utils::ConfigurationManager& config;
    const Core::Grid& grid;
    const Core::Parameters& params;
    Core::HaloExchanger& halo_exchanger;
    Core::State& state;
    std::unique_ptr<CleoState> cleo;
    unsigned int since_couple = 0; /**< model time accumulated since the last coupling, in CLEO steps */

    Impl(const Utils::ConfigurationManager& config,
        const Core::Grid& grid,
        const Core::Parameters& params,
        Core::HaloExchanger& halo_exchanger,
        Core::State& state)
        : config(config), grid(grid), params(params), halo_exchanger(halo_exchanger), state(state) {
    }

    void
    initialize() {
        cleo = std::make_unique<CleoState>(config, grid);

        std::cout << "[CLEO] rank " << grid.get_mpi_rank()
                  << " ready: " << cleo->gbxmaps.get_local_ngridboxes_hostcopy() << " of "
                  << config.get_value<std::size_t>("physics.cleo.ngbxs") << " gridboxes, couplstep "
                  << cleo->sdm->get_couplstep() << " (dimensionless, TIME0=" << dlc::TIME0
                  << "s), zarr at "
                  << config.get_value<std::string>("physics.cleo.output.zarrbasedir") << std::endl;

        // send initial field to cleo gridbox
        cleo->send_dynamics(state);
    }
};

CLEO_Interface::CLEO_Interface(const Utils::ConfigurationManager& config,
    const Core::Grid& grid,
    const Core::Parameters& params,
    Core::HaloExchanger& halo_exchanger,
    Core::State& state)
    : impl_(std::make_unique<Impl>(config, grid, params, halo_exchanger, state)) {
    state.add_field<3>("qcond",
        {grid.get_local_total_points_z(),
            grid.get_local_total_points_y(),
            grid.get_local_total_points_x()},
        Core::FieldMetadata{Core::GridStaggering::Centered,
            "kg kg-1",
            "total condensation mixing ratio"});
}

CLEO_Interface::~CLEO_Interface() = default;

void
CLEO_Interface::initialize(Core::State& state) {
    if (impl_->grid.get_mpi_rank() == 0) {
        std::cout << "[CLEO] initializing..." << std::endl;
    }
    impl_->initialize();
}

void
CLEO_Interface::run(Core::State& state, const VVM::Real dt) {
    auto& cleo = *impl_->cleo;
    auto& sdm = *cleo.sdm;

    // Accumulate first, then subtract what was spent: comparing before adding
    // costs one whole coupling interval on the first call and then runs at half
    // the intended rate, and resetting to 0 instead of subtracting lets the phase
    // drift whenever dt does not divide the coupling step.
    impl_->since_couple += realtime2step(dt);
    if (impl_->since_couple < sdm.get_couplstep()) {
        return;
    }
    impl_->since_couple -= sdm.get_couplstep();

    const auto t_mdl = realtime2step(state.get_time());

    cleo.send_dynamics(state);
    sdm.at_start_step(t_mdl, cleo.gbxs, cleo.allsupers);
    sdm.run_step(t_mdl, t_mdl + sdm.get_couplstep(), cleo.gbxs.view_device(), cleo.allsupers);
    cleo.receive_dynamics(state);
}

void
CLEO_Interface::finalize() {
    // Flushes the observers' final output; must run while the FSStore is alive.
    if (impl_ && impl_->cleo) {
        impl_->cleo->sdm->after_timestepping();
    }
}

} // namespace Physics
} // namespace VVM
