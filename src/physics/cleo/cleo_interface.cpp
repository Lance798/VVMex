#include "cleo_interface.hpp"

#include <mpi.h>

#include <iostream>
#include <memory>

// TODO: update CLEO constants to be consistent with VVMex
#include "cleoconstants.hpp"


namespace dlc = dimless_constants;

namespace VVM {
namespace Physics {

struct CLEO_Interface::Impl {
    const Utils::ConfigurationManager& config;
    const Core::Grid& grid;
    const Core::Parameters& params;
    Core::HaloExchanger& halo_exchanger;
    Core::State& state;

    double couplstep_s = 0.0;
    double couplstep_dimless = 0.0;

    double t_model_s = 0.0;

    Impl(const Utils::ConfigurationManager& config,
         const Core::Grid& grid,
         const Core::Parameters& params,
         Core::HaloExchanger& halo_exchanger,
         Core::State& state)
        : config(config),
          grid(grid),
          params(params),
          halo_exchanger(halo_exchanger),
          state(state) {}

    void initialize() {
        couplstep_dimless = couplstep_s / dlc::TIME0;
        t_model_s = 0.0;
        if (grid.get_mpi_rank() == 0) {
            std::cout << "[CLEO] interface built (skeleton: no SDM yet)."
                      << " couplstep = " << couplstep_s << " s"
                      << " = " << couplstep_dimless << " (dimensionless, TIME0="
                      << dlc::TIME0 << "s)" << std::endl;
        }
    }
};

CLEO_Interface::CLEO_Interface(const Utils::ConfigurationManager& config,
                               const Core::Grid& grid,
                               const Core::Parameters& params,
                               Core::HaloExchanger& halo_exchanger,
                               Core::State& state)
    : impl_(std::make_unique<Impl>(config, grid, params, halo_exchanger, state)) {
    impl_->couplstep_s = config.get_value<double>("physics.cleo.coupling_step_seconds", 1.0);
}

CLEO_Interface::~CLEO_Interface() = default;

void CLEO_Interface::initialize(Core::State& state) {
    if (impl_->grid.get_mpi_rank() == 0) {
        std::cout << "[CLEO] initializing..." << std::endl;
    }
    impl_->initialize();
}

void CLEO_Interface::run(Core::State& state, const VVM::Real dt) {
}

void CLEO_Interface::finalize() {
}

} // namespace Physics
} // namespace VVM
