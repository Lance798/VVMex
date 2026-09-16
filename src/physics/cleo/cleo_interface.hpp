#ifndef CLEO_INTERFACE_HPP
#define CLEO_INTERFACE_HPP

#include <memory>

#include "core/Grid.hpp"
#include "core/HaloExchanger.hpp"
#include "core/State.hpp"
#include "core/vvm_types.hpp"
#include "utils/ConfigurationManager.hpp"

namespace VVM {
namespace Physics {

class CLEO_Interface {
public:
    CLEO_Interface(const VVM::Utils::ConfigurationManager& config,
                   const VVM::Core::Grid& grid,
                   Core::HaloExchanger& halo_exchanger,
                   Core::State& state);

    ~CLEO_Interface();

    CLEO_Interface(const CLEO_Interface&) = delete;
    CLEO_Interface& operator=(const CLEO_Interface&) = delete;

    void initialize(VVM::Core::State& state);
    void run(VVM::Core::State& state, const VVM::Real dt);
    void finalize();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace Physics
} // namespace VVM

#endif // CLEO_INTERFACE_HPP
