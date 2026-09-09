add_library(vvm_cleo cleo_interface.cpp)

# CLEO's public headers use C++20 concepts; the rest of VVMex is C++17
# (CMakeLists.txt: set(CMAKE_CXX_STANDARD 17)). CXX_STANDARD is not a usage
# requirement, so linking CLEO::* does not bring it along -- it has to be set
# here. This is the only VVMex target built as C++20, which is why
# cleo_interface.hpp must not include any CLEO header: it is pulled in by
# Model.hpp and hence by most of the C++17 tree.
set_target_properties(vvm_cleo PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)

target_link_libraries(vvm_cleo
    PUBLIC
        vvm_core
        vvm_utils
        Kokkos::kokkos
    PRIVATE
        # PRIVATE keeps CLEO's include dirs off every consumer of vvm_cleo, so a
        # C++17 TU cannot accidentally reach a CLEO header.
        CLEO::runcleo
        CLEO::cartesiandomain
        CLEO::movement       # MoveSupersInDomain lives under cartesiandomain/movement
        CLEO::gridboxes
        CLEO::superdrops
        CLEO::initialise
        CLEO::observers
        CLEO::zarr
        CLEO::configuration
)

# The Release build already carries file/line debug info (nvc++ emits it), so
# breakpoints and backtraces work without this. What -fast -O3 does take away is
# locals: gdb reports "No locals." in every CLEO frame. Turning this on rebuilds
# only this one translation unit at -O0, which brings variables back while the
# rest of the model stays optimized.
#
#   cmake -S . -B build -DVVM_CLEO_DEBUG=ON && cmake --build build --target vvm -j
#
# Leave it OFF for anything you intend to time or trust numerically.
option(VVM_CLEO_DEBUG "Build the CLEO interface at -O0 -g for debugging" OFF)
if(VVM_CLEO_DEBUG)
    message(STATUS "VVM CLEO debug build: cleo_interface.cpp at -O0 -g")
    target_compile_options(vvm_cleo PRIVATE -O0 -g)
endif()

target_include_directories(vvm_cleo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(ENABLE_NCCL)
    target_include_directories(vvm_cleo PUBLIC ${NCCL_INCLUDE_DIR})
endif()
