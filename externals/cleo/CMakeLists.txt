message(STATUS "Fetching CLEO library...")

include(FetchContent)
FetchContent_Declare(
  cleo
  GIT_REPOSITORY https://github.com/Lance798/CLEO.git
  GIT_TAG        b92eb77
  FIND_PACKAGE_ARGS 0.69 NAMES CLEO
)

# VVMex drives CLEO itself: Model::run_step owns the time loop and calls
# SDMMethods directly, so none of CLEO's own CoupledDynamics solvers are built.
# (Valid values, from CLEO libs/CMakeLists.txt: null fromfile cvode yac numpy)
set(CLEO_COUPLED_DYNAMICS "null")
set(CLEO_DOMAIN "cartesian")
FetchContent_MakeAvailable(cleo)
