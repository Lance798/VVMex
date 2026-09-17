# Profiling VVMex

`source env_setup.sh` first: it puts the NVHPC copies of `nsys` (2026.2.1) and
`ncu` (2026.1.0) on PATH, ahead of the older system ones under /usr/local/cuda
(2025.3.x). On a GB10 (sm_121) use the newer pair.

## Timeline: Nsight Systems

    tools/profile/nsys_run.sh rundata/input_configs/nytest/prof_smoke.json
    nsys stats --report cuda_gpu_kern_sum --format table /tmp/vvm_prof/prof_smoke.nsys-rep

Profile a SHORT config. `prof_smoke.json` is 0.5 model seconds (40 steps), which
is ~9 s wall and a 2.7 MB report; a production run would give an unreadable one.
Set `output.bp5.existing_dataset` to `replace` or the second run aborts.

Useful reports: `cuda_gpu_kern_sum` (kernels), `cuda_gpu_mem_time_sum` (copies),
`cuda_api_sum` (launch overhead), `nvtx_sum` (if NVTX ranges exist).

Kernel names come out as `Kokkos::Impl::cuda_parallel_launch_local_memory<
ParallelFor<VVM::Dynamics::WindSolver::solve...>>` -- the VVMex name is in the
template arguments. `--format table` truncates it; use `--format csv` to see the
whole signature. (kokkos-tools' `kp_nvtx_connector.so` would put the
`parallel_for("name")` labels on the timeline instead, but it is not installed.)

## One kernel in detail: Nsight Compute

    sudo -n env PATH="$PATH" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
      ncu --kernel-name-base demangled --kernel-name regex:"WindSolver::solve" \
          --launch-skip 400 --launch-count 2 --set basic \
          ./build/vvm <config.json>

Three things that will each cost you an hour if you do not know them:

* **`--kernel-name-base demangled` is mandatory.** Every Kokkos kernel shares the
  same mangled entry point (`cuda_parallel_launch_local_memory`), so a regex on
  the default name base matches either nothing or everything.
* **ncu needs `sudo` here.** Without it: `ERR_NVGPUCTRPERM`. The permanent fix is
  `NVreg_RestrictProfilingToAdminUsers=0` in /etc/modprobe.d plus a reboot;
  passwordless sudo works today without one. Send the model's output to /tmp when
  running as root, or it will leave root-owned files in output/.
* **Always bound the launches** with `--launch-skip` / `--launch-count`. ncu
  replays each kernel several times per metric set; the wind solver alone is
  launched 200 times per timestep.

`--set basic` is the cheap set. `--set full` adds the roofline and source
counters but needs `-lineinfo` at compile time, which this build does not pass.

## Memory errors: compute-sanitizer

`submit.py --wrap` puts a tool in front of the model inside mpirun, so each rank
gets its own instance:

    ./submit.py -c <config.json> --preset spark --local --compute 2 --io 0 \
        --wrap "compute-sanitizer --tool memcheck --report-api-errors no \
                --log-file /tmp/san/san.%q{OMPI_COMM_WORLD_RANK}.log"

`--report-api-errors no` is not optional here. UCX opens its CUDA transports
before any CUDA context exists, so a plain memcheck run drowns in hundreds of
`CUDA API Error: No CUDA context is current to the calling thread` from
`uct_cuda_copy_md_open` -- all benign, none of them memory errors. Suppressing
the API-error class leaves the memory checking intact.

`%q{OMPI_COMM_WORLD_RANK}` is expanded by compute-sanitizer itself, so one
quoted --wrap string still gives one log per rank.

### Why this exists

`submit.py` used to derive VVM_BINARY from the CMake preset and overwrite any
exported value, so there was no way to get a tool between mpirun and the model.
It now honours an explicit `VVM_BINARY`, and `--wrap` / `VVM_LAUNCH_PREFIX` adds
a prefix in `core_run.sh`'s two exec sites.

### Editing core_run.sh

Both exec sites live inside `INLINE_WRAPPER='...'` (core_run.sh:145), a single-
quoted string that each rank runs through `bash -c`. **An apostrophe anywhere in
that block ends the string early** and the file stops parsing hundreds of lines
later, with an error pointing at the wrong place. No contractions in comments.

### Reading a CUDA crash

`cudaErrorIllegalAddress` reported at a `Kokkos::Cuda::fence` says where the
error was *detected*, not where it happened: CUDA errors are asynchronous and
sticky, so the culprit is any kernel launched since the previous sync. Resolve
the backtrace for the call path (the binary is non-PIE and carries .debug_line
even at -O3, so `addr2line -e build/vvm -f -C -i -p <addrs>` works directly),
then use compute-sanitizer -- or `CUDA_LAUNCH_BLOCKING=1` for a cheaper, coarser
version -- to find the kernel that actually did it.

Line numbers at -O3 are approximate because of inlining; function names and the
call order are reliable. `cmake -DVVM_CLEO_DEBUG=ON` rebuilds just the CLEO
translation unit at -O0 -g when you need exact lines there.
