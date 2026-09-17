"""
Generate the binary input files CLEO needs, from the VVMex configuration itself.

  <init_gbx_path>     gridbox boundaries
  <init_supers_path>  initial superdroplets
  cleo_config.yaml    written beside them; see "About the yaml" below

Everything is read from the `physics.cleo` block of the VVMex JSON you are going
to run, and the two binaries are written to the paths that block names. There is
nothing to keep in sync by hand and nothing to edit in a second file: point this
at a config, run it, then run the model with the same config.

Two VVMex-specific traps this exists to avoid:

  * grid.nz is the count of PHYSICAL levels; n_halo_cells more are allocated on
    each side. So nz needs no "-2" here -- but the surface is not at array index
    0 either. Initializer.cpp sets z_up(h-1) = 0, so physical cell k spans
    [z_up(k-1), z_up(k)] for k = h .. nz+h-1, and the nz+1 boundaries CLEO wants
    are z_up[h-1 : nz+h]. Off by one level and every field is shifted, which
    looks like odd behaviour near the boundaries while the interior stays
    plausible.

  * grid.dz1 != grid.dz means the column is stretched by z -> z*(CZ1 + CZ2*z).
    Handing CLEO a uniform grid in that case puts the gridboxes somewhere the
    model's levels are not. The stretch is rebuilt here with the model's own
    formula, and the DOMAIN constant it needs is read out of Initializer.cpp
    rather than copied, so this follows the model if that ever changes.

About the yaml: it is a derived artifact, not a place to edit. cleopy reads
nspacedims out of a CLEO config file, and CLEO's own RequiredConfigParams wants
the inputfiles/outputdata/domain/timesteps sections, so one has to exist. Every
value in it comes from the JSON; regenerating overwrites it.

Thermodynamic binaries are deliberately not generated: VVMex supplies pressure,
temperature and winds through the coupling, which is why CLEO is built with
CLEO_COUPLED_DYNAMICS=null (externals/cleo/CMakeLists.txt).

Usage:
    .venv/bin/python tools/cleo/gen_cleo_inputs.py -c tests/configs/2dbubble.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


class NumberSampledRadiiGen:
    """Draw radii from the number distribution itself, so every superdroplet can
    carry the SAME multiplicity -- Shima et al. (2009): "we impose a common initial
    multiplicity". This is the opposite convention to cleopy's SampleLog10RadiiGen,
    which spaces radii evenly in log10(r) and puts the distribution into xi instead.
    Radii outside rspan are rejected and redrawn, so rspan truncates the physical
    distribution here rather than merely windowing the sampling."""

    def __init__(self, geomeans, geosigs, scalefacs, rspan, seed):
        self.geomeans = np.asarray(geomeans, dtype=float)
        self.geosigs = np.asarray(geosigs, dtype=float)
        w = np.asarray(scalefacs, dtype=float)
        self.weights = w / w.sum()
        self.rspan = rspan
        self.rng = np.random.default_rng(seed)

    def __call__(self, nsupers):
        out = np.empty(int(nsupers))
        n = 0
        while n < out.size:
            need = out.size - n
            m = self.rng.choice(len(self.weights), size=need, p=self.weights)
            r = self.rng.lognormal(np.log(self.geomeans[m]), np.log(self.geosigs[m]))
            r = r[(r >= self.rspan[0]) & (r <= self.rspan[1])]
            take = min(r.size, need)
            out[n:n + take] = r[:take]
            n += take
        return out


class UniformProbDist:
    """Equal multiplicity for every superdroplet. The size distribution lives entirely
    in the sampled radii (see NumberSampledRadiiGen), not in xi."""

    def __call__(self, radii, totxi):
        radii = np.asarray(radii)
        # cleopy calls this for every gridbox, including the ones above the seeding
        # limit that get no superdroplets at all; 1.0/0 there is a ZeroDivisionError
        # rather than an empty result.
        if radii.size == 0:
            return np.empty(0)
        return np.full(radii.shape, 1.0 / radii.size)


def die(msg):
    sys.exit(f"gen_cleo_inputs: {msg}")


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        die(f"cannot read {path}: {e}")


def need(node: dict, key: str, where: str):
    """Fetch a required key, naming the block it is missing from.

    Nothing here silently defaults: a missing knob would otherwise be a plausible
    number nobody chose, which is the worst kind of input to a physics run."""
    if key not in node:
        die(f"{where}.{key} is missing from the configuration")
    return node[key]


def seed_limit(value):
    """Upper bound below which gridboxes get superdroplets; null means all of them.

    cleopy's nsupers_at_domain_base tests `gbx_zupper <= zlim`, so infinity is the
    honest spelling of "no limit" and needs no special case downstream. Note the
    test is on the gridbox UPPER bound and is inclusive: a limit of 4000 m seeds
    the level that ends exactly at 4000 m."""
    return float("inf") if value is None else float(value)


def read_grid(cfg: dict) -> dict:
    g = cfg.get("grid")
    if g is None:
        die("the configuration has no 'grid' block")
    out = {}
    for key, cast in (("nx", int), ("ny", int), ("nz", int), ("n_halo_cells", int),
                      ("dx", float), ("dy", float), ("dz", float), ("dz1", float)):
        out[key] = cast(need(g, key, "grid"))
    out["h"] = out.pop("n_halo_cells")
    out["vtype"] = g.get("vertical_coordinate_type", "default")
    out["rcemip_path"] = g.get(
        "rcemip_grid_data_path",
        "./rundata/initial_conditions/profiles/snd_rcemip_anal300_v3.txt")
    return out


def parse_initializer_domain(initializer: Path) -> float:
    """Pull DOMAIN out of Initializer.cpp.

    It is a magic constant in the C++ (`VVM::Real DOMAIN = real(15000.);`) rather
    than a config key, and it sets the vertical stretching. Reading it from the
    source means this script follows the model if it ever changes, instead of
    carrying a copy that quietly goes stale."""
    try:
        text = initializer.read_text()
    except OSError as e:
        die(f"cannot read {initializer}: {e}")
    m = re.search(r"\bDOMAIN\s*=\s*real\(\s*([0-9eE.+\-]+)\s*\)\s*;", text)
    if not m:
        die(f"could not find 'DOMAIN = real(...)' in {initializer}")
    return float(m.group(1))


def vertical_boundaries(g: dict, domain: float) -> np.ndarray:
    """The nz+1 physical level boundaries in metres, rebuilt as Initializer.cpp does."""
    h, nz, dz, dz1 = g["h"], g["nz"], g["dz"], g["dz1"]
    ntot = nz + 2 * h
    z_up = np.zeros(ntot, dtype=float)
    vtype = g["vtype"]

    if vtype == "default":
        # z_up(h-1) = 0, then a uniform dz ladder, then the whole column mapped
        # through z -> z*(CZ1 + CZ2*z). With dz1 == dz, CZ2 is 0 and CZ1 is 1, so
        # the map is the identity and the grid stays uniform.
        z_up[h - 1] = 0.0
        for k in range(h, ntot):
            z_up[k] = z_up[k - 1] + dz
        cz2 = (dz - dz1) / (dz * (domain - dz))
        cz1 = 1.0 - cz2 * domain
        for k in range(h - 1, ntot):
            z_up[k] = z_up[k] * (cz1 + cz2 * z_up[k])

    elif vtype == "rcemip":
        # Read the same two-column profile the model reads, so the levels agree by
        # construction rather than by a matching formula.
        src = (REPO / g["rcemip_path"]).resolve()
        if not src.is_file():
            die(f"rcemip profile not found: {src}")
        rows = src.read_text().splitlines()[1:]  # first line is a header
        want = ntot - 2 * h  # == nz, the levels the model reads from this file
        if len(rows) < want:
            die(f"{src} has {len(rows)} data rows, need {want}")
        z_up[h - 1] = 0.0
        for i, k in enumerate(range(h, ntot - h)):
            z_up[k] = float(rows[i].split()[0])  # column 0 is zz (= z_up)

    else:
        # The taiwanvvm branch redistributes levels through a KT/KT1 loop that
        # depends on intermediate state this script does not reproduce. Guessing
        # would put CLEO's gridboxes somewhere the model's levels are not, which
        # is exactly the failure this script exists to prevent.
        die(f"grid.vertical_coordinate_type = '{vtype}' is not supported; "
            "only 'default' and 'rcemip' are rebuilt faithfully here")

    bounds = z_up[h - 1: nz + h]
    if bounds.size != nz + 1:
        die(f"internal error: {bounds.size} boundaries for {nz} levels")
    if not np.all(np.diff(bounds) > 0):
        die("vertical boundaries are not strictly increasing")
    return bounds


def model_grid(g: dict, domain: float) -> dict:
    nz, nx, ny = g["nz"], g["nx"], g["ny"]
    if nx == 1 and ny > 1:
        # CLEO's 2-D cartesian domain is z-x, so a y-only VVMex run has no
        # matching CLEO layout; swapping the axes would silently transpose the
        # coupling.
        die("nx=1 with ny>1 has no CLEO equivalent (CLEO's 2-D domain is z-x)")
    nspacedims = 3 if ny > 1 else (2 if nx > 1 else 1)

    zgrid = vertical_boundaries(g, domain)
    xgrid = [0.0, nx * g["dx"], g["dx"]]
    if ny > 1:
        ygrid = [0.0, ny * g["dy"], g["dy"]]
    else:
        # A run with no y extent still needs a gridbox thickness: it sets the
        # gridbox volume and therefore how many real droplets each superdroplet
        # stands for. dy keeps the cells the model's own shape.
        ygrid = np.array([0.0, g["dy"]])

    return {"nz": nz, "nx": nx, "ny": ny, "nspacedims": nspacedims,
            "zgrid": zgrid, "xgrid": xgrid, "ygrid": ygrid,
            "ngbxs": nz * nx * max(ny, 1), "ztop": float(zgrid[-1])}


def read_settings(cfg: dict, grid: dict) -> dict:
    """Everything CLEO-specific, straight out of physics.cleo."""
    c = cfg.get("physics", {}).get("cleo")
    if c is None:
        die("the configuration has no 'physics.cleo' block")

    # ngbxs is in the JSON because the C++ side reads it; recomputing it here and
    # comparing is the cheap way to catch a config edited for one grid and run on
    # another, which otherwise surfaces as GbxBoundsFromBinary rejecting the file.
    declared = int(need(c, "ngbxs", "physics.cleo"))
    if declared != grid["ngbxs"]:
        die(f"physics.cleo.ngbxs is {declared} but grid.nz*nx*ny is {grid['ngbxs']}")

    ts = need(c, "timesteps_s", "physics.cleo")
    sd = need(c, "superdroplets", "physics.cleo")
    ae = need(c, "aerosol", "physics.cleo")
    mp = need(c, "microphysics", "physics.cleo")
    out = need(c, "output", "physics.cleo")
    cond = need(mp, "condensation", "physics.cleo.microphysics")

    s = {
        "gbx_path": resolve(need(c, "init_gbx_path", "physics.cleo")),
        "supers_path": resolve(need(c, "init_supers_path", "physics.cleo")),
        "couplstep": float(need(ts, "coupling", "physics.cleo.timesteps_s")),
        "condtstep": float(need(ts, "condensation", "physics.cleo.timesteps_s")),
        "colltstep": float(need(ts, "collision", "physics.cleo.timesteps_s")),
        "motiontstep": float(need(ts, "motion", "physics.cleo.timesteps_s")),
        "obstep": float(need(ts, "observation", "physics.cleo.timesteps_s")),
        "t_end": float(cfg.get("simulation", {}).get("total_time_s", 0.0)),

        "npergbx": int(need(sd, "per_gridbox", "physics.cleo.superdroplets")),
        "zlim": seed_limit(need(sd, "seed_below_z_m", "physics.cleo.superdroplets")),
        "rmin": float(need(sd, "radius_min_m", "physics.cleo.superdroplets")),
        "rmax": float(need(sd, "radius_max_m", "physics.cleo.superdroplets")),
        "numconc_tolerance": float(need(sd, "numconc_tolerance", "physics.cleo.superdroplets")),
        "common_xi": bool(need(sd, "common_multiplicity", "physics.cleo.superdroplets")),
        "wet_scale": float(need(sd, "wet_scale", "physics.cleo.superdroplets")),
        "seed": int(need(sd, "seed", "physics.cleo.superdroplets")),

        "geomeans": [float(v) for v in need(ae, "geometric_mean_radius_m", "physics.cleo.aerosol")],
        "geosigs": [float(v) for v in need(ae, "geometric_stddev", "physics.cleo.aerosol")],
        "scalefacs": [float(v) for v in need(ae, "number_conc_per_m3", "physics.cleo.aerosol")],

        "cond": cond,
        "coaleff": float(need(mp, "coalescence_efficiency", "physics.cleo.microphysics")),
        "nfrags": float(need(mp, "breakup_nfrags", "physics.cleo.microphysics")),

        "setup_filename": resolve(need(out, "setup_filename", "physics.cleo.output")),
        "zarrbasedir": resolve(need(out, "zarrbasedir", "physics.cleo.output")),
        "maxchunk": int(need(out, "maxchunk", "physics.cleo.output")),

        "num_threads": int(c.get("kokkos_num_threads", 0)),
    }

    n = len(s["geomeans"])
    if not (len(s["geosigs"]) == len(s["scalefacs"]) == n) or n == 0:
        die("physics.cleo.aerosol: the three mode arrays must be non-empty and the same length")
    if s["t_end"] <= 0.0:
        die("simulation.total_time_s must be positive; CLEO's T_END is taken from it")
    check_timesteps(s)
    return s


def resolve(p) -> Path:
    """Paths in the config are relative to the repository, not the caller's cwd."""
    q = Path(str(p))
    return q if q.is_absolute() else (REPO / q).resolve()


def check_timesteps(s: dict) -> None:
    """CLEO enforces max(CONDTSTEP, COLLTSTEP) <= min(COUPLTSTEP, OBSTSTEP, MOTIONTSTEP)
    -- the microphysics substeps must be the smallest steps of all. Catch it here
    rather than letting CLEO abort at run time (libs/initialise/timesteps.cpp)."""
    steps = {"condensation": s["condtstep"], "collision": s["colltstep"],
             "motion": s["motiontstep"], "coupling": s["couplstep"],
             "observation": s["obstep"], "T_END": s["t_end"]}
    # realtime2step rounds to units of 1/100 s; anything smaller collapses to 0
    for name, v in steps.items():
        if round(v * 100) == 0:
            die(f"{name}={v}s rounds to 0 model steps (CLEO's resolution is 0.01 s)")

    maxsub = max(s["condtstep"], s["colltstep"])
    minstep = min(s["couplstep"], s["obstep"], s["motiontstep"])
    if minstep < maxsub:
        die(f"invalid timestep hierarchy: max(condensation, collision) = {maxsub}s "
            f"must not exceed min(coupling, motion, observation) = {minstep}s")


def write_config(path: Path, grid: dict, s: dict, consts: Path, src_json: Path) -> None:
    # No initnsupers key: maxnsupers is the initial count exactly, which is what
    # InitAllSupersFromBinary requires. Headroom for superdroplets created during
    # the run would need initnsupers here AND InitSupersFromBinary on the C++ side.
    kokkos = (f"kokkos_settings:\n  num_threads: {s['num_threads']}\n\n"
              if s["num_threads"] > 0 else "")
    c = s["cond"]
    path.write_text(
        f"""# GENERATED by tools/cleo/gen_cleo_inputs.py -- do not edit.
# Every value here comes from the physics.cleo block of:
#   {src_json}
# Edit that instead and re-run the generator.

{kokkos}domain:
  nspacedims: {grid["nspacedims"]}
  ngbxs: {grid["ngbxs"]}
  maxnsupers: {s["maxnsupers"]}

timesteps:
  CONDTSTEP: {s["condtstep"]}
  COLLTSTEP: {s["colltstep"]}
  MOTIONTSTEP: {s["motiontstep"]}
  COUPLTSTEP: {s["couplstep"]}
  OBSTSTEP: {s["obstep"]}
  T_END: {s["t_end"]}

inputfiles:
  constants_filename: {consts}
  grid_filename: {s["gbx_path"]}

initsupers:
  type: frombinary
  initsupers_filename: {s["supers_path"]}

outputdata:
  setup_filename: {s["setup_filename"]}
  zarrbasedir: {s["zarrbasedir"]}
  maxchunk: {s["maxchunk"]}

microphysics:
  condensation:
    do_alter_thermo: {str(c["do_alter_thermo"]).lower()}
    maxniters: {c["maxniters"]}
    MINSUBTSTEP: {c["MINSUBTSTEP"]}
    rtol: {c["rtol"]}
    atol: {c["atol"]}
  coalescence:
    constcoaleff:
      coaleff: {s["coaleff"]}
  breakup:
    constnfrags:
      nfrags: {s["nfrags"]}
""")


def write_grid(paths: dict, grid: dict, figures, figpath: Path) -> None:
    from cleopy import geninitconds
    geninitconds.generate_gridbox_boundaries(
        str(paths["grid"]), grid["zgrid"], grid["xgrid"], grid["ygrid"],
        str(paths["constants"]), isprintinfo=True, isfigures=figures,
        # A Path, not a str: cleopy builds the figure filename with the "/"
        # operator. read_gbxboundaries happens to survive a str because its right
        # operand is a Path, but read_initsuperdrops has str on both sides and
        # raises TypeError -- after the binaries are already written.
        savefigpath=figpath)


def compute_nsupers(paths: dict, s: dict) -> dict:
    """{gbxindex: count}, seeded only below the limit. Needs the grid file to exist."""
    from cleopy.initsuperdropsbinary_src import crdgens
    nsupers = crdgens.nsupers_at_domain_base(
        str(paths["grid"]), str(paths["constants"]), s["npergbx"], s["zlim"])
    total = sum(nsupers.values())
    seeded = sum(v > 0 for v in nsupers.values())
    # Worth stating outright: a limit above the domain top and a limit of null are
    # the same run, and a mistyped 40000 for 4000 otherwise shows up only as a
    # larger number that nobody double-checks.
    where = ("every gridbox -- seed_below_z_m is at or above the domain top"
             if seeded == len(nsupers) else f"{seeded} seeded gridboxes")
    print(f"CLEO  : {total} superdroplets in {where}")
    if total > 1_000_000:
        print(f"\n  WARNING: {total} superdroplets. SDM costs two to three orders of\n"
              f"  magnitude more than a bulk scheme; verify the index mapping and unit\n"
              f"  conversion at a coarser resolution before running this.\n")
    return nsupers


def write_initsupers(paths: dict, s: dict, grid: dict, nsupers: dict,
                     figures, figpath: Path) -> None:
    from cleopy import geninitconds
    from cleopy.initsuperdropsbinary_src import (attrsgen, crdgens, dryrgens,
                                                 probdists, rgens)

    # ScaledRadiiGen divides: dryradius = radius / wet_scale. rspan and the
    # spectrum modes are scaled with it so the DRY population is unchanged and
    # only the water content differs.
    sf = s["wet_scale"]
    means = [g * sf for g in s["geomeans"]]
    span = [s["rmin"] * sf, s["rmax"] * sf]
    if s["common_xi"]:
        radiigen = NumberSampledRadiiGen(means, s["geosigs"], s["scalefacs"], span, s["seed"])
        xiprobdist = UniformProbDist()
    else:
        radiigen = rgens.SampleLog10RadiiGen(span)
        xiprobdist = probdists.LnNormal(means, s["geosigs"], s["scalefacs"])
    dryradiigen = dryrgens.ScaledRadiiGen(sf)
    coord3gen = crdgens.SampleCoordGen(True)                                  # z
    coord1gen = crdgens.SampleCoordGen(True) if grid["nspacedims"] >= 2 else None  # x
    coord2gen = crdgens.SampleCoordGen(True) if grid["nspacedims"] >= 3 else None  # y

    # xi_by_pressure is left off on purpose: it needs pressure read from
    # thermodynamics binaries, which this setup does not produce (VVMex supplies
    # the thermodynamic state through the coupling instead).
    initattrsgen = attrsgen.AttrsGenerator(radiigen, dryradiigen, xiprobdist,
                                           coord3gen, coord1gen, coord2gen)

    geninitconds.generate_initial_superdroplet_conditions(
        initattrsgen, str(paths["initsupers"]), str(paths["config"]),
        str(paths["constants"]), str(paths["grid"]), nsupers,
        np.sum(s["scalefacs"]), numconc_tolerance=s["numconc_tolerance"],
        isprintinfo=True, isfigures=figures, savefigpath=figpath, gbxs2plt=[0])


def report_domain_totals(paths: dict, s: dict) -> None:
    """Re-print the domain totals, computed in floating point.

    cleopy's own "DOMAIN SUPERDROPLETS INFO" sums xi with np.sum on a uint64
    array. Each superdroplet stands for ~1e11 real droplets, so a domain of any
    size overflows 2**64 and the number it prints is a wrapped one -- for the
    2dbubble grid it under-reports by a factor of ten, which looks exactly like
    an aerosol loading bug that is not there. The binaries are fine; only that
    diagnostic is wrong, so print a correct one next to it."""
    from cleopy.gbxboundariesbinary_src import read_gbxboundaries as rg
    from cleopy.initsuperdropsbinary_src import read_initsuperdrops as rs

    vols = np.asarray(rg.get_gbxvols_from_gridfile(
        str(paths["grid"]), constants_filename=str(paths["constants"]), isprint=False))
    attrs = rs.get_superdroplet_attributes(
        str(paths["config"]), str(paths["constants"]), str(paths["initsupers"]))
    xi = np.asarray(attrs.xi).flatten().astype(np.float64)
    seeded_conc = np.sum(s["scalefacs"]) / 1e6
    mean_conc = xi.sum() / vols.sum() / 1e6
    print(f"\ndomain-mean droplet number concentration: {mean_conc:.4g} /cm^3")
    # The two agree only when every gridbox is seeded; saying so keeps the gap
    # from reading as an aerosol loading error.
    print(f"  seeded gridboxes hold {seeded_conc:.4g} /cm^3"
          + ("" if np.isinf(s["zlim"]) or mean_conc >= seeded_conc * 0.999
             else "; the rest of the domain is empty"))


def write_back_max_total(config_path: Path, total: int) -> bool:
    """Record the superdroplet count in the config the model will read.

    InitAllSupersFromBinary demands maxnsupers exactly equal the number of
    superdroplets in the binary -- it throws on either side of that -- and only
    this script knows the number: it falls out of per_gridbox times however many
    gridboxes sit below seed_below_z_m. Rather than have the model re-derive it
    (two implementations that must agree), the value is written back here, so the
    config stays the single source of truth. Rewritten in place, preserving key
    order and the __ documentation entries."""
    import collections

    d = json.loads(config_path.read_text(),
                   object_pairs_hook=collections.OrderedDict)
    sd = d["physics"]["cleo"]["superdroplets"]
    if sd.get("max_total") == total:
        return False
    sd["max_total"] = total
    config_path.write_text(json.dumps(d, indent=4, ensure_ascii=False) + "\n")
    return True


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-c", "--config", type=Path,
                   default=REPO / "tests" / "configs" / "2dbubble.json",
                   help="the VVMex JSON to read; every setting comes from it")
    p.add_argument("--path2cleo", type=Path, default=None,
                   help="CLEO source root, for cleopy and libs/cleoconstants.hpp "
                        "(default: build/_deps/cleo-src, then ../CLEO)")
    p.add_argument("--save-figures", action="store_true")
    return p.parse_args()


def cleo_from_cmake_cache() -> Path:
    """The CLEO the build is actually configured against.

    build/_deps/cleo-src is FetchContent's copy, which is stale whenever
    FETCHCONTENT_SOURCE_DIR_CLEO points somewhere else. Generating the binaries
    against a different cleoconstants.hpp than the model links means the
    de-dimensionalisation silently disagrees, so take the answer from the cache
    rather than guessing."""
    cache = REPO / "build" / "CMakeCache.txt"
    if not cache.is_file():
        return None
    m = re.search(r"^CLEO_SOURCE_DIR:\w+=(.+)$", cache.read_text(), re.M)
    return Path(m.group(1).strip()) if m else None


def find_cleo(explicit) -> Path:
    if explicit is not None:
        candidates = [explicit]
    else:
        candidates = [c for c in (cleo_from_cmake_cache(),
                                  REPO / "build" / "_deps" / "cleo-src",
                                  REPO.parent / "CLEO") if c is not None]
    for c in candidates:
        if (c / "cleopy").is_dir() and (c / "libs" / "cleoconstants.hpp").is_file():
            return c.resolve()
    die("no CLEO checkout found (looked for cleopy/ and libs/cleoconstants.hpp in "
        + ", ".join(str(c) for c in candidates) + "); pass --path2cleo")


def main():
    args = parse_args()
    config_path = args.config.resolve()
    cfg = load_json(config_path)

    # cleopy is not installed in the venv; it ships inside the CLEO checkout.
    path2cleo = find_cleo(args.path2cleo)
    sys.path.insert(0, str(path2cleo))

    g = read_grid(cfg)
    domain = parse_initializer_domain(REPO / "src" / "core" / "Initializer.cpp")
    grid = model_grid(g, domain)
    s = read_settings(cfg, grid)

    print(f"config: {config_path}")
    print(f"model : nx={g['nx']} ny={g['ny']} nz={g['nz']} halo={g['h']} "
          f"dx={g['dx']} dy={g['dy']} dz={g['dz']} dz1={g['dz1']} vcoord={g['vtype']}")
    print(f"CLEO  : nspacedims={grid['nspacedims']} ngbxs={grid['ngbxs']}  (from {path2cleo})")
    print(f"        z 0..{grid['ztop']} m ({grid['nz']} levels), "
          f"x 0..{g['nx'] * g['dx']} m, y 0..{g['ny'] * g['dy']} m")

    share = s["gbx_path"].parent
    figpath = share / "figs"
    for d in (share, s["supers_path"].parent, s["setup_filename"].parent,
              Path(s["zarrbasedir"]).parent, figpath):
        d.mkdir(parents=True, exist_ok=True)

    paths = {"constants": path2cleo / "libs" / "cleoconstants.hpp",
             "config": share / "cleo_config.yaml",
             "grid": s["gbx_path"], "initsupers": s["supers_path"]}
    figures = [False, args.save_figures]

    # Order is forced by the dependencies: nsupers_at_domain_base reads the grid
    # file, maxnsupers in the yaml has to know the resulting superdroplet count,
    # and generate_initial_superdroplet_conditions reads nspacedims from the yaml.
    for f in (paths["grid"], paths["initsupers"]):
        f.unlink(missing_ok=True)

    write_grid(paths, grid, figures, figpath)
    nsupers = compute_nsupers(paths, s)

    total = sum(nsupers.values())
    if total == 0:
        die(f"no gridbox has its upper bound below physics.cleo.superdroplets."
            f"seed_below_z_m = {s['zlim']} m; the lowest level ends at "
            f"{grid['zgrid'][1]:.0f} m. Use null to seed every gridbox.")
    # CLEO's binary format stores each variable's byte offset as a 32-bit unsigned
    # int (VarMetadata::b0 in libs/initialise/readbinary.hpp), so past about 97
    # million superdroplets the offsets of the last variables wrap around 2^32 and
    # point back into the middle of xi and radius. The C++ reader here derives the
    # offsets in 64 bits instead of trusting the stored ones, so such a file reads
    # correctly -- but the value written into the file is still wrapped, so any
    # other consumer that trusts it will be silently wrong. Say so rather than
    # leaving it to be discovered.
    B0_LIMIT = 2 ** 32
    if 44 * total + 4096 >= B0_LIMIT:
        print(f"\n  NOTE: {total} superdroplets puts the last variables' byte offsets past "
              f"{B0_LIMIT} bytes,\n  which this format records in 32 bits. Reading this file "
              f"needs the 64-bit offset\n  derivation in readbinary.cpp; a stock CLEO would "
              f"read the coordinates as garbage.\n")

    # max_total is derived, never read as an input: it is per_gridbox times the
    # number of seeded gridboxes, and InitAllSupersFromBinary demands it equal the
    # binary's count exactly. Validating against whatever is already in the JSON
    # would let a stale value block the very run that refreshes it, which is what
    # happens the moment per_gridbox or seed_below_z_m changes.
    s["maxnsupers"] = total

    if write_back_max_total(config_path, total):
        print(f"\nupdated physics.cleo.superdroplets.max_total = {total} in\n  {config_path}")

    write_config(paths["config"], grid, s, paths["constants"], config_path)
    write_initsupers(paths, s, grid, nsupers, figures, figpath)
    report_domain_totals(paths, s)

    print("\nwrote:")
    for k in ("grid", "initsupers", "config"):
        print(f"  {paths[k]}")
    print("\nthe model reads the first two straight from physics.cleo of the same JSON.")


if __name__ == "__main__":
    main()
