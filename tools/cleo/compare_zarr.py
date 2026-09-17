#!/usr/bin/env python3
"""Bit-compare two CLEO zarr stores written by runs that should agree exactly.

Single-rank and multi-rank runs chunk the SAME logical array differently
(CollectiveDataset gives each rank its own chunk), so the chunk files cannot be
diffed directly -- each array is reassembled from its chunks first.

This is only a meaningful test when collision is off. With collision on the
Kokkos random pool draws per-thread, not per-gridbox, so the two runs legitimately
diverge; see physics.cleo.microphysics.enable_collision.
"""
import json, sys, pathlib
import numpy as np

def meta(store, v):
    return json.loads((store / v / '.zarray').read_text())

def read_flat(store, v):
    """Reassemble a 2-D array from its chunk files.

    The collective writer appends each rank's buffer to a flat stream instead of
    placing it at the chunk grid position its .zarray advertises, so the chunk
    files concatenated in name order ARE the array in C order -- and reading them
    as the declared (time, space) chunks interleaves ranks and times instead.
    The stream is padded to a whole number of chunks, so trim to shape.
    """
    m = meta(store, v)
    if len(m['shape']) != 2 or 0 in m['shape']:
        return None
    nt, ns = m['shape']
    d = store / v
    files = sorted((q.name for q in d.iterdir() if q.name[0].isdigit()),
                   key=lambda t: tuple(int(x) for x in t.split('.')))
    flat = np.concatenate([np.fromfile(d / f, dtype=np.dtype(m['dtype'])) for f in files])
    return flat[:nt * ns].reshape(nt, ns)


def read(store, v, ti=None):
    """Whole array, or one time slice of a 2-D array."""
    m = meta(store, v)
    dt = np.dtype(m['dtype'])
    shape, chunks = m['shape'], m['chunks']
    if 0 in shape:
        return None
    if len(shape) == 1:
        n, cs = shape[0], chunks[0]
        out = np.empty(n, dtype=dt)
        for j in range(-(-n // cs)):
            f = store / v / f'{j}'
            if not f.exists():
                return None
            a = np.fromfile(f, dtype=dt)
            lo, hi = j * cs, min((j + 1) * cs, n)
            out[lo:hi] = a[:hi - lo]
        return out
    nt, ns = shape
    ct, cs = chunks
    ci, off = divmod(ti, ct)
    out = np.empty(ns, dtype=dt)
    for j in range(-(-ns // cs)):
        f = store / v / f'{ci}.{j}'
        if not f.exists():
            return None
        a = np.fromfile(f, dtype=dt).reshape(ct, cs)
        lo, hi = j * cs, min((j + 1) * cs, ns)
        out[lo:hi] = a[off, :hi - lo]
    return out

def main(d1, d2):
    a, b = pathlib.Path(d1) / 'vvm_sol.zarr', pathlib.Path(d2) / 'vvm_sol.zarr'
    names = sorted(p.name for p in a.iterdir() if (p / '.zarray').exists())
    nt = min(meta(a, 'qcond')['shape'][0], meta(b, 'qcond')['shape'][0])
    print(f"A = {a}\nB = {b}\n共同觀測數 = {nt}")

    ga, gb = read(a, 'gbxindex'), read(b, 'gbxindex')
    oa, ob = np.argsort(ga, kind='stable'), np.argsort(gb, kind='stable')
    same_order = np.array_equal(ga, gb)
    print(f"gbxindex: A 與 B {'順序相同' if same_order else '順序不同 (已依 gbxindex 重排後比較)'}"
          f", 兩者皆為 0..{len(ga)-1} 的完整排列: "
          f"{np.array_equal(ga[oa], np.arange(len(ga))) and np.array_equal(gb[ob], np.arange(len(gb)))}\n")
    print(f"{'變數':<26} {'形狀':>18} {'結果':<10} 最大絕對差 / 首個不同處")
    print("-" * 92)
    bad = 0
    for v in names:
        ma, mb = meta(a, v), meta(b, v)
        if ma['shape'] != mb['shape']:
            print(f"{v:<26} {'':>18} {'長度不同':<10} A={ma['shape']} B={mb['shape']}"
                  f"  (差 {mb['shape'][0]-ma['shape'][0]:+d})")
            bad += 1
            continue
        if 0 in ma['shape']:
            print(f"{v:<26} {str(ma['shape']):>18} {'未寫出':<10} (chunk 尚未 flush)")
            continue
        if len(ma['shape']) == 1:
            x, y = read(a, v), read(b, v)
            slices = [(None, x, y)]
        else:
            # A multi-rank run may lay the global array out in a different gridbox
            # order than a single-rank one, which would make an element-wise diff
            # fail on a permutation rather than on a real disagreement. Put both
            # into ascending gbxindex order first so the comparison is on values.
            slices = []
            for ti in range(nt):
                x, y = read(a, v, ti), read(b, v, ti)
                slices.append((ti, None if x is None else x[oa],
                                   None if y is None else y[ob]))
        worst, where = 0.0, None
        for ti, x, y in slices:
            if x is None or y is None:
                continue
            if x.dtype.kind in 'fc':
                d = np.abs(x.astype(np.float64) - y.astype(np.float64))
                m = float(d.max()) if d.size else 0.0
            else:
                d = (x != y)
                m = float(d.sum())
            if m > worst:
                worst = m
                idx = int(np.argmax(d)) if d.size else -1
                where = f"t[{ti}] 格點 {idx}" if ti is not None else f"索引 {idx}"
        ok = worst == 0.0
        bad += (not ok)
        print(f"{v:<26} {str(ma['shape']):>18} {'一致' if ok else '不同':<10} "
              f"{'' if ok else f'{worst:.6e}  @ {where}'}")
    print("\n" + ("全部逐位相同" if bad == 0 else f"{bad} 個變數有差異"))
    return 1 if bad else 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2]))
