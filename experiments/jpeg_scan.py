#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 the foldz authors
"""jpeg_scan.py - how much can format-aware modelling save on JPEG files? (handover section 7, item 2)

For every baseline JPEG it:
  1. parses the markers and Huffman-decodes the scan into quantized DCT coefficients,
  2. re-encodes the coefficients with the file's own Huffman tables and checks the result is
     byte-identical to the original scan (no match -> the file counts as kept raw),
  3. estimates the coded size of the coefficients with adaptive binary context models (the cost an
     arithmetic/range coder would reach), using foldz-style ideas: the DC prediction residual is
     centre-folded (signed -> unsigned), block nonzero counts are coded as a 6-bit binary tree, and
     every AC coefficient is modelled from the same coefficient in the left and above blocks.

    python3 jpeg_scan.py DIR_OR_FILE [...] [-j N] [--csv out.csv]

Headers (EXIF, tables...) are kept, costed at min(raw, LZMA2). Pure Python 3.8+, no numpy.
Estimates only: nothing is written yet. Progressive and arithmetic-coded JPEGs are reported as
unsupported for now.
"""
import argparse, collections, csv, lzma, math, mmap, os, sys, time
from array import array
from multiprocessing import Pool

ZIGZAG_LEN = 64
log2 = math.log2

# ---------------------------------------------------------------- marker parsing
class Unsupported(Exception):
    pass

def parse(buf):
    """Return frame info, Huffman tables, scan info, header bytes and the entropy-coded segments."""
    if buf[:2] != b"\xff\xd8":
        raise Unsupported("not a JPEG")
    pos, n = 2, len(buf)
    dht, frame, dri, scans = {}, None, 0, []
    header_len = 2
    while pos < n:
        if buf[pos] != 0xFF:
            raise Unsupported("bad marker")
        while pos < n and buf[pos] == 0xFF:
            pos += 1
        m = buf[pos]; pos += 1
        if m == 0xD9:                                   # EOI
            header_len += 2
            break
        if 0xD0 <= m <= 0xD7 or m == 0x01:
            header_len += 2
            continue
        ln = int.from_bytes(buf[pos:pos + 2], "big")
        seg = buf[pos + 2:pos + ln]
        header_len += 2 + ln
        if m in (0xC0, 0xC1):
            if seg[0] != 8:
                raise Unsupported(f"{seg[0]}-bit precision")
            Y, X, nc = int.from_bytes(seg[1:3], "big"), int.from_bytes(seg[3:5], "big"), seg[5]
            comps = []
            for i in range(nc):
                cid, hv, tq = seg[6 + 3 * i:9 + 3 * i]
                comps.append(dict(id=cid, H=hv >> 4, V=hv & 15, tq=tq))
            frame = dict(X=X, Y=Y, comps=comps)
        elif m in (0xC2, 0xC6, 0xCA):
            raise Unsupported("progressive")
        elif m in (0xC3, 0xC5, 0xC7, 0xC9, 0xCB, 0xCD, 0xCE, 0xCF):
            raise Unsupported("lossless/hierarchical/arithmetic")
        elif m == 0xC4:
            p = 0
            while p < len(seg):
                tc_th = seg[p]
                counts = list(seg[p + 1:p + 17])
                vals = list(seg[p + 17:p + 17 + sum(counts)])
                dht[(tc_th >> 4, tc_th & 15)] = (counts, vals)
                p += 17 + sum(counts)
        elif m == 0xDD:
            dri = int.from_bytes(seg[:2], "big")
        elif m == 0xDA:
            if frame is None:
                raise Unsupported("scan before frame")
            if scans:
                raise Unsupported("multi-scan baseline")
            ns = seg[0]
            sc = [(seg[1 + 2 * i], seg[2 + 2 * i] >> 4, seg[2 + 2 * i] & 15) for i in range(ns)]
            if seg[1 + 2 * ns:4 + 2 * ns] != b"\x00\x3f\x00":
                raise Unsupported("non-baseline scan parameters")
            pos += ln
            start = pos
            segments, spans = [], []                    # entropy-coded data, split at RST markers
            while True:
                f = buf.find(b"\xff", pos)
                if f < 0 or f + 1 >= n:
                    raise Unsupported("truncated scan")
                nxt = buf[f + 1]
                if nxt == 0x00:
                    pos = f + 2
                elif 0xD0 <= nxt <= 0xD7:
                    segments.append(bytes(buf[start:f])); spans.append((start, f))
                    header_len += 2
                    pos = start = f + 2
                else:
                    segments.append(bytes(buf[start:f])); spans.append((start, f))
                    pos = f
                    break
            scans.append((sc, segments, dict(dht), spans))
            continue
        pos += ln
    if not scans:
        raise Unsupported("no scan")
    tail = n - pos if pos < n else 0                    # bytes after EOI (some cameras append data)
    return frame, dri, scans[0], header_len + tail, buf

def build_tables(counts, vals):
    """Decode lookup (16-bit peek -> sym<<5 | len) and encode table (sym -> (code, len))."""
    dec, enc = [-1] * 65536, {}
    code, k = 0, 0
    for L in range(1, 17):
        for _ in range(counts[L - 1]):
            sym = vals[k]; k += 1
            enc[sym] = (code, L)
            base = code << (16 - L)
            e = (sym << 5) | L
            for i in range(base, base + (1 << (16 - L))):
                dec[i] = e
            code += 1
        code <<= 1
    return dec, enc

# ---------------------------------------------------------------- decode / re-encode
def layout(frame, sc):
    comps = {c["id"]: c for c in frame["comps"]}
    Hmax = max(c["H"] for c in frame["comps"]); Vmax = max(c["V"] for c in frame["comps"])
    X, Y = frame["X"], frame["Y"]
    out = []
    if len(sc) == 1:
        c = comps[sc[0][0]]
        bw = -(-(-(-X * c["H"] // Hmax)) // 8); bh = -(-(-(-Y * c["V"] // Vmax)) // 8)
        out.append(dict(c, td=sc[0][1], ta=sc[0][2], bw=bw, bh=bh, h=1, v=1))
        return out, bw * bh, bw
    mcux, mcuy = -(-X // (8 * Hmax)), -(-Y // (8 * Vmax))
    for cid, td, ta in sc:
        c = comps[cid]
        out.append(dict(c, td=td, ta=ta, bw=mcux * c["H"], bh=mcuy * c["V"], h=c["H"], v=c["V"]))
    return out, mcux * mcuy, mcux

def block_order(cl, nmcu, mcux):
    """For each MCU, the (component index, block index) pairs in decode order."""
    order = []
    for m in range(nmcu):
        my, mx = divmod(m, mcux)
        for ci, c in enumerate(cl):
            for v in range(c["v"]):
                for h in range(c["h"]):
                    order.append((ci, (my * c["v"] + v) * c["bw"] + mx * c["h"] + h))
    return order

def decode(frame, dri, scan):
    sc, segments, dht, _ = scan
    cl, nmcu, mcux = layout(frame, sc)
    tabs = {}
    for c in cl:
        for key in ((0, c["td"]), (1, c["ta"])):
            if key not in dht:
                raise Unsupported("missing Huffman table")
            if key not in tabs:
                tabs[key] = build_tables(*dht[key])
    coef = [array("h", bytes(2 * 64 * c["bw"] * c["bh"])) for c in cl]
    per = sum(c["h"] * c["v"] for c in cl)
    order = block_order(cl, nmcu, mcux)
    interval = dri if dri else nmcu
    if -(-nmcu // interval) != len(segments):
        raise Unsupported(f"restart count mismatch ({len(segments)} segments, {-(-nmcu // interval)} expected)")
    pads = set()
    for si, seg in enumerate(segments):
        data = seg.replace(b"\xff\x00", b"\xff")
        ndata = len(data)
        acc = nb = pos = 0
        pred = [0] * len(cl)
        lo, hi = si * interval * per, min(nmcu, (si + 1) * interval) * per
        for ci, bi in order[lo:hi]:
            c = cl[ci]
            dct, _ = tabs[(0, c["td"])]; act, _ = tabs[(1, c["ta"])]
            out = coef[ci]; base = bi * 64
            # DC
            while nb < 32:
                acc = (acc << 8) | (data[pos] if pos < ndata else 0); pos += 1; nb += 8
            e = dct[(acc >> (nb - 16)) & 0xFFFF]
            if e < 0:
                raise Unsupported("corrupt scan (bad DC code)")
            nb -= e & 31; s = e >> 5
            if s:
                if s > 11:
                    raise Unsupported("corrupt scan")
                v = (acc >> (nb - s)) & ((1 << s) - 1); nb -= s
                if v < (1 << (s - 1)):
                    v -= (1 << s) - 1
                pred[ci] += v
            out[base] = pred[ci]
            k = 1
            while k < 64:
                if nb < 32:
                    acc = ((acc & ((1 << nb) - 1)) << 32) | int.from_bytes(data[pos:pos + 4].ljust(4, b"\0"), "big")
                    pos += 4; nb += 32
                e = act[(acc >> (nb - 16)) & 0xFFFF]
                if e < 0:
                    raise Unsupported("corrupt scan (bad AC code)")
                nb -= e & 31; rs = e >> 5
                r, s = rs >> 4, rs & 15
                if s == 0:
                    if r == 15:
                        k += 16; continue
                    break
                k += r
                if k > 63:
                    raise Unsupported("corrupt scan (run past block end)")
                v = (acc >> (nb - s)) & ((1 << s) - 1); nb -= s
                if v < (1 << (s - 1)):
                    v -= (1 << s) - 1
                out[base + k] = v
                k += 1
            acc &= (1 << nb) - 1
        used = pos * 8 - nb
        if (used + 7) // 8 != ndata:
            raise Unsupported("scan length mismatch")
        pad = (-used) % 8
        if pad:
            pads.add((acc >> (nb - pad)) & ((1 << pad) - 1) == (1 << pad) - 1)
    return cl, coef, order, interval, per, tabs, segments, pads

def encode(cl, coef, order, interval, per, tabs, nseg, pad_ones=True):
    """Re-encode the coefficients exactly as a standard baseline encoder would; return the segments."""
    segs = []
    for si in range(nseg):
        out = bytearray(); acc = nb = 0
        pred = [0] * len(cl)
        lo, hi = si * interval * per, min(len(order), (si + 1) * interval * per)
        for ci, bi in order[lo:hi]:
            c = cl[ci]
            _, dce = tabs[(0, c["td"])]; _, ace = tabs[(1, c["ta"])]
            blk = coef[ci]; base = bi * 64
            dc = blk[base]; d = dc - pred[ci]; pred[ci] = dc
            s = abs(d).bit_length()
            code, L = dce[s]
            acc = (acc << L) | code; nb += L
            if s:
                acc = (acc << s) | (d if d > 0 else d + (1 << s) - 1); nb += s
            run = 0
            for k in range(1, 64):
                v = blk[base + k]
                if v == 0:
                    run += 1; continue
                while run > 15:
                    code, L = ace[0xF0]; acc = (acc << L) | code; nb += L; run -= 16
                s = abs(v).bit_length()
                code, L = ace[(run << 4) | s]
                acc = (acc << (L + s)) | (code << s) | (v if v > 0 else v + (1 << s) - 1); nb += L + s
                run = 0
            if run:
                code, L = ace[0x00]; acc = (acc << L) | code; nb += L
            while nb >= 8:
                nb -= 8; b = (acc >> nb) & 0xFF; out.append(b)
                if b == 0xFF:
                    out.append(0)
            acc &= (1 << nb) - 1
        if nb:
            pad = 8 - nb
            b = ((acc << pad) | ((1 << pad) - 1 if pad_ones else 0)) & 0xFF
            out.append(b)
            if b == 0xFF:
                out.append(0)
        segs.append(bytes(out))
    return segs

# ---------------------------------------------------------------- cost model
class Bits:
    """Adaptive binary models (counts with KT-style prior, halved when large); sums -log2 p."""
    __slots__ = ("n0", "n1", "cost")
    def __init__(self, size):
        self.n0 = [0.4] * size; self.n1 = [0.4] * size; self.cost = 0.0
    def code(self, ctx, bit):
        n0, n1 = self.n0[ctx], self.n1[ctx]
        if bit:
            self.cost -= log2(n1 / (n0 + n1)); n1 += 1
        else:
            self.cost -= log2(n0 / (n0 + n1)); n0 += 1
        if n0 + n1 > 255:
            n0 *= .5; n1 *= .5
        self.n0[ctx] = n0; self.n1[ctx] = n1

def qbucket(x):                                         # 0,1,2,3,4-5,6-8,9-14,15+
    return x if x < 4 else (4 if x < 6 else 5 if x < 9 else 6 if x < 15 else 7)

KBAND = [0] + [min(7, (k + 1).bit_length() - 1) for k in range(1, 64)]   # 1,2-3,4-7,... bands

def model_cost(cl, coef):
    """Estimated bits for all coefficients, per part."""
    parts = collections.Counter()
    for ci, c in enumerate(cl):
        cc = 0 if ci == 0 else 1                         # luma vs chroma statistics
        bw, bh = c["bw"], c["bh"]
        blk = coef[ci]
        nzc = array("B", bytes(bw * bh))
        m_nz, m_zero, m_exp, m_man, m_sign, m_dc = (Bits(1 << 14) for _ in range(6))
        for by in range(bh):
            for bx in range(bw):
                bi = by * bw + bx; base = bi * 64
                hasL, hasA = bx > 0, by > 0
                L = base - 64 if hasL else -1; A = base - 64 * bw if hasA else -1
                # ---- DC: MED prediction from left/above/above-left, residual centre-folded
                dc = blk[base]
                if hasL and hasA:
                    a, b, cdl = blk[L], blk[A], blk[A - 64]
                    p = min(a, b) if cdl >= max(a, b) else max(a, b) if cdl <= min(a, b) else a + b - cdl
                    g = qbucket(min(15, abs(a - b)))
                elif hasL:
                    p, g = blk[L], 8
                elif hasA:
                    p, g = blk[A], 8
                else:
                    p, g = 0, 9
                r = dc - p
                u = 2 * r if r >= 0 else -2 * r - 1            # the fold: values near 0 -> small codes
                e = u.bit_length()
                ctxb = (cc * 10 + g) * 16
                for i in range(e):
                    m_dc.code(ctxb + i, 1)
                if e < 15:
                    m_dc.code(ctxb + e, 0)
                for j in range(e - 2, -1, -1):
                    m_dc.code(8192 + (cc * 16 + e) * 16 + min(j, e - 2 - j, 3) * 2 + (j == e - 2), (u >> j) & 1)
                # ---- number of nonzero AC coefficients: 6-bit binary tree, context = neighbours' counts
                nz = 0
                for k in range(1, 64):
                    if blk[base + k]:
                        nz += 1
                nzc[bi] = nz
                if hasL and hasA:
                    pn = (nzc[bi - 1] + nzc[bi - bw] + 1) >> 1
                elif hasL:
                    pn = nzc[bi - 1]
                elif hasA:
                    pn = nzc[bi - bw]
                else:
                    pn = 0
                pb = qbucket(pn) if pn < 15 else 7 + min(4, (pn - 15) // 8)
                node = 1
                for j in range(5, -1, -1):
                    bit = (nz >> j) & 1
                    m_nz.code(((cc * 12 + pb) * 64 + node), bit)
                    node = node * 2 + bit
                # ---- AC coefficients in zigzag order until the last nonzero
                rem = nz; k = 1
                while rem:
                    v = blk[base + k]
                    nl = abs(blk[L + k]) if hasL else 0
                    na = abs(blk[A + k]) if hasA else 0
                    nm = nl + na if hasL == hasA else 2 * (nl + na)     # one neighbour counts double
                    mb = qbucket(nm)
                    if rem < 64 - k:                       # otherwise every remaining one is nonzero
                        m_zero.code(((cc * 64 + k) * 8 + min(rem - 1, 7)) * 8 + mb, 1 if v else 0)
                    if v:
                        av = abs(v); e = av.bit_length()
                        ctxe = ((cc * 8 + KBAND[k]) * 8 + mb) * 12
                        for i in range(1, e):
                            m_exp.code(ctxe + i, 1)
                        if e < 11:
                            m_exp.code(ctxe + e, 0)
                        for j in range(e - 2, -1, -1):
                            m_man.code((cc * 12 + e) * 12 + min(j, 10), (av >> j) & 1)
                        sl = (blk[L + k] > 0) - (blk[L + k] < 0) if hasL else 0
                        sa = (blk[A + k] > 0) - (blk[A + k] < 0) if hasA else 0
                        m_sign.code(((cc * 64 + k) * 3 + sl + 1) * 3 + sa + 1, v < 0)
                        rem -= 1
                    k += 1
        for name, m in (("dc", m_dc), ("nonzero count", m_nz), ("zero flags", m_zero),
                        ("magnitude", m_exp), ("mantissa", m_man), ("sign", m_sign)):
            parts[name] += m.cost
    return parts

# ---------------------------------------------------------------- per-file worker
FIELDS = ["file", "size", "header", "scan", "status", "model", "after", "saved", "xz", "note",
          "dc", "nonzero count", "zero flags", "magnitude", "mantissa", "sign"]

def scan_file(path):
    row = dict(file=path, size=0, header=0, scan=0, status="", model=0, after=0, saved=0.0, xz=0, note="")
    t = time.time()
    try:
        with open(path, "rb") as f:
            size = os.fstat(f.fileno()).st_size
            row["size"] = size
            if size < 4:
                row["status"] = "not jpeg"; return row
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            if mm[:3] != b"\xff\xd8\xff":
                row["status"] = "not jpeg"; return row
            row["xz"] = len(lzma.compress(mm[:], format=lzma.FORMAT_RAW, filters=[
                {"id": lzma.FILTER_LZMA2, "preset": 9, "dict_size": max(4096, min(size, 16 << 20))}]))
            frame, dri, scan, _, _ = parse(mm)
            scan_bytes = sum(len(s) for s in scan[1])
            row.update(header=size - scan_bytes, scan=scan_bytes)
            cl, coef, order, interval, per, tabs, segments, pads = decode(frame, dri, scan)
            if len(pads) > 1:
                raise Unsupported("mixed padding bits")
            pad_ones = pads != {False}
            if encode(cl, coef, order, interval, per, tabs, len(segments), pad_ones) != segments:
                raise Unsupported("re-encode differs (non-standard encoder choices)")
            parts = model_cost(cl, coef)
            model = int(sum(parts.values()) / 8) + 8          # + a few bytes: padding style, lengths
            hdr = bytearray(); p = 0                          # header = everything but the scan data
            for a, b in scan[3]:
                hdr += mm[p:a]; p = b
            hdr += mm[p:size]
            hdr_cost = min(len(hdr), len(lzma.compress(bytes(hdr), format=lzma.FORMAT_RAW,
                                                       filters=[{"id": lzma.FILTER_LZMA2, "preset": 9}])) + 2)
            after = hdr_cost + model
            row.update(status="ok", model=model, after=after, saved=round(1 - after / size, 4),
                       note=f"{frame['X']}x{frame['Y']}, {len(cl)} comp, "
                            f"{'x'.join(str(c['H']) + str(c['V']) for c in cl)}, rst {dri}, {time.time()-t:.1f}s")
            for k, v in parts.items():
                row[k] = int(v / 8)
        finally:
            try:
                mm.close()
            except BufferError:
                pass
    except Unsupported as e:
        row["status"] = "unsupported"; row["note"] = str(e)
    except MemoryError:
        row["status"] = "error"; row["note"] = "out of memory (lower -j)"
    except Exception as e:
        row["status"] = "error"; row["note"] = f"{type(e).__name__}: {e}"
    return row

# ---------------------------------------------------------------- report
def report(rows, secs):
    jp = [r for r in rows if r["status"] != "not jpeg"]
    ok = [r for r in jp if r["status"] == "ok"]
    print(f"\n{len(rows)} files, {len(jp)} JPEG, {len(ok)} decoded and rebuilt byte-exact | {secs:.0f}s\n")
    if not jp:
        print("no JPEG files found"); return
    size = sum(r["size"] for r in jp)
    after = sum(r["after"] if r["status"] == "ok" else r["size"] for r in jp)
    oks, oka = sum(r["size"] for r in ok), sum(r["after"] for r in ok)
    xz = sum(r["xz"] or r["size"] for r in jp)
    hdr = f"{'':22} {'files':>5} {'size MB':>9} {'after MB':>9} {'saved':>7}"
    print(hdr); print("-" * len(hdr))
    print(f"{'rebuilt exactly':22} {len(ok):5d} {oks/1e6:9.3f} {oka/1e6:9.3f} {1-oka/max(oks,1):7.1%}")
    print(f"{'all JPEG (rest raw)':22} {len(jp):5d} {size/1e6:9.3f} {after/1e6:9.3f} {1-after/max(size,1):7.1%}")
    print(f"{'for contrast: xz -9':22} {len(jp):5d} {size/1e6:9.3f} {xz/1e6:9.3f} {1-xz/max(size,1):7.1%}\n")
    if ok:
        scan = sum(r["scan"] for r in ok); model = sum(r["model"] for r in ok)
        print(f"coefficient data: {scan/1e6:.3f} MB Huffman -> {model/1e6:.3f} MB modelled "
              f"({1-model/max(scan,1):.1%} smaller); where the modelled bits go:")
        for k in FIELDS[10:]:
            v = sum(r.get(k, 0) for r in ok)
            print(f"  {k:14} {v/max(model,1):6.1%}")
        print()
    bad = collections.Counter(f"{r['status']}: {r['note']}" for r in jp if r["status"] != "ok")
    for k, v in bad.most_common(6):
        print(f"{v:5d} x {k}")
    print("\nEstimates: adaptive-model cost (what a range coder would reach), headers at min(raw, LZMA2).")

def walk(paths):
    for p in paths:
        if os.path.isdir(p):
            for root, _, names in os.walk(p):
                for nme in sorted(names):
                    fp = os.path.join(root, nme)
                    if os.path.isfile(fp) and not os.path.islink(fp):
                        yield fp
        elif os.path.isfile(p):
            yield p
        else:
            print(f"error: no such file or directory: {p}", file=sys.stderr)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("-j", type=int, default=max(1, min(os.cpu_count() or 1, 3)), help="worker processes (default up to 3)")
    ap.add_argument("--csv", help="write one row per file")
    a = ap.parse_args()
    files = list(walk(a.paths))
    if not files:
        sys.exit("error: nothing to scan")
    out = open(a.csv, "w", newline="") if a.csv else None
    w = csv.DictWriter(out, fieldnames=FIELDS, extrasaction="ignore") if out else None
    if w:
        w.writeheader()
    t0, rows = time.time(), []
    with Pool(a.j, maxtasksperchild=8) as pool:
        for i, r in enumerate(pool.imap_unordered(scan_file, files)):
            rows.append(r)
            if w:
                w.writerow(r); out.flush()
            if sys.stderr.isatty():
                print(f"\r{i+1}/{len(files)} files", end="", file=sys.stderr)
    if sys.stderr.isatty():
        print(file=sys.stderr)
    if out:
        out.close()
    report(rows, time.time() - t0)

if __name__ == "__main__":
    main()
