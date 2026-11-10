#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 the foldz authors
"""deflate_scan.py - how often can real deflate streams be rebuilt bit-exactly with zlib?

Step 1 of the recompression plan (handover section 7). For every deflate stream found in
gzip, ZIP (also jar/whl/docx...), PNG, PDF and bare zlib files, it inflates the content and
searches zlib settings (level, memLevel, window bits, strategy) for one that reproduces the
original compressed bytes exactly. Matched streams are costed as LZMA2 of the content plus
3 bytes of settings; unmatched ones stay raw.

    python3 deflate_scan.py DIR_OR_FILE [...] [-j 8] [--mem-mb 1500] [--quick] [--diagnose] [--csv out.csv]

Containers inside streams (an APK inside a ZIP, a ZIP inside a gzip) are scanned too, up to
3 levels deep. --diagnose parses unmatched streams to tell which kind of encoder made them.

Only deflate bodies are compared; wrappers (gzip header/trailer, zlib header/adler, chunk
framing) are kept verbatim and not counted. Requires Python 3.8+ only (no numpy).

Memory: files are memory-mapped, not read, so large non-deflate files (videos) cost almost nothing.
A total budget (--mem-mb, default 60% of MemAvailable) is split across workers; each worker's
LZMA dictionary and the largest stream it will inflate are sized to its share, and the worker is
capped with RLIMIT_DATA so an outlier fails that one file instead of the OS killing everything.
"""
import argparse, collections, csv, io, lzma, mmap, os, re, struct, sys, time, zipfile, zlib
from multiprocessing import Pool

SETTINGS_BYTES = 3
CHUNK = 1 << 12
IN_STEP, OUT_STEP = 1 << 20, 16 << 20             # bounded inflate: never one huge output buffer
LIMIT = {"content": 256 * 10**6, "dict": 64 << 20}    # set per worker from the memory budget
STRATS = [zlib.Z_DEFAULT_STRATEGY, zlib.Z_FILTERED, zlib.Z_HUFFMAN_ONLY,
          getattr(zlib, "Z_RLE", 3), getattr(zlib, "Z_FIXED", 4)]
STRAT_NAME = {0: "default", 1: "filtered", 2: "huffman", 3: "rle", 4: "fixed"}

# ---------------------------------------------------------------- stream extraction
def as_file(buf):
    return buf if isinstance(buf, mmap.mmap) else io.BytesIO(buf)

def inflate_raw(buf, start, wbits=-15, end=None):
    """Inflate from buf[start:end]; return (content, end_offset, size) or None.
    content is None when the output exceeds LIMIT['content'] (the stream is still measured, not kept)."""
    d = zlib.decompressobj(wbits)
    mv = memoryview(buf)[start:end]
    parts, size, keep, limit = [], 0, True, LIMIT["content"]
    try:
        for i in range(0, len(mv), IN_STEP):
            data = mv[i:i + IN_STEP]
            while True:
                out = d.decompress(data, OUT_STEP)
                size += len(out)
                if keep:
                    if size > limit:
                        keep, parts = False, []
                    else:
                        parts.append(out)
                if d.eof or not d.unconsumed_tail:
                    break
                data = d.unconsumed_tail
            if d.eof:
                stop = start + i + len(mv[i:i + IN_STEP]) - len(d.unused_data)
                break
        else:
            out = d.flush()
            size += len(out)
            if keep and size <= limit:
                parts.append(out)
            elif keep:
                keep, parts = False, []
            if not d.eof:
                return None
            stop = start + len(mv) - len(d.unused_data)
    except zlib.error:
        return None
    finally:
        mv.release()
    return (b"".join(parts) if keep else None), stop, size

def zlib_body(buf, start):
    """A zlib-wrapped stream at buf[start]; return (raw_deflate, content, hint_level) or None."""
    if len(buf) - start < 6:
        return None
    cmf, flg = buf[start], buf[start + 1]
    if cmf & 0x0F != 8 or (cmf * 256 + flg) % 31 or flg & 0x20:   # not deflate, bad check, or preset dict
        return None
    r = inflate_raw(buf, start + 2)
    if not r:
        return None
    content, end, size = r
    raw = bytes(buf[start + 2:end]) if content is not None else None
    return raw, end - start - 2, content, size, {0: 1, 1: 5, 2: 6, 3: 9}[flg >> 6]

def streams_gzip(buf):
    pos = 0
    while pos + 18 <= len(buf) and buf[pos:pos + 3] == b"\x1f\x8b\x08":
        flg, xfl = buf[pos + 3], buf[pos + 8]
        p = pos + 10
        if flg & 4: p += 2 + struct.unpack_from("<H", buf, p)[0]
        for bit in (8, 16):                             # FNAME, FCOMMENT (mmap has find, not index)
            if flg & bit:
                p = buf.find(b"\0", p) + 1
                if p == 0:
                    return
        if flg & 2: p += 2
        r = inflate_raw(buf, p)
        if not r:
            return
        content, end, size = r
        raw = bytes(buf[p:end]) if content is not None else None
        yield "gzip", p, raw, end - p, content, size, {2: 9, 4: 1}.get(xfl)
        pos = end + 8                                   # skip CRC32 + ISIZE, maybe another member

def streams_zip(buf):
    with zipfile.ZipFile(as_file(buf)) as z:
        infos = z.infolist()
    for info in infos:
        if info.compress_type != zipfile.ZIP_DEFLATED or info.compress_size == 0:
            continue
        o = info.header_offset
        n, m = struct.unpack_from("<HH", buf, o + 26)
        start = o + 30 + n + m
        r = inflate_raw(buf, start, end=start + info.compress_size)
        if not r:
            continue
        content, _, size = r
        raw = bytes(buf[start:start + info.compress_size]) if content is not None else None
        yield "zip", start, raw, info.compress_size, content, size, None

def png_info(buf):
    """Encoder hints: the Software text chunk and the IDAT chunk sizes (encoders differ in both)."""
    pos, sizes, soft = 8, [], ""
    while pos + 8 <= len(buf):
        ln, typ = struct.unpack_from(">I4s", buf, pos)
        body = buf[pos + 8:pos + 8 + ln]
        if typ == b"IDAT":
            sizes.append(ln)
        elif typ in (b"tEXt", b"iTXt") and body.split(b"\0", 1)[0] == b"Software":
            soft = body.split(b"\0")[-1].decode("latin-1", "replace")[:60]
        pos += 12 + ln
    common = collections.Counter(sizes[:-1] or sizes).most_common(1)
    return f"{len(sizes)} IDAT chunks of {common[0][0] if common else 0} B" + (f", Software: {soft}" if soft else "")

def streams_png(buf):
    pos, idat = 8, bytearray()
    while pos + 8 <= len(buf):
        ln, typ = struct.unpack_from(">I4s", buf, pos)
        if typ == b"IDAT":
            idat += buf[pos + 8:pos + 8 + ln]
        pos += 12 + ln
    r = zlib_body(idat, 0)
    del idat
    if r:
        yield ("png", 8) + r

PDF_STREAM = re.compile(rb"stream\r?\n")
def streams_pdf(buf):
    for mt in PDF_STREAM.finditer(buf):
        start = mt.end()
        head = buf[max(0, start - 600):mt.start()]
        d = head.rfind(b"<<")
        if d < 0 or b"/FlateDecode" not in head[d:]:
            continue
        r = zlib_body(buf, start)
        if r:
            yield ("pdf", start) + r

def is_container(buf):
    return (buf[:3] == b"\x1f\x8b\x08" or buf[:4] == b"PK\x03\x04" or buf[:8] == b"\x89PNG\r\n\x1a\n"
            or buf[:5] == b"%PDF-")

def find_streams(buf):
    if buf[:3] == b"\x1f\x8b\x08":
        yield from streams_gzip(buf)
    elif buf[:4] == b"PK\x03\x04" and zipfile.is_zipfile(as_file(buf)):
        yield from streams_zip(buf)
    elif buf[:8] == b"\x89PNG\r\n\x1a\n":
        yield from streams_png(buf)
    elif buf[:5] == b"%PDF-":
        yield from streams_pdf(buf)
    else:
        r = zlib_body(buf, 0)
        if r:
            yield ("zlib", 2) + r

# ---------------------------------------------------------------- settings search
def configs(hint, quick):
    levels = list(range(10))
    def key(c):
        lvl, mem, wb, st = c
        return (st != 0, mem != 8, wb != 15, lvl != hint, lvl != 6, lvl != 9, abs(mem - 8), 15 - wb, st, lvl)
    space = [(l, m, w, s) for l in levels for m in (range(8, 10) if quick else range(1, 10))
             for w in ((15,) if quick else range(9, 16)) for s in ((0,) if quick else STRATS)]
    seen, out = set(), []
    for c in sorted(space, key=key):
        if c[0] == 0 and c[3] != 0:                     # level 0 ignores strategy
            continue
        if c not in seen:
            seen.add(c); out.append(c)
    return out

def try_config(raw, content, cfg):
    lvl, mem, wb, st = cfg
    c = zlib.compressobj(lvl, zlib.DEFLATED, -wb, mem, st)
    if lvl == 0:                                        # stored output depends on input chunking: one shot
        return c.compress(content) + c.flush() == raw
    pos = 0
    for i in range(0, len(content), CHUNK):             # early abort on the first differing byte
        out = c.compress(content[i:i + CHUNK])
        if out:
            if raw[pos:pos + len(out)] != out:
                return False
            pos += len(out)
    out = c.flush()
    return pos + len(out) == len(raw) and raw[pos:] == out

def find_settings(raw, content, hint, quick):
    for i, cfg in enumerate(configs(hint, quick)):
        if try_config(raw, content, cfg):
            return cfg, i + 1
    return None, None

def lzma_size(content):
    """LZMA2 preset 9, dictionary no larger than the content (same result, far less memory) and
    no larger than this worker's cap (bigger streams then come out slightly larger: conservative)."""
    dict_size = max(4096, min(len(content), LIMIT["dict"]))
    return len(lzma.compress(content, format=lzma.FORMAT_RAW,
                             filters=[{"id": lzma.FILTER_LZMA2, "preset": 9, "dict_size": dict_size}]))

# ---------------------------------------------------------------- diagnosis of unmatched streams
LBASE = [3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,59,67,83,99,115,131,163,195,227,258]
LEXT = [0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0]
DBASE = [1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,257,385,513,769,1025,1537,2049,3073,4097,6145,
         8193,12289,16385,24577]
DEXT = [0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13]
ZLIB_MAX_DIST = 32768 - 262          # zlib and GNU gzip never reach further back than this

def huff_table(lengths):
    mx = max(lengths) or 1
    tab, code, nxt = [None] * (1 << mx), 0, {}
    cnt = collections.Counter(l for l in lengths if l)
    for L in range(1, mx + 1):
        code = (code + cnt.get(L - 1, 0)) << 1 if L > 1 else 0
        nxt[L] = code
    for sym, L in enumerate(lengths):
        if not L:
            continue
        c = nxt[L]; nxt[L] += 1
        rev = int(f"{c:0{L}b}"[::-1], 2)
        for f in range(rev, 1 << mx, 1 << L):
            tab[f] = (sym, L)
    return tab, mx

def diagnose(raw, limit=2_000_000):
    """Parse deflate block headers and symbols to fingerprint the encoder (pure Python, first `limit` bytes)."""
    data = bytes(raw[:limit]) + b"\0" * 8
    pos = bb = bc = 0
    def need(n):
        nonlocal pos, bb, bc
        while bc < n:
            bb |= data[pos] << bc; pos += 1; bc += 8
    def bits(n):
        nonlocal bb, bc
        need(n); v = bb & ((1 << n) - 1); bb >>= n; bc -= n; return v
    def sym(t):
        nonlocal bb, bc
        tab, mx = t; need(mx); s_, L = tab[bb & ((1 << mx) - 1)]; bb >>= L; bc -= L; return s_
    fixed = (huff_table([8]*144 + [9]*112 + [7]*24 + [8]*8), huff_table([5]*30))
    kinds, syms, empty, far, maxd, final = collections.Counter(), [], 0, 0, 0, 0
    try:
        while not final and pos < limit:
            final, btype = bits(1), bits(2)
            if btype == 0:
                bb >>= bc % 8; bc -= bc % 8
                ln = bits(16); bits(16)
                kinds["stored"] += 1; empty += ln == 0
                for _ in range(ln): bits(8)
                continue
            if btype == 1:
                lt, dt = fixed; kinds["fixed"] += 1
            else:
                hl, hd, hc = bits(5) + 257, bits(5) + 1, bits(4) + 4
                order = [16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15]
                cl = [0] * 19
                for i in range(hc): cl[order[i]] = bits(3)
                ct, lens = huff_table(cl), []
                while len(lens) < hl + hd:
                    x = sym(ct)
                    if x < 16: lens.append(x)
                    elif x == 16: lens += [lens[-1]] * (3 + bits(2))
                    elif x == 17: lens += [0] * (3 + bits(3))
                    else: lens += [0] * (11 + bits(7))
                lt, dt = huff_table(lens[:hl]), huff_table(lens[hl:] or [1])
                kinds["dynamic"] += 1
            n = 0
            while True:
                x = sym(lt)
                if x == 256: break
                n += 1
                if x > 256:
                    bits(LEXT[x - 257]); d = sym(dt); dist = DBASE[d] + bits(DEXT[d])
                    maxd = max(maxd, dist); far += dist > ZLIB_MAX_DIST
            syms.append(n)
    except (IndexError, TypeError):
        pass                                          # ran past the limit or hit corrupt data
    whole = syms[:-1] if final else syms
    mode, share = (collections.Counter(whole).most_common(1)[0] if whole else (0, 0))
    share = share / max(len(whole), 1)
    d = (f"{sum(kinds.values())} blocks ({', '.join(f'{v} {k}' for k, v in kinds.items())}); "
         f"symbols/block: most common {mode} ({share:.0%}); max distance {maxd}; "
         f"{far} beyond zlib's limit; {empty} empty stored (flush markers)")
    if far:
        guess = "not zlib/GNU gzip: uses the full 32 KB window (libdeflate, zopfli, 7-Zip, Apple, ISA-L...)"
    elif empty:
        guess = "zlib-like with flushes at encoder-chosen points; zlib can only match if flush points are reproduced"
    elif share > .8 and mode in {(1 << (m + 6)) - 1 for m in range(1, 10)}:
        guess = f"zlib-style fixed buffer (memLevel {mode.bit_length() - 6}) but different choices: other zlib version or fork (zlib-ng, Chromium zlib) or GNU gzip"
    else:
        guess = "adaptive block splitting: not zlib (libdeflate, zopfli, 7-Zip, Rust miniz/zune...)"
    return d + " -> " + guess

# ---------------------------------------------------------------- per-file worker
MAX_DEPTH = 3

def scan_buf(buf, label, depth, parent, opts, rows):
    quick, diag = opts[:2]
    for fmt, off, raw, comp, content, size, hint in find_streams(buf):
        rid = len(rows)
        row = dict(file=label, depth=depth, id=rid, parent=parent, fmt=fmt, offset=off, comp=comp,
                   content=size, match="", trials=0, lzma=0, note="", diag="")
        rows.append(row)
        if fmt == "png":
            row["note"] = png_info(buf)
        if content is None:
            row["note"] = f"skipped: content {size/1e6:.1f} MB > --max-mb {LIMIT['content']/1e6:.0f}"
            continue
        if size > LIMIT["dict"]:
            row["note"] += ("; " if row["note"] else "") + f"lzma dict capped at {LIMIT['dict'] >> 20} MiB"
        cfg, trials = find_settings(raw, content, hint, quick)
        row.update(match=fmt_cfg(cfg) if cfg else "", trials=trials or 0, lzma=lzma_size(content))
        if not cfg and diag:
            row["diag"] = diagnose(raw)
        del raw
        if depth < MAX_DEPTH and is_container(content):        # e.g. an APK stored inside a ZIP
            scan_buf(content, f"{label} > {fmt}@{off}", depth + 1, rid, opts, rows)
        del content
    if depth < MAX_DEPTH:                                    # stored members need no outer rebuild
        for name, off, inner in stored_members(buf):
            scan_buf(inner, f"{label} > {name}", depth + 1, parent, opts, rows)

def stored_members(buf):
    """ZIP members stored without compression that are themselves containers (e.g. an APK in a ZIP)."""
    if buf[:4] != b"PK\x03\x04":
        return
    try:
        with zipfile.ZipFile(as_file(buf)) as z:
            infos = z.infolist()
    except zipfile.BadZipFile:
        return
    for info in infos:
        if info.compress_type == zipfile.ZIP_STORED and 64 < info.file_size <= LIMIT["content"]:
            o = info.header_offset
            n, m = struct.unpack_from("<HH", buf, o + 26)
            start = o + 30 + n + m
            if is_container(buf[start:start + 8]):
                yield info.filename, start, buf[start:start + info.file_size]

# ---------------------------------------------------------------- files with no deflate stream
GRAY = bytes(i ^ (i >> 1) for i in range(256))         # the fold from recomp.py; reversible per byte
SAMPLE = 1 << 20

def incompressible(mm, size):
    """Already-compressed data (video, JPEG, xz...): three 1 MB samples that zlib -1 can't shrink."""
    if size < 4 * SAMPLE:
        return False
    for at in (0, size // 2, size - SAMPLE):
        piece = mm[at:at + SAMPLE]
        if len(zlib.compress(piece, 1)) < 0.97 * len(piece):
            return False
    return True

def lzma_stream(mm, size, table=None):
    """LZMA2 preset 9 of the whole file, fed in 1 MB pieces (memory: the dictionary, not the file)."""
    c = lzma.LZMACompressor(format=lzma.FORMAT_RAW, filters=[
        {"id": lzma.FILTER_LZMA2, "preset": 9, "dict_size": max(4096, min(size, LIMIT["dict"]))}])
    n = 0
    for i in range(0, size, SAMPLE):
        piece = mm[i:i + SAMPLE]
        n += len(c.compress(piece.translate(table) if table else piece))
    return n + len(c.flush())

def whole_file(mm, path, rows):
    """Experimental fallback: no deflate stream, so compress the file itself, plain and folded."""
    size = len(mm)
    row = dict(file=path, depth=0, id=len(rows), parent=-1, fmt="raw", offset=0, comp=size, content=size,
               match="", trials=0, lzma=0, note="", diag="")
    rows.append(row)
    if incompressible(mm, size):
        row["note"] = "warning: no deflate stream; already compressed (sampled), kept raw"
        return
    plain, folded = lzma_stream(mm, size), lzma_stream(mm, size, GRAY)
    row["lzma"] = min(plain, folded)
    row["match"] = "fold+lzma" if folded < plain else "lzma"
    row["note"] = (f"warning: no deflate stream; whole file compressed (plain {plain}, fold {folded} B)"
                   + (f"; lzma dict capped at {LIMIT['dict'] >> 20} MiB" if size > LIMIT["dict"] else ""))

def init_worker(content_limit, dict_limit, data_limit):
    LIMIT.update(content=content_limit, dict=dict_limit)
    if data_limit:
        try:                                            # heap + anonymous mmaps; file mappings don't count
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (data_limit, data_limit))
        except (ImportError, ValueError, OSError):
            pass

def scan_file(args):
    path, opts = args
    rows = []
    try:
        with open(path, "rb") as f:
            if os.fstat(f.fileno()).st_size == 0:
                return rows
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            scan_buf(mm, path, 0, -1, opts, rows)
            if not rows and opts[2]:
                whole_file(mm, path, rows)
        finally:
            try:
                mm.close()
            except BufferError:                         # a view still alive: freed with the worker
                pass
    except MemoryError:
        rows.append(dict(file=path, depth=0, id=len(rows), parent=-1, fmt="?", offset=0, comp=0, content=0,
                         match="", trials=0, lzma=0, diag="",
                         note="error: out of per-worker memory (raise --mem-mb or lower -j)"))
    except Exception as e:                              # malformed file: report, keep going
        rows.append(dict(file=path, depth=0, id=len(rows), parent=-1, fmt="?", offset=0, comp=0, content=0,
                         match="", trials=0, lzma=0, note=f"error: {type(e).__name__}: {e}", diag=""))
    return rows

def fmt_cfg(c):
    return f"L{c[0]} mem{c[1]} w{c[2]} {STRAT_NAME[c[3]]}"

# ---------------------------------------------------------------- report
def best(r):
    return min(r["comp"], r["lzma"] + SETTINGS_BYTES) if r["lzma"] else r["comp"]

def raw_table(rows):
    """Experimental whole-file results for files with no deflate stream, grouped by extension."""
    by = collections.defaultdict(list)
    for r in rows:
        ext = os.path.splitext(r["file"])[1].lower() or "(none)"
        by[ext].append(r)
    hdr = (f"{'extension':12} {'files':>5} {'size MB':>9} {'after MB':>9} {'saved':>7} {'fold won':>9} "
           f"{'kept raw':>9}")
    print("files with no deflate stream (experimental, whole file: LZMA2 plain or folded, smaller kept)")
    print(hdr); print("-" * len(hdr))
    groups = sorted(by.items(), key=lambda kv: -sum(r["comp"] for r in kv[1]))
    shown = groups[:12] + ([("(other)", [r for _, v in groups[12:] for r in v])] if len(groups) > 12 else [])
    for ext, rs in shown + ([("ALL", rows)] if len(by) > 1 else []):
        if ext == "ALL":
            print("-" * len(hdr))
        size = sum(r["comp"] for r in rs)
        after = sum(best(r) for r in rs)
        print(f"{ext[:12]:12} {len(rs):5d} {size/1e6:9.2f} {after/1e6:9.2f} {1-after/max(size,1):7.1%} "
              f"{sum(r['match'] == 'fold+lzma' for r in rs):9d} {sum(not r['lzma'] for r in rs):9d}")
    print("'kept raw' = sampled as already compressed (video, JPEG, xz...), not run through LZMA.\n")

def table(rows, title):
    by = collections.defaultdict(list)
    for r in rows:
        if r["comp"]:
            by[r["fmt"]].append(r)
    if not by:
        return
    hdr = f"{title:9} {'files':>5} {'streams':>8} {'matched':>8} {'by bytes':>9} {'size MB':>9} {'after MB':>9} {'saved':>7} {'if all matched':>15}"
    print(hdr); print("-" * len(hdr))
    for fmt in sorted(by) + (["ALL"] if len(by) > 1 else []):
        rs = [r for v in by.values() for r in v] if fmt == "ALL" else by[fmt]
        if fmt == "ALL":
            print("-" * len(hdr))
        m = [r for r in rs if r["match"]]
        size = sum(r["comp"] for r in rs)
        after = sum(best(r) if r["match"] else r["comp"] for r in rs)
        ideal = sum(best(r) for r in rs)
        print(f"{fmt:9} {len({r['file'].split(' > ')[0] for r in rs}):5d} {len(rs):8d} {len(m)/len(rs):8.1%} "
              f"{sum(r['comp'] for r in m)/max(size,1):9.1%} {size/1e6:9.2f} {after/1e6:9.2f} "
              f"{1-after/max(size,1):7.1%} {1-ideal/max(size,1):15.1%}")
    print()

def report(rows, nfiles, nskipped, secs):
    print(f"\nzlib {zlib.ZLIB_RUNTIME_VERSION} | {nfiles} files scanned, {nskipped} with no deflate "
          f"stream | {secs:.0f}s\n")
    raw = [r for r in rows if r["fmt"] == "raw"]
    rows = [r for r in rows if r["fmt"] != "raw"]
    top = [r for r in rows if r["depth"] == 0]
    nested = [r for r in rows if r["depth"] > 0]
    if not rows and not raw:
        print("no deflate streams found: nothing to recompress (use without --no-raw to compress them anyway)\n")
    table(top, "format")
    if nested:
        table(nested, "nested")
        # a nested gain is usable only if this stream and every stream around it can be rebuilt
        byid = {(r["file"].split(" > ")[0], r["id"]): r for r in rows}
        def usable(r):
            while True:
                if not r["match"]:
                    return False
                if r["parent"] < 0:
                    return True
                r = byid[(r["file"].split(" > ")[0], r["parent"])]
        roots = [r for r in top if r["comp"]]
        size = sum(r["comp"] for r in roots)
        after = sum(best(r) if r["match"] else r["comp"] for r in roots)
        gain = sum(r["comp"] - best(r) for r in nested if r["parent"] >= 0 and usable(r))
        print(f"total with nested streams: {size/1e6:.2f} -> {(after-gain)/1e6:.2f} MB "
              f"({1-(after-gain)/max(size,1):.1%} saved; nested gains count only where every enclosing "
              f"stream matched)\n")
    hist = collections.Counter(r["match"] for r in rows if r["match"])
    if hist:
        print("most common matching settings:")
        for k, v in hist.most_common(8):
            print(f"  {v:7d}  {k}")
    notes = collections.Counter(r["note"] for r in rows if r["note"].startswith(("skipped", "error")) or "lzma dict" in r["note"])
    for k, v in notes.most_common(5):
        print(f"{v} x {k}")
    unmatched = sorted((r for r in rows if r["comp"] and not r["match"] and not r["note"].startswith("skipped")),
                       key=lambda r: -r["comp"])
    if unmatched:
        print(f"\nlargest unmatched streams{'' if any(r['diag'] for r in unmatched) else ' (add --diagnose to fingerprint the encoder)'}:")
        for r in unmatched[:5]:
            print(f"  {r['comp']/1e6:7.2f} MB  {r['fmt']}  {r['file']}")
            for k in ("note", "diag"):
                if r[k]:
                    print(f"             {r[k]}")
    if raw:
        print()
        raw_table(raw)
    print("'saved' counts only exact matches (LZMA2 of content + 3 bytes); the rest stay raw.\n"
          "'if all matched' is the ceiling a diff-based rebuild for unmatched streams could chase.")

def walk(paths, missing):
    for p in paths:
        if not os.path.exists(p):
            missing.append(p)
        if os.path.isdir(p):
            for root, _, names in os.walk(p):
                for n in names:
                    fp = os.path.join(root, n)
                    if os.path.isfile(fp) and not os.path.islink(fp):
                        yield fp
        elif os.path.isfile(p):
            yield p

def mem_available_mb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1000
    except OSError:
        pass
    return 2000

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("-j", type=int, help="worker processes (default: what the memory budget allows, max one per CPU)")
    ap.add_argument("--mem-mb", type=float, help="total memory budget for all workers (default: 60%% of MemAvailable)")
    ap.add_argument("--quick", action="store_true", help="only memLevel 8-9, window 15, default strategy")
    ap.add_argument("--max-mb", type=float, help="skip streams whose content exceeds this (default: from the budget)")
    ap.add_argument("--diagnose", action="store_true",
                    help="parse unmatched deflate streams to fingerprint the encoder (slow, first 2 MB)")
    ap.add_argument("--csv", help="write one row per stream")
    ap.add_argument("--no-raw", action="store_true",
                    help="skip files with no deflate stream instead of compressing them whole (experimental)")
    a = ap.parse_args()
    missing = []
    files = list(walk(a.paths, missing))
    for p in missing:
        print(f"error: no such file or directory: {p}", file=sys.stderr)
    if not files:
        sys.exit("error: nothing to scan" if missing else "error: no files found under the given paths")
    fields = ["file", "depth", "id", "parent", "fmt", "offset", "comp", "content", "match", "trials", "lzma",
              "note", "diag"]
    out = open(a.csv, "w", newline="") if a.csv else None
    w = csv.DictWriter(out, fieldnames=fields) if out else None
    if w:
        w.writeheader()
    budget = (a.mem_mb or 0.6 * mem_available_mb()) * 1e6
    jobs = a.j or max(1, min(os.cpu_count() or 1, int(budget // 400e6)))
    share = budget / jobs
    # per worker: inflated content at up to 3 nesting levels + LZMA (about 11x its dictionary)
    content_limit = a.max_mb * 1e6 if a.max_mb else min(256e6, share / 5)
    dict_limit = 1 << 20
    while dict_limit * 2 * 11 <= share / 2 and dict_limit < 64 << 20:
        dict_limit *= 2
    print(f"memory budget {budget/1e6:.0f} MB: {jobs} workers x {share/1e6:.0f} MB, LZMA dict <= "
          f"{dict_limit >> 20} MiB, streams over {content_limit/1e6:.0f} MB skipped", file=sys.stderr)
    t0, rows, seen, nwarn = time.time(), [], set(), 0
    init = (content_limit, dict_limit, int(max(share, 200e6) + 100e6))
    with Pool(jobs, initializer=init_worker, initargs=init, maxtasksperchild=16) as pool:
        # rows are written as they arrive: long runs lose nothing
        for i, rs in enumerate(pool.imap_unordered(scan_file, [(f, (a.quick, a.diagnose, not a.no_raw)) for f in files], 1)):
            rows += rs; seen.update(r["file"].split(" > ")[0] for r in rs if r["fmt"] != "raw")
            for r in rs:
                if r["fmt"] == "raw":
                    nwarn += 1
                    if nwarn <= 10:
                        print(f"\rwarning: {r['file']}: no deflate stream, compressing the whole file "
                              f"(experimental)", file=sys.stderr)
                    elif nwarn == 11:
                        print("\rwarning: more files with no deflate stream; see the table and CSV",
                              file=sys.stderr)
            if w:
                w.writerows(rs); out.flush()
            if sys.stderr.isatty():
                print(f"\r{i+1}/{len(files)} files", end="", file=sys.stderr)
    if sys.stderr.isatty():
        print(file=sys.stderr)
    if out:
        out.close()
    report(rows, len(files), len(files) - len(seen), time.time() - t0)

if __name__ == "__main__":
    main()
