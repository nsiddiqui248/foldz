# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 the foldz authors
# Handover experiment: blind vs format-aware recompression (needs numpy).
import numpy as np, zlib, bz2, lzma, glob
from math import lgamma, log
def kt_bits(counts):                      # adaptive KT cost of binary counts (C,2)
    c = np.asarray(counts, float); n = c.sum(1)
    lg = np.vectorize(lgamma)
    return float((lg(n + 1) - lg(1.0) - (lg(c[:, 0] + .5) - lg(.5)) - (lg(c[:, 1] + .5) - lg(.5))).sum() / log(2))
def H0(b):
    c = np.bincount(np.frombuffer(b, np.uint8), minlength=256); p = c[c > 0] / len(b); return -(p * np.log2(p)).sum()
def bit_ctx_cost(b, k):                   # bit-level model: each bit given the previous k bits
    bits = np.unpackbits(np.frombuffer(b, np.uint8)).astype(np.int64)
    ctx = np.zeros_like(bits)
    for j in range(1, k + 1): ctx[j:] |= bits[:-j] << (j - 1)
    t = np.bincount(ctx * 2 + bits, minlength=2 << k).reshape(-1, 2)
    return kt_bits(t[t.sum(1) > 0]) / len(b)
def byte_bit_cost(b):                     # bits of a byte given its higher bits + the previous byte (order-1 CM)
    x = np.frombuffer(b, np.uint8).astype(np.int64); prev = np.r_[0, x[:-1]]; tot = 0
    for i in range(7, -1, -1):
        partial = (x >> (i + 1)) | (1 << (7 - i))          # bits already seen in this byte
        ctx = prev * 256 + partial; bit = (x >> i) & 1
        t = np.bincount(ctx * 2 + bit, minlength=65536 * 2).reshape(-1, 2); tot += kt_bits(t[t.sum(1) > 0])
    return tot / len(b)
gray = lambda b: (np.frombuffer(b, np.uint8) ^ (np.frombuffer(b, np.uint8) >> 1)).astype(np.uint8).tobytes()

raw = b"".join(open(f, "rb").read() for f in sorted(glob.glob("/usr/lib/python3*/*.py"))[:60])[:2_000_000]
outs = {"gzip -6 (deflate)": zlib.compress(raw, 6), "bzip2 -9": bz2.compress(raw, 9), "xz -6 (LZMA)": lzma.compress(raw)}
print(f"text: {len(raw)/1e6:.2f} MB\n")
print(f"{'compressed with':18} {'size':>8} | bits per bit after:  {'H0 byte':>7} {'Gray':>6} {'16 bits':>7} {'byte+prev':>9} {'xz again':>8}")
for name, b in outs.items():
    n = len(b)
    print(f"{name:18} {n:8d} | {'':19} {H0(b)/8:7.4f} {H0(gray(b))/8:6.4f} {bit_ctx_cost(b,16)/8:7.4f} "
          f"{byte_bit_cost(b)/8:9.4f} {len(lzma.compress(b, preset=9))/n:8.4f}")
# format-aware: undo deflate, model the content, and prove the exact bytes can be rebuilt
z = outs["gzip -6 (deflate)"]
rebuilt = zlib.compress(zlib.decompress(z), 6)
inner = lzma.compress(zlib.decompress(z), preset=9)
print(f"\ngzip file, format-aware: inflate -> xz the content; rebuild with deflate level 6: "
      f"{'identical' if rebuilt == z else 'DIFFERENT'}")
print(f"  {len(z)} -> {len(inner) + 2} bytes ({(1 - (len(inner) + 2)/len(z))*100:.1f}% smaller, 2 bytes to store the deflate settings)")
