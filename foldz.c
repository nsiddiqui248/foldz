/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 the foldz authors
 *
 * foldz 0.9 - lossless compressor for GGUF models (file format 9)
 *
 * The version and the file format share one number: foldz 0.9 writes and reads format 9
 * (magic "FOLDZ09"). Until 1.0 the format may change between versions; a file is only
 * guaranteed to decompress with the version that made it. Check with: foldz v
 *
 * Works on any GGUF file: tensors of the supported types are coded with
 * context models learned from the data itself; everything else (metadata,
 * other tensor types, padding) is stored as is. Decompression gives back the
 * identical file, byte for byte. Nothing depends on the model's name or
 * architecture.
 *
 * Every other byte (metadata, tensor index, padding, unsupported tensor
 * types) is coded with a byte-level context-mixing model.
 *
 * Supported types and models
 *   Q4_K, Q5_K  values: the folds (Gray code g = n ^ (n >> 1)), every fold bit
 *               coded given the lower ones, context: zero point, extremes seen
 *               (0 and max), position. Header: d, dmin (fp16), scales, mins.
 *   Q6_K        values: context = column offset and width (running mean and
 *               spread of the column, in units of the sub-block's step),
 *               position, extremes seen; or, if smaller, the AWQ lattice
 *               (models converted from 4-bit AWQ/GPTQ checkpoints).
 *   Q8_0        values: context = column sign bias, extremes seen, average
 *               magnitude so far in the block. Header: d (fp16).
 *   F32, F16, BF16  sign, exponent (context: previous exponent), mantissa
 *               (top bits given the exponent).
 * Safety: every segment whose coded size would not be smaller is stored raw,
 * so the output is never larger than the input (apart from a small index).
 *
 * Coder: binary adaptive range coder (LZMA style), probabilities updated like
 * the KT estimator. Big tensors are split into segments that are coded
 * independently, in parallel. Contexts use only exact or IEEE-754 arithmetic
 * (no libm functions except exact ones), so files decode identically on any
 * platform.
 *
 * Build:  gcc -O2 -pthread -ffp-contract=off -o foldz foldz.c -lm
 *         (clang, e.g. Termux: same flags). Requires IEEE double arithmetic without excess
 *         precision or fused multiply-add, so that every platform computes the same contexts:
 *         64-bit x86, ARM and RISC-V are fine; 32-bit x86 needs -msse2 -mfpmath=sse.
 *         The build stops with an error where this can't be guaranteed (x87 maths, -ffast-math).
 * Platforms: Linux, Android (Termux), macOS, BSD. Windows: use WSL (needs mmap and pwrite).
 * Usage:  foldz c model.gguf model.foldz [-t threads] [-s segment_MiB]
 *         foldz d model.foldz model.gguf [-t threads]
 *         foldz t model.gguf [-t threads] [-s segment_MiB]   (round trip in memory)
 */
#define _FILE_OFFSET_BITS 64
#define _GNU_SOURCE
#include <float.h>
#if defined(_WIN32) && !defined(__CYGWIN__)
#error "foldz needs POSIX mmap and pwrite: build it under WSL (or Cygwin)"
#endif
#if defined(__FAST_MATH__)
#error "do not build foldz with -ffast-math: contexts must be computed identically everywhere"
#endif
#if !defined(FLT_EVAL_METHOD) || FLT_EVAL_METHOD != 0
#error "floating point uses excess precision (x87): on 32-bit x86 build with -msse2 -mfpmath=sse"
#endif
/* no fused multiply-add, even if -ffp-contract=off was forgotten (results would differ across CPUs) */
#if defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#elif defined(__GNUC__)
#pragma GCC optimize ("fp-contract=off")
#endif
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>

#define FOLDZ_VERSION "0.9"
static const char MAGIC[8] = {'F','O','L','D','Z','0','9','\n'};    /* file format 9 */
enum { T_F32 = 0, T_F16 = 1, T_Q8_0 = 8, T_Q4K = 12, T_Q5K = 13, T_Q6K = 14, T_BF16 = 30, T_BYTES = 0xFFFF };
enum { M_NORMAL = 0, M_LATTICE = 1, M_RAW = 255 };

typedef struct { uint32_t id; int vpb, bs; const char *name; } TypeInfo;
static const TypeInfo TYPES[] = {
    {T_Q4K, 256, 144, "Q4_K"}, {T_Q5K, 256, 176, "Q5_K"}, {T_Q6K, 256, 210, "Q6_K"},
    {T_Q8_0, 32, 34, "Q8_0"}, {T_BF16, 1, 2, "BF16"}, {T_F16, 1, 2, "F16"}, {T_F32, 1, 4, "F32"},
    {T_BYTES, 1, 1, "other"},            /* metadata, tensor index, padding, unsupported tensor types */
};
#define NTYPES (int)(sizeof TYPES / sizeof TYPES[0])
static int type_index(uint32_t id) { for (int i = 0; i < NTYPES; i++) if (TYPES[i].id == id) return i; return -1; }

static void die(const char *m) { fprintf(stderr, "foldz: %s\n", m); exit(1); }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static void *xmalloc(size_t n) { void *p = malloc(n ? n : 1); if (!p) die("out of memory (try fewer threads: -t 2)"); return p; }

/* ---------------- adaptive bit model ---------------- */
typedef struct { uint16_t p; uint16_t n; } Bit;      /* p = P(bit = 0) * 65536 */
#define NLIM 1022
static uint32_t RATE[NLIM + 1];
static double LOG2T[66];
static void init_tables(void) {
    for (int n = 0; n <= NLIM; n++) RATE[n] = 65536u / (n + 2);
    for (int i = 1; i < 66; i++) LOG2T[i] = log2((double)i);   /* encoder-side estimates only */
}
static void bits_init(Bit *b, size_t n) { for (size_t i = 0; i < n; i++) { b[i].p = 32768; b[i].n = 0; } }
static Bit *bits_new(size_t n) { Bit *b = xmalloc(n * sizeof *b); bits_init(b, n); return b; }
static inline void bit_update(Bit *b, int bit) {
    uint32_t k = RATE[b->n]; int p = b->p;
    if (!bit) p += (int)(((uint32_t)(65536 - p) * k) >> 16);
    else      p -= (int)(((uint32_t)p * k) >> 16);
    if (p < 64) p = 64;
    if (p > 65472) p = 65472;
    b->p = (uint16_t)p; if (b->n < NLIM) b->n++;
}

/* ---------------- range coder (encoder and decoder in one) ---------------- */
typedef struct {
    int dec;
    uint64_t low; uint32_t range; uint8_t cache; uint64_t csize;
    uint8_t *buf; size_t pos, cap;
    uint32_t code; const uint8_t *in, *end;
} Coder;

static void put(Coder *c, uint8_t b) {
    if (c->pos == c->cap) { c->cap = c->cap * 2 + 65536; c->buf = realloc(c->buf, c->cap); if (!c->buf) die("out of memory"); }
    c->buf[c->pos++] = b;
}
static void shift_low(Coder *c) {
    if ((uint32_t)c->low < 0xFF000000u || (c->low >> 32) != 0) {
        uint8_t carry = (uint8_t)(c->low >> 32), t = c->cache;
        do { put(c, (uint8_t)(t + carry)); t = 0xFF; } while (--c->csize != 0);
        c->cache = (uint8_t)((uint32_t)c->low >> 24);
    }
    c->csize++;
    c->low = (uint64_t)((uint32_t)c->low << 8);
}
static inline uint8_t next(Coder *c) { return c->in < c->end ? *c->in++ : 0; }
static void enc_init(Coder *c) { memset(c, 0, sizeof *c); c->range = 0xFFFFFFFFu; c->csize = 1; }
static void enc_flush(Coder *c) { for (int i = 0; i < 5; i++) shift_low(c); }
static void dec_init(Coder *c, const uint8_t *p, size_t n) {
    memset(c, 0, sizeof *c); c->dec = 1; c->in = p; c->end = p + n; c->range = 0xFFFFFFFFu;
    for (int i = 0; i < 5; i++) c->code = (c->code << 8) | next(c);
}
static inline void norm(Coder *c) {
    while (c->range < (1u << 24)) {
        c->range <<= 8;
        if (c->dec) c->code = (c->code << 8) | next(c); else shift_low(c);
    }
}
static inline int cbit(Coder *c, Bit *b, int bit) {
    uint32_t bound = (c->range >> 16) * b->p;
    if (c->dec) {
        if (c->code < bound) { c->range = bound; bit = 0; }
        else { c->code -= bound; c->range -= bound; bit = 1; }
    } else {
        if (!bit) c->range = bound; else { c->low += bound; c->range -= bound; }
    }
    bit_update(b, bit); norm(c);
    return bit;
}
static inline int cdirect(Coder *c, int bit) {
    c->range >>= 1;
    if (c->dec) { bit = 0; if (c->code >= c->range) { c->code -= c->range; bit = 1; } }
    else if (bit) c->low += c->range;
    norm(c);
    return bit;
}
/* nbits symbol through a binary tree t[1 .. 2^nbits - 1], high bit first */
static inline int ctree(Coder *c, Bit *t, int nbits, int sym) {
    int node = 1;
    for (int i = nbits - 1; i >= 0; i--) node = (node << 1) | cbit(c, &t[node], (sym >> i) & 1);
    return node - (1 << nbits);
}

/* ---------------- helpers ---------------- */
static float h2f(uint16_t h) {                     /* exact fp16 -> float */
    uint32_t s = h >> 15, e = (h >> 10) & 31, m = h & 1023; float v;
    if (e == 0) v = ldexpf((float)m, -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = ldexpf((float)(m | 1024), (int)e - 25);
    return s ? -v : v;
}
static inline uint16_t rd16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static inline void wr16(uint8_t *p, uint16_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }
static inline int sgn16(uint16_t d) { return (d & 0x7FFF) == 0 ? 0 : ((d >> 15) ? -1 : 1); }
static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : v > hi ? hi : v; }

/* fp16 as high byte (context: previous high byte) + low byte (context: high byte) */
static uint16_t code_f16(Coder *c, Bit *hi_m, Bit *lo_m, int *prev_hi, uint16_t v) {
    int hi = ctree(c, hi_m + (*prev_hi) * 256, 8, v >> 8);
    int lo = ctree(c, lo_m + hi * 256, 8, v & 255);
    *prev_hi = hi;
    return (uint16_t)((hi << 8) | lo);
}
/* share of positive weights so far in a column -> 8 buckets (edges .3 .4 .45 .5 .55 .6 .7) */
static inline int share_bucket(int64_t pos, int64_t tot) {
    static const int E[7] = {30, 40, 45, 50, 55, 60, 70};
    int b = 0; int64_t x = 100 * (pos + 1);
    for (int i = 0; i < 7; i++) b += x >= (int64_t)E[i] * (tot + 2);
    return b;
}

/* ---------------- Q4_K and Q5_K ---------------- */
/* block: d(2) dmin(2) scales(12) [Q5_K: qh(32)] qs(128); 8 sub-blocks of 32 */
static void k_unpack_scales(const uint8_t *s, int *sc, int *mn) {
    for (int j = 0; j < 4; j++) { sc[j] = s[j] & 63; mn[j] = s[j + 4] & 63; }
    for (int j = 4; j < 8; j++) {
        sc[j] = (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4);
        mn[j] = (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
    }
}
static void k_pack_scales(uint8_t *s, const int *sc, const int *mn) {
    for (int j = 0; j < 4; j++) {
        s[j]     = (uint8_t)(sc[j] | ((sc[j + 4] >> 4) << 6));
        s[j + 4] = (uint8_t)(mn[j] | ((mn[j + 4] >> 4) << 6));
        s[j + 8] = (uint8_t)((sc[j + 4] & 0xF) | ((mn[j + 4] & 0xF) << 4));
    }
}
static void k_unpack_vals(const uint8_t *B, int bits, uint8_t *q) {
    const uint8_t *qs = B + (bits == 4 ? 16 : 48), *qh = B + 16;
    for (int j = 0; j < 4; j++) for (int l = 0; l < 32; l++) {
        q[64 * j + l]      = qs[32 * j + l] & 15;
        q[64 * j + 32 + l] = qs[32 * j + l] >> 4;
        if (bits == 5) {
            q[64 * j + l]      |= ((qh[l] >> (2 * j)) & 1) << 4;
            q[64 * j + 32 + l] |= ((qh[l] >> (2 * j + 1)) & 1) << 4;
        }
    }
}
static void k_pack_vals(uint8_t *B, int bits, const uint8_t *q) {
    uint8_t *qs = B + (bits == 4 ? 16 : 48), *qh = B + 16;
    if (bits == 5) memset(qh, 0, 32);
    for (int j = 0; j < 4; j++) for (int l = 0; l < 32; l++) {
        qs[32 * j + l] = (uint8_t)((q[64 * j + l] & 15) | ((q[64 * j + 32 + l] & 15) << 4));
        if (bits == 5)
            qh[l] |= (uint8_t)((((q[64 * j + l] >> 4) & 1) << (2 * j)) | (((q[64 * j + 32 + l] >> 4) & 1) << (2 * j + 1)));
    }
}

static void kq_code(Coder *c, uint8_t *data, uint64_t nblocks, int bits) {
    int bs = bits == 4 ? 144 : 176, maxv = (1 << bits) - 1, zmax = bits == 4 ? 31 : 63;
    size_t nctx = (size_t)(zmax + 1) * 36;
    Bit *dhi = bits_new(65536), *dlo = bits_new(65536), *mhi = bits_new(65536), *mlo = bits_new(65536);
    Bit *msc = bits_new(64 * 64), *mmn = bits_new(4096 * 64), *val = bits_new(nctx << bits);
    int pd = 0, pm = 0, psc = 0, pmn = 0;
    for (uint64_t b = 0; b < nblocks; b++) {
        uint8_t *B = data + b * bs;
        int sc[8], mn[8], zb[8]; uint8_t q[256];
        if (!c->dec) { k_unpack_scales(B + 4, sc, mn); k_unpack_vals(B, bits, q); }
        uint16_t d  = code_f16(c, dhi, dlo, &pd, c->dec ? 0 : rd16(B));
        uint16_t dm = code_f16(c, mhi, mlo, &pm, c->dec ? 0 : rd16(B + 2));
        for (int j = 0; j < 8; j++) {
            sc[j] = ctree(c, msc + psc * 64, 6, sc[j]); psc = sc[j];
            mn[j] = ctree(c, mmn + (sc[j] * 64 + pmn) * 64, 6, mn[j]); pmn = mn[j];
        }
        float df = h2f(d), mf = h2f(dm);
        for (int j = 0; j < 8; j++) {
            float step = df * (float)sc[j];
            float z = step > 0 ? mf * (float)mn[j] / step : 0.0f;
            zb[j] = (z == z && z < 1e6f) ? clampi((int)rintf(z * 2.0f), 0, zmax) : 0;
        }
        for (int s = 0; s < 8; s++) {
            int n0 = 0, nm = 0;
            for (int i = 0; i < 32; i++) {
                int idx = s * 32 + i;
                size_t ctx = (size_t)(zb[s] * 9 + (n0 < 2 ? n0 : 2) * 3 + (nm < 2 ? nm : 2)) * 4 + (i >> 3);
                Bit *t = val + (ctx << bits);
                /* the folds: g = n ^ (n >> 1); bit k is coded given bits 0..k-1 */
                int g = q[idx] ^ (q[idx] >> 1), node = 1, gd = 0;
                for (int k = 0; k < bits; k++) {
                    int bk = cbit(c, &t[node], (g >> k) & 1);
                    gd |= bk << k; node = (node << 1) | bk;
                }
                int n = gd; n ^= n >> 1; n ^= n >> 2; n ^= n >> 4;     /* undo the folds */
                q[idx] = (uint8_t)n; n0 += n == 0; nm += n == maxv;
            }
        }
        if (c->dec) {
            wr16(B, d); wr16(B + 2, dm); k_pack_scales(B + 4, sc, mn); k_pack_vals(B, bits, q);
        }
    }
    free(dhi); free(dlo); free(mhi); free(mlo); free(msc); free(mmn); free(val);
}

/* ---------------- Q6_K ---------------- */
/* block: ql(128) qh(64) scales(16, int8) d(2); 16 sub-blocks of 16 */
static void q6_unpack(const uint8_t *B, uint8_t *q) {
    for (int h = 0; h < 2; h++) {
        const uint8_t *ql = B + 64 * h, *qh = B + 128 + 32 * h; uint8_t *y = q + 128 * h;
        for (int l = 0; l < 32; l++) {
            y[l]      = (uint8_t)((ql[l] & 15)      | (((qh[l] >> 0) & 3) << 4));
            y[l + 32] = (uint8_t)((ql[l + 32] & 15) | (((qh[l] >> 2) & 3) << 4));
            y[l + 64] = (uint8_t)((ql[l] >> 4)      | (((qh[l] >> 4) & 3) << 4));
            y[l + 96] = (uint8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4));
        }
    }
}
static void q6_pack(uint8_t *B, const uint8_t *q) {
    for (int h = 0; h < 2; h++) {
        uint8_t *ql = B + 64 * h, *qh = B + 128 + 32 * h; const uint8_t *y = q + 128 * h;
        for (int l = 0; l < 32; l++) {
            ql[l]      = (uint8_t)((y[l] & 15)      | ((y[l + 64] & 15) << 4));
            ql[l + 32] = (uint8_t)((y[l + 32] & 15) | ((y[l + 96] & 15) << 4));
            qh[l] = (uint8_t)((y[l] >> 4) | ((y[l + 32] >> 4) << 2) | ((y[l + 64] >> 4) << 4) | ((y[l + 96] >> 4) << 6));
        }
    }
}
/* AWQ lattice: y = q - 32 = round(k * r) + e, r = span / kmax (encoder picks kmax) */
static void lattice_fit(const int *y, int *km_out, int *span_out, int *k, int *e) {
    int ext = 0; for (int i = 0; i < 16; i++) if (y[i] < ext) ext = y[i];
    int span = ext < 0 ? -ext : 32;
    double best = 1e30; int bm = span < 32 ? span : 32;
    for (int m = 1; m <= 32 && m <= span; m++) {
        double r = (double)span / m, inv = m / (double)span; int nz = 0;
        for (int i = 0; i < 16; i++) { int kk = (int)rint(y[i] * inv); nz += y[i] != (int)rint(kk * r); }
        double est = 16 * LOG2T[2 * m + 1] + 8.0 * nz;
        if (est < best) { best = est; bm = m; }
        if (nz == 0) break;
    }
    double r = (double)span / bm, inv = bm / (double)span;
    for (int i = 0; i < 16; i++) { k[i] = (int)rint(y[i] * inv); e[i] = y[i] - (int)rint(k[i] * r); }
    *km_out = bm; *span_out = ext < 0 ? -ext : 0;
}
static int lattice_sample(const uint8_t *data, uint64_t nblocks) {   /* median kmax of a sample */
    int hist[33] = {0}, n = 0;
    uint64_t step = nblocks / 32 ? nblocks / 32 : 1;
    for (uint64_t b = 0; b < nblocks && n < 512; b += step) {
        uint8_t q[256]; q6_unpack(data + b * 210, q);
        for (int sb = 0; sb < 16; sb++) {
            int y[16], k[16], e[16], km, es;
            for (int i = 0; i < 16; i++) y[i] = q[sb * 16 + i] - 32;
            lattice_fit(y, &km, &es, k, e); hist[km]++; n++;
        }
    }
    int acc = 0; for (int m = 1; m <= 32; m++) { acc += hist[m]; if (2 * acc >= n) return m; }
    return 32;
}

#define Q6_CTX 9600                    /* 25 offset x 8 width x 16 pos x 3 extremes */
static void q6k_code(Coder *c, uint8_t *data, uint64_t rows, uint64_t ne0, int lat) {
    Bit *dhi = bits_new(65536), *dlo = bits_new(65536), *msc = bits_new(65536);
    Bit *val = lat ? NULL : bits_new((size_t)Q6_CTX * 64);
    Bit *km_m = bits_new(33 * 64), *ex_m = bits_new(9 * 64), *k_m = bits_new(66 * 128), *e_m = bits_new(33 * 8);
    double *SW = lat ? NULL : calloc(ne0, sizeof *SW), *SW2 = lat ? NULL : calloc(ne0, sizeof *SW2);
    if (!lat && (!SW || !SW2)) die("out of memory");
    uint64_t bpr = ne0 / 256; int pd = 0, psc = 128, lkm = 0;
    for (uint64_t r = 0; r < rows; r++) {
        for (uint64_t bb = 0; bb < bpr; bb++) {
            uint8_t *B = data + (r * bpr + bb) * 210;
            uint8_t q[256]; int scs[16];
            uint16_t d = code_f16(c, dhi, dlo, &pd, c->dec ? 0 : rd16(B + 208));
            for (int k = 0; k < 16; k++) {
                int sym = c->dec ? 0 : (int)(int8_t)B[192 + k] + 128;
                sym = ctree(c, msc + psc * 256, 8, sym); psc = sym; scs[k] = sym;
            }
            if (!c->dec) q6_unpack(B, q);
            if (lat) {
                for (int sb = 0; sb < 16; sb++) {
                    int y[16], kk[16], ee[16], km = 0, es = 0;
                    if (!c->dec) { for (int i = 0; i < 16; i++) y[i] = q[sb * 16 + i] - 32;
                                   lattice_fit(y, &km, &es, kk, ee); }
                    if (bb == 0 && sb == 0) lkm = 0;
                    km = ctree(c, km_m + lkm * 64, 6, km); lkm = km;
                    if (km < 1) km = 1;                                /* corrupt input guard */
                    es = ctree(c, ex_m + (km >> 2) * 64, 6, es);
                    double rr = (double)(es ? es : 32) / km;
                    int seen = 0;
                    for (int i = 0; i < 16; i++) {
                        int k = ctree(c, k_m + (km * 2 + (seen > 0)) * 128, 7, c->dec ? 0 : kk[i] + 32) - 32;
                        int sym = c->dec ? 0 : (ee[i] >= -3 && ee[i] <= 3 ? ee[i] + 3 : 7);
                        sym = ctree(c, e_m + km * 8, 3, sym);
                        int v;
                        if (sym == 7) { v = 0; for (int b2 = 5; b2 >= 0; b2--) v |= cdirect(c, (q[sb * 16 + i] >> b2) & 1) << b2; }
                        else v = 32 + (int)rint(k * rr) + sym - 3;
                        q[sb * 16 + i] = (uint8_t)(v & 63); seen += k == -km;
                    }
                }
            } else {
                double dd = (double)h2f(d), D[16];
                for (int k = 0; k < 16; k++) D[k] = dd * (double)(scs[k] - 128);   /* signed step, exact */
                int nx = 0;
                for (int i = 0; i < 256; i++) {
                    int sb = i >> 4, pos = i & 15; uint64_t col = bb * 256 + i;
                    if (pos == 0) nx = 0;
                    int offb = 12, widb = 7; double Ds = D[sb];
                    if (r > 0 && Ds != 0 && isfinite(Ds)) {
                        /* running mean and variance of the column (rows above), in units of Ds */
                        double m = SW[col] / (double)r, t = m / Ds;
                        if (isfinite(t)) offb = clampi((int)rint(t < -100 ? -100 : t > 100 ? 100 : t), -12, 12) + 12;
                        double var = SW2[col] / (double)r - m * m, r2 = var / (Ds * Ds);
                        if (isfinite(r2) && r2 > 0) {
                            int e; double mant = frexp(r2, &e);           /* exact */
                            widb = clampi((mant >= 0.70710678118654752 ? e : e - 1) - 2, 0, 7);
                        } else widb = 0;
                    }
                    int ctx = (offb * 8 + widb) * 48 + pos * 3 + (nx < 2 ? nx : 2);
                    int v = ctree(c, val + (size_t)ctx * 64, 6, q[i]);
                    q[i] = (uint8_t)v; nx += v <= 1;
                    double W = (v - 32) * Ds;
                    if (isfinite(W)) { SW[col] += W; SW2[col] += W * W; }
                }
            }
            if (c->dec) {
                q6_pack(B, q);
                for (int k = 0; k < 16; k++) B[192 + k] = (uint8_t)(int8_t)(scs[k] - 128);
                wr16(B + 208, d);
            }
        }
    }
    free(dhi); free(dlo); free(msc); free(val); free(km_m); free(ex_m); free(k_m); free(e_m); free(SW); free(SW2);
}

/* ---------------- Q8_0 ---------------- */
/* block: d(2) + 32 int8; w = d * q; the quantizer maps the largest |w| to +-127 */
#define Q8_CTX (8 * 3 * 16)
static void q80_code(Coder *c, uint8_t *data, uint64_t rows, uint64_t ne0) {
    Bit *dhi = bits_new(65536), *dlo = bits_new(65536), *val = bits_new((size_t)Q8_CTX * 256);
    int64_t *P = calloc(ne0, sizeof *P); if (!P) die("out of memory");
    uint64_t bpr = ne0 / 32; int pd = 0;
    for (uint64_t r = 0; r < rows; r++) {
        for (uint64_t bb = 0; bb < bpr; bb++) {
            uint8_t *B = data + (r * bpr + bb) * 34;
            uint16_t d = code_f16(c, dhi, dlo, &pd, c->dec ? 0 : rd16(B));
            int ds = sgn16(d), nx = 0, sa = 0;
            for (int i = 0; i < 32; i++) {
                uint64_t col = bb * 32 + i;
                int cb = share_bucket(P[col], (int64_t)r); if (ds < 0) cb = 7 - cb;
                int avgb = i ? clampi((sa / i) >> 3, 0, 15) : 0;
                int ctx = (cb * 3 + (nx < 2 ? nx : 2)) * 16 + avgb;
                int v = ctree(c, val + (size_t)ctx * 256, 8, c->dec ? 0 : (int)(int8_t)B[2 + i] + 128) - 128;
                nx += v >= 127 || v <= -127; sa += v < 0 ? -v : v;
                P[col] += (int64_t)v * ds > 0;
                if (c->dec) B[2 + i] = (uint8_t)(int8_t)v;
            }
            if (c->dec) wr16(B, d);
        }
    }
    free(dhi); free(dlo); free(val); free(P);
}

/* ---------------- F32, F16, BF16 ---------------- */
static void float_code(Coder *c, uint8_t *data, uint64_t n, uint32_t type) {
    Bit *sg = bits_new(4), *ex = bits_new(64 * 256), *mh = bits_new(64 * 256), *m1 = bits_new(256), *m0 = bits_new(256);
    int ps = 0, pe = 0;
    for (uint64_t i = 0; i < n; i++) {
        if (type == T_F32) {
            uint8_t *p = data + i * 4; uint32_t v = 0;
            if (!c->dec) memcpy(&v, p, 4);
            int s = cbit(c, &sg[ps], v >> 31); ps = s;
            int e = ctree(c, ex + clampi(pe - 96, 0, 63) * 256, 8, (v >> 23) & 255); pe = e;
            int eb = clampi(e - 96, 0, 63);
            uint32_t mt = (uint32_t)ctree(c, mh + eb * 256, 7, (v >> 16) & 127);
            uint32_t b1 = (uint32_t)ctree(c, m1, 8, (v >> 8) & 255), b0 = (uint32_t)ctree(c, m0, 8, v & 255);
            v = ((uint32_t)s << 31) | ((uint32_t)e << 23) | (mt << 16) | (b1 << 8) | b0;
            if (c->dec) memcpy(p, &v, 4);
        } else {
            uint8_t *p = data + i * 2; uint16_t v = c->dec ? 0 : rd16(p);
            int s = cbit(c, &sg[ps], v >> 15); ps = s;
            if (type == T_BF16) {
                int e = ctree(c, ex + clampi(pe - 96, 0, 63) * 256, 8, (v >> 7) & 255); pe = e;
                int m = ctree(c, mh + clampi(e - 96, 0, 63) * 256, 7, v & 127);
                v = (uint16_t)((s << 15) | (e << 7) | m);
            } else {                                                     /* F16 */
                int e = ctree(c, ex + pe * 256, 5, (v >> 10) & 31); pe = e;
                int mt = ctree(c, mh + e * 256, 3, (v >> 7) & 7);
                int ml = ctree(c, m1, 7, v & 127);
                v = (uint16_t)((s << 15) | (e << 10) | (mt << 7) | ml);
            }
            if (c->dec) wr16(p, v);
        }
    }
    free(sg); free(ex); free(mh); free(m1); free(m0);
}

/* ---------------- other bytes: context mixing ---------------- */
/* Predictors with the previous 0, 1, 2, 3, 4 and 6 bytes as context (orders 2+ hashed),
   mixed in the logistic domain by a small neural mixer chosen by the bits of the current
   byte seen so far. Integer arithmetic only, so results are identical on every platform. */
static int SQT[33] = {1, 2, 3, 6, 10, 16, 27, 45, 73, 120, 194, 310, 488, 747, 1101, 1546, 2047,
                      2549, 2994, 3348, 3607, 3785, 3901, 3975, 4024, 4050, 4068, 4079, 4085, 4089, 4092, 4093, 4094};
static int squash(int d) {                     /* stretch domain (x256) -> probability (12 bits) */
    if (d > 2047) d = 2047;
    if (d < -2047) d = -2047;
    int w = d & 127, i = (d >> 7) + 16;
    return (SQT[i] * (128 - w) + SQT[i + 1] * w + 64) >> 7;
}
static short STR[4096];                        /* inverse of squash */
static void init_stretch(void) {
    int pi = 0;
    for (int x = -2047; x <= 2047; x++) { int v = squash(x); for (int j = pi; j <= v; j++) STR[j] = (short)x; pi = v + 1; }
    for (int j = pi; j < 4096; j++) STR[j] = 2047;
}
static inline void bit_update_fast(Bit *b, int bit) {      /* KT-like, but adapts to changing data */
    uint32_t k = RATE[b->n]; int p = b->p;
    if (!bit) p += (int)(((uint32_t)(65536 - p) * k) >> 16);
    else      p -= (int)(((uint32_t)p * k) >> 16);
    if (p < 64) p = 64;
    if (p > 65472) p = 65472;
    b->p = (uint16_t)p; if (b->n < 20) b->n++;
}
static inline int cbit_p(Coder *c, uint32_t p0, int bit) {   /* bit with a given P(0) * 65536 */
    uint32_t bound = (c->range >> 16) * p0;
    if (c->dec) {
        if (c->code < bound) { c->range = bound; bit = 0; }
        else { c->code -= bound; c->range -= bound; bit = 1; }
    } else {
        if (!bit) c->range = bound; else { c->low += bound; c->range -= bound; }
    }
    norm(c);
    return bit;
}
#define NIN 8
static Bit *bits_new_aligned(size_t n) {
    void *p = NULL; if (posix_memalign(&p, 64, n * sizeof(Bit))) die("out of memory");
    bits_init(p, n); return p;
}
/* bucket of 16 counters (one cache line) for a context and the bits of the current nibble */
static inline uint32_t bucket(uint32_t h, int hb) { return ((h * 0x85EBCA6Bu) >> (32 - (hb - 4))) << 4; }
static void bytes_code(Coder *c, uint8_t *data, uint64_t n) {
    int hb = 16; while (hb < 23 && ((uint64_t)1 << hb) < n * 2) hb++;
    Bit *o0 = bits_new(256), *o1 = bits_new(65536);
    Bit *h[4]; for (int i = 0; i < 4; i++) h[i] = bits_new_aligned((size_t)1 << hb);
    int32_t *W = xmalloc(256 * 3 * NIN * sizeof *W);
    for (int i = 0; i < 256 * 3 * NIN; i++) W[i] = 19661;      /* 0.3 */
    uint64_t hist = 0;
    int mb = hb - 2; uint32_t *mt = calloc((size_t)1 << mb, sizeof *mt); if (!mt) die("out of memory");
    Bit *mm = bits_new(64 * 2); uint64_t ptr = 0; int mlen = 0;
    for (uint64_t i = 0; i < n; i++) {
        int byte = c->dec ? 0 : data[i], node = 1;
        uint32_t c1 = (uint32_t)(hist & 0xFF);
        uint32_t hs[4] = {(uint32_t)(hist & 0xFFFF) * 0x2545F491u + 1, (uint32_t)(hist & 0xFFFFFF) * 0x9E3779B1u + 2,
                          (uint32_t)(hist & 0xFFFFFFFFu) * 0x85EBCA77u + 3,
                          ((uint32_t)(hist & 0xFFFFFFFFu) * 0xC2B2AE3Du) ^ ((uint32_t)((hist >> 32) & 0xFFFF) * 0x27D4EB2Fu) ^ 4};
        uint32_t bk[4];
        int pb = mlen ? data[ptr] : 0;
        for (int k = 7; k >= 0; k--) {
            if (k == 7 || k == 3) {                            /* new nibble: one cache line per context */
                uint32_t hi = k == 3 ? (uint32_t)node * 0x01000193u : 0;
                for (int m = 0; m < 4; m++) { bk[m] = bucket(hs[m] ^ hi, hb); __builtin_prefetch(&h[m][bk[m]]); }
            }
            int ok = mlen && ((pb | 256) >> (k + 1)) == node, eb = (pb >> k) & 1;
            Bit *bm = &mm[(ok ? (mlen < 31 ? mlen : 31) + 1 : 0) * 2 + (ok ? eb : 0)];
            /* position inside the nibble: 1..15 (high nibble: node, low nibble: 1 + bits seen of it) */
            int in = k >= 4 ? node : (1 << (3 - k)) | (node & ((1 << (3 - k)) - 1));
            Bit *b[6] = {&o0[node], &o1[c1 * 256 + node],
                         &h[0][bk[0] + in], &h[1][bk[1] + in], &h[2][bk[2] + in], &h[3][bk[3] + in]};
            int st[NIN]; int64_t dot = 0; int32_t *w = W + node * NIN;
            for (int m = 0; m < 6; m++) { int p1 = (65536 - b[m]->p) >> 4; st[m] = STR[p1 < 1 ? 1 : p1 > 4095 ? 4095 : p1]; }
            { int p1m = (65536 - bm->p) >> 4; st[6] = ok ? STR[p1m < 1 ? 1 : p1m > 4095 ? 4095 : p1m] : 0; }
            st[7] = 256;
            w += (size_t)(ok ? (mlen < 16 ? 1 : 2) : 0) * 256 * NIN;
            for (int m = 0; m < NIN; m++) dot += (int64_t)st[m] * w[m];
            int p1 = squash((int)(dot >> 16));
            uint32_t p0 = (uint32_t)(4096 - p1) << 4;
            if (p0 < 64) p0 = 64;
            if (p0 > 65472) p0 = 65472;
            int bit = cbit_p(c, p0, (byte >> k) & 1);
            int err = (bit << 12) - p1;
            for (int m = 0; m < NIN; m++) w[m] += (st[m] * err) >> 10;
            for (int m = 0; m < 6; m++) bit_update_fast(b[m], bit);
            bit_update_fast(bm, bit);
            node = (node << 1) | bit;
        }
        byte = node - 256;
        if (c->dec) data[i] = (uint8_t)byte;
        hist = (hist << 8) | (uint64_t)byte;
        if (mlen && data[ptr] == byte) { mlen++; ptr++; } else mlen = 0;
        if (i + 1 >= 4) {
            uint32_t hh = (uint32_t)(((hist & 0xFFFFFFFFull) * 0x9E3779B97F4A7C15ull) >> (64 - mb));
            if (!mlen && mt[hh]) {
                ptr = mt[hh]; mlen = 0;
                while (mlen < 32 && ptr > (uint64_t)mlen && data[ptr - 1 - mlen] == data[i - mlen]) mlen++;
                if (!mlen) ptr = 0;
            }
            mt[hh] = (uint32_t)(i + 1);
        }
    }
    free(o0); free(o1); for (int i = 0; i < 4; i++) free(h[i]); free(W); free(mt); free(mm);
}

/* ---------------- segments ---------------- */
typedef struct { uint64_t off, size, ne0, rows; uint32_t type, model; } Seg;
static uint64_t seg_bytes(const Seg *s) {                 /* untrusted on decompress: check everything */
    int ti = type_index(s->type); uint64_t a, b;
    if (ti < 0 || !s->ne0 || !s->rows || s->ne0 % TYPES[ti].vpb) die("corrupt index");
    if (__builtin_mul_overflow(s->rows, s->ne0 / TYPES[ti].vpb, &a) ||
        __builtin_mul_overflow(a, (uint64_t)TYPES[ti].bs, &b) ||
        __builtin_mul_overflow(s->rows, s->ne0, &a)) die("corrupt index");
    if (s->type == T_Q6K && s->ne0 % 256) die("corrupt index");
    return b;
}
static void code_seg(Coder *c, Seg *s, uint8_t *p) {
    switch (s->type) {
        case T_Q4K: kq_code(c, p, s->size / 144, 4); break;
        case T_Q5K: kq_code(c, p, s->size / 176, 5); break;
        case T_Q6K: q6k_code(c, p, s->rows, s->ne0, s->model == M_LATTICE); break;
        case T_Q8_0: q80_code(c, p, s->rows, s->ne0); break;
        case T_BYTES: bytes_code(c, p, s->size); break;
        default: float_code(c, p, s->rows * s->ne0, s->type);
    }
}

/* ---------------- GGUF parsing ---------------- */
typedef struct { const uint8_t *p; size_t n, pos; } Rd;
static void need(Rd *r, size_t k) { if (k > r->n || r->pos > r->n - k) die("truncated or invalid GGUF"); }
static uint32_t ru32(Rd *r) { need(r, 4); uint32_t v; memcpy(&v, r->p + r->pos, 4); r->pos += 4; return v; }
static uint64_t ru64(Rd *r) { need(r, 8); uint64_t v; memcpy(&v, r->p + r->pos, 8); r->pos += 8; return v; }
static void skip_val(Rd *r, uint32_t t, int depth) {
    static const int SZ[13] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
    if (t > 12 || depth > 4) die("unknown GGUF value type");
    if (t == 8) { uint64_t l = ru64(r); need(r, l); r->pos += l; return; }
    if (t == 9) {
        uint32_t et = ru32(r); uint64_t cnt = ru64(r);
        if (et <= 12 && et != 8 && et != 9) { if (cnt > r->n) die("bad array"); need(r, cnt * SZ[et]); r->pos += cnt * SZ[et]; }
        else for (uint64_t i = 0; i < cnt; i++) skip_val(r, et, depth + 1);
        return;
    }
    need(r, SZ[t]); r->pos += SZ[t];
}
static int cmp_seg(const void *a, const void *b) {
    uint64_t x = ((const Seg *)a)->off, y = ((const Seg *)b)->off; return x < y ? -1 : x > y;
}
static Seg *find_segments(const uint8_t *p, size_t n, uint64_t seg_target, int *nseg) {
    Rd r = {p, n, 0};
    if (n < 24 || memcmp(p, "GGUF", 4)) {           /* any other file: the byte model alone */
        Seg *s = NULL; int cnt = 0; uint64_t bt = (uint64_t)64 << 20;
        for (uint64_t a = 0; a < n; a += bt) {
            uint64_t len = n - a < bt ? n - a : bt;
            s = realloc(s, (cnt + 1) * sizeof *s); if (!s) die("out of memory");
            s[cnt++] = (Seg){a, len, len, 1, T_BYTES, M_NORMAL};
        }
        *nseg = cnt; return s;
    }
    r.pos = 4; uint32_t ver = ru32(&r); if (ver < 2) die("GGUF version 1 is not supported");
    uint64_t nt = ru64(&r), nkv = ru64(&r), align = 32;
    if (nt > n || nkv > n) die("invalid GGUF header");
    for (uint64_t i = 0; i < nkv; i++) {
        uint64_t kl = ru64(&r); need(&r, kl);
        int is_align = kl == 17 && !memcmp(p + r.pos, "general.alignment", 17);
        r.pos += kl; uint32_t t = ru32(&r);
        if (is_align && t == 4) { align = ru32(&r); if (!align) align = 32; } else skip_val(&r, t, 0);
    }
    Seg *s = NULL; int cnt = 0, cap = 0;
    uint64_t *offs = xmalloc(nt * 8), *ne0s = xmalloc(nt * 8), *nes = xmalloc(nt * 8);
    uint32_t *types = xmalloc(nt * 4);
    for (uint64_t i = 0; i < nt; i++) {
        uint64_t kl = ru64(&r); need(&r, kl); r.pos += kl;
        uint32_t nd = ru32(&r); if (nd > 8) die("bad tensor dims");
        uint64_t ne = 1, e0 = 1;
        for (uint32_t k = 0; k < nd; k++) {
            uint64_t d = ru64(&r); if (k == 0) e0 = d;
            if (__builtin_mul_overflow(ne, d, &ne)) die("bad tensor dims");
        }
        types[i] = ru32(&r); offs[i] = ru64(&r); ne0s[i] = e0; nes[i] = ne;
    }
    uint64_t data = (r.pos + align - 1) / align * align;
    for (uint64_t i = 0; i < nt; i++) {
        int ti = type_index(types[i]); if (ti < 0) continue;
        uint64_t e0 = ne0s[i], ne = nes[i]; int vpb = TYPES[ti].vpb, bs = TYPES[ti].bs;
        if (!e0 || !ne || e0 % vpb || ne % e0) continue;
        uint64_t rows = ne / e0, rowb = e0 / vpb * bs, off, tb;
        if (__builtin_add_overflow(data, offs[i], &off) || __builtin_mul_overflow(rows, rowb, &tb)
            || off > n || tb > n - off) die("tensor beyond end of file");
        uint64_t rps = seg_target / rowb; if (rps < 1) rps = 1;
        for (uint64_t r0 = 0; r0 < rows; r0 += rps) {
            uint64_t nr = rows - r0 < rps ? rows - r0 : rps;
            if (cnt == cap) { cap = cap * 2 + 64; s = realloc(s, cap * sizeof *s); if (!s) die("out of memory"); }
            s[cnt] = (Seg){off + r0 * rowb, nr * rowb, e0, nr, types[i], M_NORMAL}; cnt++;
        }
    }
    free(offs); free(ne0s); free(nes); free(types);
    qsort(s, cnt, sizeof *s, cmp_seg);
    for (int i = 1; i < cnt; i++) if (s[i].off < s[i - 1].off + s[i - 1].size) die("overlapping tensors");
    int nt_seg = cnt; uint64_t at = 0;
    for (int i = 0; i <= nt_seg; i++) {
        uint64_t end = i < nt_seg ? s[i].off : n;
        uint64_t bt = (uint64_t)64 << 20;           /* keep metadata whole: the match model needs its history */
        for (uint64_t a = at; a < end; a += bt) {
            uint64_t len = end - a < bt ? end - a : bt;
            if (cnt == cap) { cap = cap * 2 + 64; s = realloc(s, cap * sizeof *s); if (!s) die("out of memory"); }
            s[cnt++] = (Seg){a, len, len, 1, T_BYTES, M_NORMAL};
        }
        if (i < nt_seg) at = s[i].off + s[i].size;
    }
    qsort(s, cnt, sizeof *s, cmp_seg);
    *nseg = cnt; return s;
}

/* ---------------- parallel driver ---------------- */
/* container: magic | u64 orig size | u32 nseg
              | nseg x (u64 off, u64 ne0, u64 rows, u32 type | model << 16, u64 clen, u64 coff)
              | raw bytes between segments | coded segments at their coff */
#define ENTRY 44
typedef struct {
    int mode;                          /* 0 compress, 1 decompress, 2 test */
    Seg *seg; int n; int next, failed, done; int *order;
    const uint8_t *in; int fd;
    uint64_t *clen, *coff;
    uint64_t end; pthread_mutex_t mu; double t0;
    double tsum[16], tmax; int tmax_seg;           /* thread-seconds per type, slowest segment */
} Job;

static void pwrite_all(int fd, const void *p, uint64_t n, uint64_t at) {
    const uint8_t *q = p;
    while (n) {
        ssize_t w = pwrite(fd, q, n > (1u << 30) ? (1u << 30) : n, (off_t)at);
        if (w <= 0) die("write failed (disk full?)");
        q += w; n -= (uint64_t)w; at += (uint64_t)w;
    }
}
static void *worker(void *arg) {
    Job *j = arg;
    for (;;) {
        int qi = __atomic_fetch_add(&j->next, 1, __ATOMIC_RELAXED);
        if (qi >= j->n) break;
        int i = j->order[qi];
        Seg *s = &j->seg[i]; Coder c; double ts = now();
        if (j->mode != 1) {
            const uint8_t *src = j->in + s->off;
            s->model = M_NORMAL;
            enc_init(&c); code_seg(&c, s, (uint8_t *)src); enc_flush(&c);
            if (s->type == T_Q6K && lattice_sample(src, s->size / 210) < 20) {
                Coder c2; s->model = M_LATTICE;
                enc_init(&c2); code_seg(&c2, s, (uint8_t *)src); enc_flush(&c2);
                if (c2.pos < c.pos) { free(c.buf); c = c2; } else { free(c2.buf); s->model = M_NORMAL; }
            }
            if (c.pos >= s->size) {                          /* safety: never larger, store raw */
                free(c.buf); c.buf = xmalloc(s->size); memcpy(c.buf, src, s->size); c.pos = s->size; s->model = M_RAW;
            }
            j->clen[i] = c.pos;
            if (j->mode == 0) {
                pthread_mutex_lock(&j->mu); uint64_t at = j->end; j->end += c.pos; pthread_mutex_unlock(&j->mu);
                pwrite_all(j->fd, c.buf, c.pos, at); j->coff[i] = at;
            } else if (s->model != M_RAW) {
                uint8_t *buf = xmalloc(s->size);
                Coder d; dec_init(&d, c.buf, c.pos); code_seg(&d, s, buf);
                if (memcmp(buf, src, s->size)) __atomic_store_n(&j->failed, 1, __ATOMIC_RELAXED);
                free(buf);
            }
            free(c.buf);
        } else {
            uint8_t *buf = xmalloc(s->size);
            if (s->model == M_RAW) { if (j->clen[i] != s->size) die("corrupt file"); memcpy(buf, j->in + j->coff[i], s->size); }
            else { dec_init(&c, j->in + j->coff[i], j->clen[i]); code_seg(&c, s, buf); }
            pwrite_all(j->fd, buf, s->size, s->off); free(buf);
        }
        double dt = now() - ts;
        pthread_mutex_lock(&j->mu);
        j->tsum[type_index(s->type)] += dt;
        if (dt > j->tmax) { j->tmax = dt; j->tmax_seg = i; }
        pthread_mutex_unlock(&j->mu);
        int d = __atomic_add_fetch(&j->done, 1, __ATOMIC_RELAXED);
        fprintf(stderr, "\r  %d/%d segments, %.0f s", d, j->n, now() - j->t0);
    }
    return NULL;
}
static double seg_cost(const Seg *s) { return (double)s->size * (s->type == T_BYTES ? 12 : s->type == T_Q6K ? 2 : 1); }
static Seg *SORT_SEGS;
static int cmp_cost(const void *a, const void *b) {
    double x = seg_cost(&SORT_SEGS[*(const int *)a]), y = seg_cost(&SORT_SEGS[*(const int *)b]);
    return x > y ? -1 : x < y;
}
static void run(Job *j, int threads) {
    j->order = xmalloc((j->n + 1) * sizeof(int));
    for (int i = 0; i < j->n; i++) j->order[i] = i;
    SORT_SEGS = j->seg; qsort(j->order, j->n, sizeof(int), cmp_cost);
    j->next = 0; j->done = 0; j->t0 = now(); pthread_mutex_init(&j->mu, NULL);
    pthread_t th[256]; if (threads > 256) threads = 256; if (threads < 1) threads = 1;
    if (threads > j->n && j->n > 0) threads = j->n;
    for (int i = 0; i < threads; i++) pthread_create(&th[i], NULL, worker, j);
    for (int i = 0; i < threads; i++) pthread_join(th[i], NULL);
    fprintf(stderr, "\n"); free(j->order);
}
static void time_report(Job *j, double wall, int threads) {
    double tot = 0; for (int k = 0; k < NTYPES; k++) tot += j->tsum[k];
    printf("  time: %.0f s on %d threads, %.0f thread-seconds of work (%.0f%% use)\n   ", wall, threads, tot,
           100.0 * tot / (wall * threads));
    for (int k = 0; k < NTYPES; k++) if (j->tsum[k] > 0) printf(" %s %.0f s", TYPES[k].name, j->tsum[k]);
    Seg *m = &j->seg[j->tmax_seg];
    printf("\n    slowest segment: %.1f s (%s, %.1f MiB)\n", j->tmax, TYPES[type_index(m->type)].name, m->size / 1048576.0);
}
static const uint8_t *map_file(const char *path, size_t *n) {
    int fd = open(path, O_RDONLY); if (fd < 0) die("cannot open input");
    struct stat st; fstat(fd, &st); *n = (size_t)st.st_size;
    if (!*n) die("empty input");
    void *p = mmap(NULL, *n, PROT_READ, MAP_PRIVATE, fd, 0); close(fd);
    if (p == MAP_FAILED) die("cannot map input");
    return p;
}
static void report(Seg *s, int n, uint64_t *clen, uint64_t total, uint64_t out_total) {
    uint64_t ob[NTYPES] = {0}, cb[NTYPES] = {0}, vals[NTYPES] = {0};
    int nraw = 0, nlat = 0, n6 = 0;
    for (int i = 0; i < n; i++) {
        int k = type_index(s[i].type); ob[k] += s[i].size; cb[k] += clen[i]; vals[k] += s[i].rows * s[i].ne0;
        nraw += s[i].model == M_RAW; if (s[i].type == T_Q6K) { n6++; nlat += s[i].model == M_LATTICE; }
    }
    for (int k = 0; k < NTYPES; k++) if (ob[k])
        printf("  %-5s %9.1f MiB -> %9.1f MiB  (%5.2f%% saved, %.3f bits/%s)\n", TYPES[k].name,
               ob[k] / 1048576.0, cb[k] / 1048576.0, 100.0 * (1 - (double)cb[k] / ob[k]), cb[k] * 8.0 / vals[k],
               TYPES[k].id == T_BYTES ? "byte (metadata etc.)" : "weight incl. headers");

    if (n6) printf("  Q6_K segments using the AWQ lattice: %d of %d\n", nlat, n6);
    if (nraw) printf("  segments stored raw (coding would not shrink them): %d\n", nraw);
    printf("  file  %9.1f MiB -> %9.1f MiB  (%5.2f%% saved)\n", total / 1048576.0, out_total / 1048576.0,
           100.0 * (1 - (double)out_total / total));
}
static int clamp_threads(int threads, Seg *s, int ns) {
    uint64_t mx = 0; for (int i = 0; i < ns; i++) if (s[i].size > mx) mx = s[i].size;
    if (threads > ns && ns > 0) threads = ns;
    if (threads < 1) threads = 1;
    printf("%d threads, largest segment %.0f MiB, peak memory about %.0f MiB\n",
           threads, mx / 1048576.0, threads * (3.1 * mx + 4e6) / 1048576.0 + 160);
    return threads;
}

static void cmd_compress(const char *in, const char *out, int threads, uint64_t segb, int test) {
    size_t n; const uint8_t *p = map_file(in, &n);
    int ns; Seg *s = find_segments(p, n, segb, &ns);
    printf("%s: %.1f MiB, %d segments to code\n", in, n / 1048576.0, ns);
    threads = clamp_threads(threads, s, ns);
    Job j; memset(&j, 0, sizeof j);
    j.mode = test ? 2 : 0; j.seg = s; j.n = ns; j.in = p; j.fd = -1;
    j.clen = calloc(ns + 1, 8); j.coff = calloc(ns + 1, 8);
    uint64_t seg_total = 0; for (int i = 0; i < ns; i++) seg_total += s[i].size;
    uint64_t hdr = 20 + (uint64_t)ns * ENTRY;
    if (!test) {
        j.fd = open(out, O_RDWR | O_CREAT | O_TRUNC, 0644); if (j.fd < 0) die("cannot create output");
        uint64_t at = 0, w = hdr;
        for (int i = 0; i <= ns; i++) {
            uint64_t end = i < ns ? s[i].off : n;
            pwrite_all(j.fd, p + at, end - at, w); w += end - at;
            if (i < ns) at = s[i].off + s[i].size;
        }
        j.end = w;
    }
    double t0 = now(); run(&j, threads); double t1 = now();
    uint64_t ctotal = 0; for (int i = 0; i < ns; i++) ctotal += j.clen[i];
    uint64_t out_total = hdr + (n - seg_total) + ctotal;
    if (test) printf("compressed and decompressed in %.1f s\n", t1 - t0);
    else printf("compressed in %.1f s (%.1f MiB/s)\n", t1 - t0, n / 1048576.0 / (t1 - t0));
    report(s, ns, j.clen, n, out_total);
    time_report(&j, t1 - t0, threads);
    if (test) {
        printf("round trip: %s\n", j.failed ? "FAILED, data differs" : "OK, identical");
        if (j.failed) exit(2);
        return;
    }
    uint8_t *h = calloc(1, hdr); uint64_t v = n; uint32_t c32 = (uint32_t)ns;
    memcpy(h, MAGIC, 8); memcpy(h + 8, &v, 8); memcpy(h + 16, &c32, 4);
    for (int i = 0; i < ns; i++) {
        uint8_t *e = h + 20 + (uint64_t)i * ENTRY; uint32_t tm = s[i].type | (s[i].model << 16);
        memcpy(e, &s[i].off, 8); memcpy(e + 8, &s[i].ne0, 8); memcpy(e + 16, &s[i].rows, 8);
        memcpy(e + 24, &tm, 4); memcpy(e + 28, &j.clen[i], 8); memcpy(e + 36, &j.coff[i], 8);
    }
    pwrite_all(j.fd, h, hdr, 0); close(j.fd); free(h);
}

static void cmd_decompress(const char *in, const char *out, int threads) {
    size_t n; const uint8_t *p = map_file(in, &n);
    if (n < 20) die("not a foldz file");
    if (memcmp(p, MAGIC, 8)) {
        if (!memcmp(p, "FOLDZ", 5) && p[7] == '\n')
            fprintf(stderr, "foldz: file format %.2s, but foldz " FOLDZ_VERSION " reads only format 9: "
                            "use the foldz version that made it\n", (const char *)p + 5), exit(1);
        die("not a foldz file");
    }
    uint64_t orig; uint32_t ns; memcpy(&orig, p + 8, 8); memcpy(&ns, p + 16, 4);
    uint64_t pos = 20; if ((uint64_t)ns * ENTRY > n - pos) die("corrupt header");
    Seg *s = calloc(ns + 1, sizeof *s); uint64_t *clen = calloc(ns + 1, 8), *coff = calloc(ns + 1, 8);
    for (uint32_t i = 0; i < ns; i++) {
        const uint8_t *e = p + pos; uint32_t tm;
        memcpy(&s[i].off, e, 8); memcpy(&s[i].ne0, e + 8, 8); memcpy(&s[i].rows, e + 16, 8);
        memcpy(&tm, e + 24, 4); memcpy(&clen[i], e + 28, 8); memcpy(&coff[i], e + 36, 8);
        s[i].type = tm & 0xFFFF; s[i].model = tm >> 16;
        if (type_index(s[i].type) < 0 || !s[i].ne0) die("corrupt header");
        s[i].size = seg_bytes(&s[i]); pos += ENTRY;
        if (coff[i] > n || clen[i] > n - coff[i] || s[i].off > orig || s[i].size > orig - s[i].off) die("corrupt header");
    }
    threads = clamp_threads(threads, s, (int)ns);
    int fd = open(out, O_RDWR | O_CREAT | O_TRUNC, 0644); if (fd < 0) die("cannot create output");
    uint64_t at = 0;
    for (uint32_t i = 0; i <= ns; i++) {
        uint64_t end = i < ns ? s[i].off : orig;
        if (end < at || end - at > n - pos) die("corrupt file");
        pwrite_all(fd, p + pos, end - at, at); pos += end - at;
        if (i < ns) at = s[i].off + s[i].size;
    }
    Job j; memset(&j, 0, sizeof j);
    j.mode = 1; j.seg = s; j.n = (int)ns; j.in = p; j.fd = fd; j.clen = clen; j.coff = coff;
    double t0 = now(); run(&j, threads); double t1 = now();
    if (ftruncate(fd, (off_t)orig)) die("cannot size output");
    close(fd);
    printf("decompressed %.1f MiB in %.1f s (%.1f MiB/s)\n", orig / 1048576.0, t1 - t0, orig / 1048576.0 / (t1 - t0));
    time_report(&j, t1 - t0, threads);
}

/* output name: given, or derived from the input (in.gguf -> in.gguf.foldz, in.gguf.foldz -> in.gguf) */
static char *out_name(const char *in, const char *given, int decompress) {
    if (given) return strdup(given);
    size_t n = strlen(in), k = strlen(".foldz");
    char *o = xmalloc(n + 8);
    if (decompress) {
        if (n > k && !strcmp(in + n - k, ".foldz")) { memcpy(o, in, n - k); o[n - k] = 0; }
        else { memcpy(o, in, n); memcpy(o + n, ".out", 5); }
    } else { memcpy(o, in, n); memcpy(o + n, ".foldz", k + 1); }
    printf("output: %s\n", o); fflush(stdout);
    return o;
}
/* never overwrite the input; never overwrite anything else unless -f */
static void check_output(const char *in, const char *out, int force) {
    struct stat a, b;
    if (stat(out, &b) != 0) return;                         /* does not exist yet */
    if (stat(in, &a) == 0 && a.st_dev == b.st_dev && a.st_ino == b.st_ino)
        die("output is the input file itself");
    if (!force) {
        fprintf(stderr, "foldz: %s already exists (use -f to overwrite)\n", out);
        exit(1);
    }
}

int main(int argc, char **argv) {
    init_tables(); init_stretch();
    if (argc == 2 && (argv[1][0] == 'v' || !strcmp(argv[1], "--version"))) {
        printf("foldz " FOLDZ_VERSION " (file format 9)\n"); return 0;
    }
    if (argc < 3) {
        fprintf(stderr, "usage: foldz c in [out.foldz] [-t threads] [-s segment_MiB] [-f]\n"
                        "       foldz d in.foldz [out] [-t threads] [-f]\n"
                        "       foldz t in [-t threads] [-s segment_MiB]   (round-trip test, no files)\n"
                        "       foldz v                                    (version)\n"
                        "  out defaults to in + .foldz (c) or in without .foldz (d); -f overwrites an existing file\n"
                        "  defaults: all cores, 48 MiB segments; lower -t (or -s) to use less memory\n");
        return 1;
    }
    int threads = (int)sysconf(_SC_NPROCESSORS_ONLN), force = 0; uint64_t segb = 48;
    const char *pos[2] = {NULL, NULL}; int np = 0;
    for (int i = 2; i < argc; i++) {                         /* options anywhere, in any order */
        if (!strcmp(argv[i], "-t") || !strcmp(argv[i], "-s")) {
            if (i + 1 >= argc) die("option needs a value");
            long long v = atoll(argv[++i]);
            if (argv[i - 1][1] == 't') threads = (int)v; else segb = (uint64_t)(v < 1 ? 1 : v);
        } else if (!strcmp(argv[i], "-f")) force = 1;
        else if (argv[i][0] == '-' && argv[i][1]) { fprintf(stderr, "foldz: unknown option %s\n", argv[i]); return 1; }
        else if (np < 2) pos[np++] = argv[i];
        else die("too many file names");
    }
    if (!pos[0]) die("missing input file");
    segb <<= 20;
    char cmd = argv[1][0];
    if ((cmd == 'c' || cmd == 'd') && argv[1][1] == 0) {
        char *out = out_name(pos[0], pos[1], cmd == 'd');
        check_output(pos[0], out, force);
        if (cmd == 'c') cmd_compress(pos[0], out, threads, segb, 0); else cmd_decompress(pos[0], out, threads);
        free(out);
    } else if (cmd == 't' && argv[1][1] == 0) {
        if (pos[1]) die("t takes no output name");
        cmd_compress(pos[0], NULL, threads, segb, 1);
    } else die("unknown command (use c, d, t or v)");
    return 0;
}
