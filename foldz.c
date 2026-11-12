/*
 * foldz 0.14 - lossless compressor for GGUF models (file format 14)
 *
 * The version and the file format share one number: foldz 0.14 writes and reads format 14
 * (magic "FOLDZ14"). Until 1.0 the format may change between versions; a file is only
 * guaranteed to decompress with the version that made it. Check with: foldz v
 *
 * Works on GGUF and safetensors files (and, with the byte model alone, any other file): tensors of the supported types are coded with
 * context models learned from the data itself; every other byte (metadata,
 * tensor index, padding, unsupported tensor types) is coded with a byte-level
 * context-mixing model. Decompression gives back the identical file, byte for
 * byte. Nothing depends on the model's name or architecture.
 *
 * Supported types and models
 *   Q4_K, Q5_K  values: whole symbols, context: zero point, extremes seen
 *               (0 and max), position. Header: d, dmin (fp16), scales, mins.
 *   Q6_K        values: context = column offset and width (running mean and
 *               spread of the column, in units of the sub-block's step),
 *               position, low values seen (whole symbols); or, if smaller, the AWQ lattice
 *               (lattice steps and errors also as symbols)
 *               (models converted from 4-bit AWQ/GPTQ checkpoints).
 *   Q8_0        values: context = column sign bias, extremes seen, average
 *               magnitude so far in the block. Header: d (fp16).
 *   F32, F16, BF16  sign, exponent (context: previous exponent), mantissa
 *               (top bits given the exponent).
 * Safety: every segment whose coded size would not be smaller is stored raw,
 * so the output is never larger than the input (apart from a small index).
 *
 * Coder: range coder (LZMA style). Q4_K, Q5_K and Q6_K values are coded as whole
 * symbols, Q8_0 values as two 4-bit symbols (high, then low given high), F32, F16 and
 * BF16 values as 4-bit symbols (each 16 bits as four, each given the ones above), all
 * from static tables: the compressor first counts every segment, builds one
 * seed table per type for the whole file, and each segment then stores, per context,
 * either "use the seed" or its own table as differences from the seed. Decoding needs
 * no learning, so it is faster than compression. Everything else is adaptive: AWQ
 * lattice values as symbols, the rest bit by bit (probabilities updated like the KT
 * estimator). With -H the block headers of
 * Q4_K, Q5_K and Q6_K are also coded as symbols: faster, about 0.05% larger. Big tensors are split into segments that are coded
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
 * Usage:  foldz c model.gguf model.foldz [-t threads] [-s segment_MiB] [-m MiB] [-H]
 *         foldz d model.foldz model.gguf [-t threads] [-m MiB]
 *         foldz t model.gguf [-t threads] [-s segment_MiB] [-m MiB] [-H]   (round trip in memory)
 * Memory: segments start only while their estimated memory fits in what the device has free,
 *         keeping a reserve for the system and other apps; -m sets a fixed limit instead.
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
#include <inttypes.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <mach/mach.h>
#elif defined(__FreeBSD__) || defined(__DragonFly__)
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

#define FOLDZ_VERSION "0.14"
static const char MAGIC[8] = {'F','O','L','D','Z','1','4','\n'};    /* file format 14 */
enum { T_F32 = 0, T_F16 = 1, T_Q8_0 = 8, T_Q4K = 12, T_Q5K = 13, T_Q6K = 14, T_BF16 = 30, T_BYTES = 0xFFFF };
enum { M_NORMAL = 0, M_LATTICE = 1, M_RAW = 255 };
/* flag in the mode field: block headers coded as symbols (-H). Faster to decode, slightly larger. */
enum { H_SYM = 0x100 };

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

/* The probability update selects instead of branching (measured faster). The coder itself keeps
   its branch: a branchless version was measured slower, because it lengthens the chain from one
   decoded bit to the next. */
#define HOT static inline __attribute__((always_inline))
HOT void bit_update(Bit *b, int bit) {
    uint32_t k = RATE[b->n]; int p = b->p;
    int up = (int)(((uint32_t)(65536 - p) * k) >> 16), dn = (int)(((uint32_t)p * k) >> 16);
    p = bit ? p - dn : p + up;
    p = p < 64 ? 64 : p;
    p = p > 65472 ? 65472 : p;
    b->p = (uint16_t)p; b->n = (uint16_t)(b->n + (b->n < NLIM));
}

/* ---------------- range coder ---------------- */
/* One set of functions for both directions. Every model takes `dec` as a constant (each model
   is compiled twice, see MODEL below), so the encoder and decoder each get their own copy of
   the inner loops with no direction test left in them. */
typedef struct {
    int dec;
    uint64_t low; uint32_t range; uint8_t cache; uint64_t csize;
    uint8_t *buf; size_t pos, cap;            /* encoder: bytes past cap are counted, not stored */
    uint32_t code; const uint8_t *in, *end;
    const int16_t *stt;                       /* static value tables (cumulative, sum 2^12), or NULL */
    uint32_t *cnt; int cnt_only;              /* encoder: count values per context (pass 1) */
} Coder;

/* The output buffer is allocated once at the segment's size. Output that reaches that size is
   stored raw anyway, so later bytes only need counting: no realloc, and memory is known in advance. */
static void put(Coder *c, uint8_t b) { if (c->pos < c->cap) c->buf[c->pos] = b; c->pos++; }
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
static void enc_init(Coder *c, size_t cap) {
    memset(c, 0, sizeof *c); c->range = 0xFFFFFFFFu; c->csize = 1;
    c->cap = cap; c->buf = xmalloc(cap);
}
static void enc_flush(Coder *c) { for (int i = 0; i < 5; i++) shift_low(c); }
static void dec_init(Coder *c, const uint8_t *p, size_t n) {
    memset(c, 0, sizeof *c); c->dec = 1; c->in = p; c->end = p + n; c->range = 0xFFFFFFFFu;
    for (int i = 0; i < 5; i++) c->code = (c->code << 8) | next(c);
}
HOT void norm(Coder *c, const int dec) {
    while (c->range < (1u << 24)) {
        c->range <<= 8;
        if (dec) c->code = (c->code << 8) | next(c); else shift_low(c);
    }
}
/* code one bit with probability p0 = P(0) * 65536; the encoder's bit must be 0 or 1 */
HOT int code_bit(Coder *c, const int dec, uint32_t p0, int bit) {
    uint32_t bound = (c->range >> 16) * p0;
    if (dec) {
        if (c->code < bound) { c->range = bound; bit = 0; }
        else { c->code -= bound; c->range -= bound; bit = 1; }
    } else {
        if (!bit) c->range = bound; else { c->low += bound; c->range -= bound; }
    }
    norm(c, dec);
    return bit;
}
HOT int cbit(Coder *c, const int dec, Bit *b, int bit) {
    bit = code_bit(c, dec, b->p, bit);
    bit_update(b, bit);
    return bit;
}
HOT int cdirect(Coder *c, const int dec, int bit) {
    c->range >>= 1;
    if (dec) bit = c->code >= c->range;
    uint32_t m = 0u - (uint32_t)bit;
    if (dec) c->code -= c->range & m; else c->low += c->range & m;
    norm(c, dec);
    return bit;
}
/* nbits symbol through a binary tree t[1 .. 2^nbits - 1], high bit first */
HOT int ctree(Coder *c, const int dec, Bit *t, int nbits, int sym) {
    int node = 1;
    for (int i = nbits - 1; i >= 0; i--) node = (node << 1) | cbit(c, dec, &t[node], (sym >> i) & 1);
    return node - (1 << nbits);
}
/* ---- whole-symbol coding ----
   One coder step per value instead of one per bit. A context holds the cumulative counts of
   its n symbols: U[i] = count of symbols 0..i, so U[n-1] is the total. Each symbol starts at 1
   and gains SYM_INC per occurrence (the KT estimator); when the total passes SYM_LIMIT, all
   counts are halved, which sets how fast old data is forgotten. 2^14 was measured best on real
   models (2^10 and 2^12 lost 0.5% and 0.1%; 2^14 matched the bit-wise coder of format 9). Counts stay below 2^15, so the
   loops below are plain int16 compares and adds that compilers turn into SIMD. */
#define SYM_INC 2
#define SYM_LIMIT (1 << 14)
static int16_t *sym_new(size_t nctx, int n) {             /* one spare element in front: U[-1] is readable */
    int16_t *t = xmalloc((nctx * n + 1) * sizeof *t); t[0] = 0;
    for (size_t c = 0; c < nctx; c++) for (int i = 0; i < n; i++) t[1 + c * n + i] = (int16_t)(i + 1);
    return t + 1;
}
static void sym_free(int16_t *t) { if (t) free(t - 1); }
static void sym_halve(int16_t *U, int n) {
    int prev = 0, acc = 0;
    for (int i = 0; i < n; i++) { int f = U[i] - prev; prev = U[i]; acc += (f + 1) >> 1; U[i] = (int16_t)acc; }
}
/* The search and the update below work on 8 counts at a time with the GCC/Clang generic vector
   type. It is not tied to any CPU: the compiler turns it into SSE2 or AVX on x86, NEON on ARM,
   the vector extension on RISC-V, or ordinary scalar code where there is none. Only integer
   adds and compares are used, so every CPU produces the same bytes. FOLDZ_SCALAR (or a compiler
   without the extension) selects plain loops that give identical output. */
#if !defined(FOLDZ_SCALAR) && (defined(__GNUC__) || defined(__clang__))
#define SYM_VEC 1
typedef int16_t V8 __attribute__((vector_size(16)));
static const V8 IOTA8 = {0, 1, 2, 3, 4, 5, 6, 7};
#endif
HOT int csym(Coder *c, const int dec, int16_t *U, const int n, int sym) {
    uint32_t tot = (uint32_t)U[n - 1], r = c->range / tot;
    if (dec) {
        uint32_t v = c->code / r; if (v >= tot) v = tot - 1;          /* clamp: corrupt input stays in range */
        /* symbol = number of cumulative counts <= v (the last one, the total, never is) */
#ifdef SYM_VEC
        V8 vv = IOTA8 * 0 + (int16_t)v, acc = IOTA8 * 0;
        for (int k = 0; k < n; k += 8) { V8 u; memcpy(&u, U + k, 16); acc -= (V8)(u <= vv); }
        uint64_t a[2]; memcpy(a, &acc, 16);                            /* add the 8 lanes; byte order does not matter */
        sym = (int)(((a[0] + a[1]) * 0x0001000100010001ull) >> 48);
#else
        sym = 0; for (int i = 0; i < n; i++) sym += U[i] <= (int16_t)v;
#endif
    }
    uint32_t lo = (uint32_t)U[sym - 1] & (0u - (uint32_t)(sym > 0)), f = (uint32_t)U[sym] - lo;
    if (dec) c->code -= r * lo; else c->low += r * lo;
    c->range = r * f;
    norm(c, dec);
#ifdef SYM_VEC
    V8 ss = IOTA8 * 0 + (int16_t)sym;                                  /* counts of sym and above += SYM_INC */
    for (int k = 0; k < n; k += 8) {
        V8 u; memcpy(&u, U + k, 16);
        u -= (V8)((IOTA8 + (int16_t)k) >= ss) * SYM_INC;
        memcpy(U + k, &u, 16);
    }
#else
    for (int i = 0; i < n; i++) U[i] = (int16_t)(U[i] + (i >= sym ? SYM_INC : 0));
#endif
    if (U[n - 1] > SYM_LIMIT) sym_halve(U, n);
    return sym;
}

/* a value from a static table: U holds cumulative frequencies summing to SM = 2^12, so the coder
   needs no division by a total and no update. Zero-frequency symbols are never chosen. */
#define SM_BITS 12
#define SM (1 << SM_BITS)
HOT int ssym(Coder *c, const int dec, const int16_t *U, const int n, int sym) {
    uint32_t r = c->range >> SM_BITS;
    if (dec) {
        uint32_t v = c->code / r; if (v > SM - 1) v = SM - 1;      /* clamp: corrupt input stays in range */
#ifdef SYM_VEC
        V8 vv = IOTA8 * 0 + (int16_t)v, acc = IOTA8 * 0;
        for (int k = 0; k < n; k += 8) { V8 u; memcpy(&u, U + k, 16); acc -= (V8)(u <= vv); }
        uint64_t a[2]; memcpy(a, &acc, 16);
        sym = (int)(((a[0] + a[1]) * 0x0001000100010001ull) >> 48);
#else
        sym = 0; for (int i = 0; i < n; i++) sym += U[i] <= (int16_t)v;
#endif
    }
    uint32_t lo = sym ? (uint32_t)U[sym - 1] : 0, f = (uint32_t)U[sym] - lo;
    if (dec) c->code -= r * lo; else c->low += r * lo;
    c->range = r * f;
    norm(c, dec);
    return sym;
}

/* values wider than 6 bits as two symbol steps, the low part given the high part.
   th has 2^(nb-4) counts per context, tl 16 per (context, high part). */
HOT int csym_split(Coder *c, const int dec, int16_t *th, int16_t *tl, const int nb, size_t ctx, int v) {
    const int nh = 1 << (nb - 4);
    int hi = csym(c, dec, th + ctx * nh, nh, v >> 4);
    int lo = csym(c, dec, tl + (ctx * nh + hi) * 16, 16, v & 15);
    return (hi << 4) | lo;
}
/* fp16 as before (high byte given the previous high byte, low byte given the high byte), with
   each byte as two 16-symbol steps. t: 4 tables, see f16s_new. */
typedef struct { int16_t *h1, *l1, *h2, *l2; } F16S;
static F16S f16s_new(void) { F16S f = { sym_new(256, 16), sym_new(4096, 16), sym_new(256, 16), sym_new(4096, 16) }; return f; }
static void f16s_free(F16S *f) { sym_free(f->h1); sym_free(f->l1); sym_free(f->h2); sym_free(f->l2); }
HOT uint16_t code_f16s(Coder *c, const int dec, F16S *f, int *prev_hi, uint16_t v) {
    int hi = csym_split(c, dec, f->h1, f->l1, 8, (size_t)*prev_hi, v >> 8);
    int lo = csym_split(c, dec, f->h2, f->l2, 8, (size_t)hi, v & 255);
    *prev_hi = hi;
    return (uint16_t)((hi << 8) | lo);
}

/* MODEL(name, args, call): name##_impl(Coder *c, const int dec, args) is the model;
   name(c, args) calls the encoder or the decoder copy of it */
#define X ,
#define MODEL(name, params, args) \
    static void name(Coder *c, params) { if (c->dec) name##_impl(c, 1, args); else name##_impl(c, 0, args); }

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
HOT uint16_t code_f16(Coder *c, const int dec, Bit *hi_m, Bit *lo_m, int *prev_hi, uint16_t v) {
    int hi = ctree(c, dec, hi_m + (*prev_hi) * 256, 8, v >> 8);
    int lo = ctree(c, dec, lo_m + hi * 256, 8, v & 255);
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

/* ss: values from static tables; cn: encoder pass 1, count values only (nothing is coded) */
HOT void kq_code_impl(Coder *c, const int dec, uint8_t *data, uint64_t nblocks, const int bits, const int sh, const int ss, const int cn) {
    int bs = bits == 4 ? 144 : 176, maxv = (1 << bits) - 1, zmax = bits == 4 ? 31 : 63;
    size_t nctx = (size_t)(zmax + 1) * 36;
    Bit *dhi = NULL, *dlo = NULL, *mhi = NULL, *mlo = NULL, *msc = NULL, *mmn = NULL;
    F16S fd = {0}, fm = {0}; int16_t *ssc = NULL, *smn = NULL;
    if (cn) {}
    else if (sh) { fd = f16s_new(); fm = f16s_new(); ssc = sym_new(64, 64); smn = sym_new(4096, 64); }
    else { dhi = bits_new(65536); dlo = bits_new(65536); mhi = bits_new(65536); mlo = bits_new(65536);
           msc = bits_new(64 * 64); mmn = bits_new(4096 * 64); }
    int16_t *sv = ss || cn ? NULL : sym_new(nctx, 1 << bits);
    int pd = 0, pm = 0, psc = 0, pmn = 0;
    for (uint64_t b = 0; b < nblocks; b++) {
        uint8_t *B = data + b * bs;
        int sc[8], mn[8], zb[8]; uint8_t q[256];
        if (!dec) { k_unpack_scales(B + 4, sc, mn); k_unpack_vals(B, bits, q); }
        uint16_t d, dm;
        if (cn) { d = rd16(B); dm = rd16(B + 2); }
        else if (sh) {
            d  = code_f16s(c, dec, &fd, &pd, dec ? 0 : rd16(B));
            dm = code_f16s(c, dec, &fm, &pm, dec ? 0 : rd16(B + 2));
            for (int j = 0; j < 8; j++) {
                sc[j] = csym(c, dec, ssc + psc * 64, 64, sc[j]); psc = sc[j];
                mn[j] = csym(c, dec, smn + (size_t)(sc[j] * 64 + pmn) * 64, 64, mn[j]); pmn = mn[j];
            }
        } else {
            d  = code_f16(c, dec, dhi, dlo, &pd, dec ? 0 : rd16(B));
            dm = code_f16(c, dec, mhi, mlo, &pm, dec ? 0 : rd16(B + 2));
            for (int j = 0; j < 8; j++) {
                sc[j] = ctree(c, dec, msc + psc * 64, 6, sc[j]); psc = sc[j];
                mn[j] = ctree(c, dec, mmn + (sc[j] * 64 + pmn) * 64, 6, mn[j]); pmn = mn[j];
            }
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
                if (!dec && c->cnt) c->cnt[(ctx << bits) + q[idx]]++;
                int n = cn ? q[idx] : ss ? ssym(c, dec, c->stt + (ctx << bits), 1 << bits, q[idx])
                                         : csym(c, dec, sv + (ctx << bits), 1 << bits, q[idx]);
                q[idx] = (uint8_t)n; n0 += n == 0; nm += n == maxv;
            }
        }
        if (dec) {
            wr16(B, d); wr16(B + 2, dm); k_pack_scales(B + 4, sc, mn); k_pack_vals(B, bits, q);
        }
    }
    free(dhi); free(dlo); free(mhi); free(mlo); free(msc); free(mmn); sym_free(sv);
    if (sh) { f16s_free(&fd); f16s_free(&fm); sym_free(ssc); sym_free(smn); }
}
/* every setting is a constant in its own copy, so the symbol loops have a fixed length */
#define KQ2(D, NB, SS, CN) (sh ? kq_code_impl(c, D, data, nblocks, NB, 1, SS, CN) : kq_code_impl(c, D, data, nblocks, NB, 0, SS, CN))
#define KQ1(D, NB) do { if (!(D) && c->cnt_only) KQ2(0, NB, 0, 1); else if (c->stt) KQ2(D, NB, 1, 0); else KQ2(D, NB, 0, 0); } while (0)
static void kq_code(Coder *c, uint8_t *data, uint64_t nblocks, int bits, int sh) {
    if (c->dec) { if (bits == 4) KQ1(1, 4); else KQ1(1, 5); }
    else        { if (bits == 4) KQ1(0, 4); else KQ1(0, 5); }
}
#undef KQ1
#undef KQ2

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

/* frexp(r2) = mant * 2^e, mant in [0.5, 1): bucket = (mant >= 1/sqrt 2 ? e : e - 1) - 2, clamped
   to 0..7. Read from the bits instead of calling frexp; identical for every finite r2 > 0. */
static inline int width_bucket(double r2) {
    uint64_t u; memcpy(&u, &r2, 8);
    int E = (int)(u >> 52) & 2047;
    if (E == 0) return 0;                      /* subnormal: e < -1021, always clamped to 0 */
    uint64_t f = (u & 0x000FFFFFFFFFFFFFull) | 0x3FE0000000000000ull;   /* mant, exactly */
    double mant; memcpy(&mant, &f, 8);
    int e = E - 1022;
    return clampi((mant >= 0.70710678118654752 ? e : e - 1) - 2, 0, 7);
}
#define Q6_CTX 9600                    /* 25 offset x 8 width x 16 pos x 3 extremes */
HOT void q6k_code_impl(Coder *c, const int dec, uint8_t *data, uint64_t rows, uint64_t ne0, const int lat, const int sh, const int ss, const int cn) {
    Bit *dhi = NULL, *dlo = NULL, *msc = NULL; F16S fd = {0}; int16_t *sch = NULL, *scl = NULL;
    if (cn) {}
    else if (sh) { fd = f16s_new(); sch = sym_new(256, 16); scl = sym_new(256 * 16, 16); }
    else { dhi = bits_new(65536); dlo = bits_new(65536); msc = bits_new(65536); }
    int16_t *sv = lat || ss || cn ? NULL : sym_new(Q6_CTX, 64);
    int16_t *skm = NULL, *sex = NULL, *skh = NULL, *skl = NULL, *se = NULL;
    if (lat) { skm = sym_new(33, 64); sex = sym_new(9, 64); skh = sym_new(66, 8); skl = sym_new(66 * 8, 16); se = sym_new(33, 8); }
    double *SW = lat ? NULL : calloc(ne0, sizeof *SW), *SW2 = lat ? NULL : calloc(ne0, sizeof *SW2);
    if (!lat && (!SW || !SW2)) die("out of memory");
    uint64_t bpr = ne0 / 256; int pd = 0, psc = 128, lkm = 0;
    for (uint64_t r = 0; r < rows; r++) {
        for (uint64_t bb = 0; bb < bpr; bb++) {
            uint8_t *B = data + (r * bpr + bb) * 210;
            uint8_t q[256]; int scs[16];
            uint16_t d = cn ? rd16(B + 208) : sh ? code_f16s(c, dec, &fd, &pd, dec ? 0 : rd16(B + 208))
                                                 : code_f16(c, dec, dhi, dlo, &pd, dec ? 0 : rd16(B + 208));
            if (cn) for (int k = 0; k < 16; k++) scs[k] = (int)(int8_t)B[192 + k] + 128;
            else for (int k = 0; k < 16; k++) {
                int sym = dec ? 0 : (int)(int8_t)B[192 + k] + 128;
                sym = sh ? csym_split(c, dec, sch, scl, 8, (size_t)psc, sym) : ctree(c, dec, msc + psc * 256, 8, sym);
                psc = sym; scs[k] = sym;
            }
            if (!dec) q6_unpack(B, q);
            if (lat) {
                for (int sb = 0; sb < 16; sb++) {
                    int y[16], kk[16], ee[16], km = 0, es = 0;
                    if (!dec) { for (int i = 0; i < 16; i++) y[i] = q[sb * 16 + i] - 32;
                                   lattice_fit(y, &km, &es, kk, ee); }
                    if (bb == 0 && sb == 0) lkm = 0;
                    km = csym(c, dec, skm + lkm * 64, 64, km); lkm = km;
                    if (km < 1 || km > 32) die("corrupt file");       /* encoder never writes 0 or > 32 */
                    es = csym(c, dec, sex + (km >> 2) * 64, 64, es);
                    double rr = (double)(es ? es : 32) / km;
                    int seen = 0;
                    for (int i = 0; i < 16; i++) {
                        int kc = km * 2 + (seen > 0);
                        int k = csym_split(c, dec, skh, skl, 7, (size_t)kc, dec ? 0 : kk[i] + 32) - 32;
                        int sym = dec ? 0 : (ee[i] >= -3 && ee[i] <= 3 ? ee[i] + 3 : 7);
                        sym = csym(c, dec, se + km * 8, 8, sym);
                        int v;
                        if (sym == 7) { v = 0; for (int b2 = 5; b2 >= 0; b2--) v |= cdirect(c, dec, (q[sb * 16 + i] >> b2) & 1) << b2; }
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
                        widb = isfinite(r2) && r2 > 0 ? width_bucket(r2) : 0;
                    }
                    int ctx = (offb * 8 + widb) * 48 + pos * 3 + (nx < 2 ? nx : 2);
                    if (!dec && c->cnt) c->cnt[(size_t)ctx * 64 + q[i]]++;
                    int v = cn ? q[i] : ss ? ssym(c, dec, c->stt + (size_t)ctx * 64, 64, q[i])
                                           : csym(c, dec, sv + (size_t)ctx * 64, 64, q[i]);
                    q[i] = (uint8_t)v; nx += v <= 1;
                    double W = (v - 32) * Ds;
                    if (isfinite(W)) { SW[col] += W; SW2[col] += W * W; }
                }
            }
            if (dec) {
                q6_pack(B, q);
                for (int k = 0; k < 16; k++) B[192 + k] = (uint8_t)(int8_t)(scs[k] - 128);
                wr16(B + 208, d);
            }
        }
    }
    free(dhi); free(dlo); free(msc); free(SW); free(SW2); sym_free(sv);
    if (sh) { f16s_free(&fd); sym_free(sch); sym_free(scl); }
    sym_free(skm); sym_free(sex); sym_free(skh); sym_free(skl); sym_free(se);
}
static void q6k_code(Coder *c, uint8_t *data, uint64_t rows, uint64_t ne0, int lat, int sh) {
    #define Q6S(D, L, SS, CN) (sh ? q6k_code_impl(c, D, data, rows, ne0, L, 1, SS, CN) : q6k_code_impl(c, D, data, rows, ne0, L, 0, SS, CN))
    #define Q6(D) do { if (lat) Q6S(D, 1, 0, 0); else if (!(D) && c->cnt_only) Q6S(0, 0, 0, 1); \
                       else if (c->stt) Q6S(D, 0, 1, 0); else Q6S(D, 0, 0, 0); } while (0)
    if (c->dec) Q6(1); else Q6(0);
    #undef Q6
    #undef Q6S
}

/* ---------------- Q8_0 ---------------- */
/* block: d(2) + 32 int8; w = d * q; the quantizer maps the largest |w| to +-127 */
#define Q8_CTX (8 * 3 * 16)
/* values as two 4-bit symbols from static tables: contexts 0..Q8_CTX-1 for the high nibble,
   then Q8_CTX + ctx * 16 + high for the low nibble. ss: static tables; cn: count only (pass 1) */
HOT void q80_code_impl(Coder *c, const int dec, uint8_t *data, uint64_t rows, uint64_t ne0, const int ss, const int cn) {
    Bit *dhi = cn ? NULL : bits_new(65536), *dlo = cn ? NULL : bits_new(65536), *val = ss || cn ? NULL : bits_new((size_t)Q8_CTX * 256);
    int64_t *P = calloc(ne0, sizeof *P); if (!P) die("out of memory");
    uint64_t bpr = ne0 / 32; int pd = 0;
    for (uint64_t r = 0; r < rows; r++) {
        for (uint64_t bb = 0; bb < bpr; bb++) {
            uint8_t *B = data + (r * bpr + bb) * 34;
            uint16_t d = cn ? rd16(B) : code_f16(c, dec, dhi, dlo, &pd, dec ? 0 : rd16(B));
            int ds = sgn16(d), nx = 0, sa = 0;
            for (int i = 0; i < 32; i++) {
                uint64_t col = bb * 32 + i;
                int cb = share_bucket(P[col], (int64_t)r); if (ds < 0) cb = 7 - cb;
                int avgb = i ? clampi((sa / i) >> 3, 0, 15) : 0;
                int ctx = (cb * 3 + (nx < 2 ? nx : 2)) * 16 + avgb;
                int u = dec ? 0 : (int)(int8_t)B[2 + i] + 128, v;
                if (cn) {
                    c->cnt[(size_t)ctx * 16 + (u >> 4)]++; c->cnt[((size_t)Q8_CTX + (size_t)ctx * 16 + (u >> 4)) * 16 + (u & 15)]++;
                    v = u - 128;
                } else if (ss) {
                    int hi = ssym(c, dec, c->stt + (size_t)ctx * 16, 16, u >> 4);
                    int lo = ssym(c, dec, c->stt + ((size_t)Q8_CTX + (size_t)ctx * 16 + hi) * 16, 16, u & 15);
                    v = ((hi << 4) | lo) - 128;
                } else v = ctree(c, dec, val + (size_t)ctx * 256, 8, u) - 128;
                nx += v >= 127 || v <= -127; sa += v < 0 ? -v : v;
                P[col] += (int64_t)v * ds > 0;
                if (dec) B[2 + i] = (uint8_t)(int8_t)v;
            }
            if (dec) wr16(B, d);
        }
    }
    free(dhi); free(dlo); free(val); free(P);
}
static void q80_code(Coder *c, uint8_t *data, uint64_t rows, uint64_t ne0) {
    if (c->dec) { if (c->stt) q80_code_impl(c, 1, data, rows, ne0, 1, 0); else q80_code_impl(c, 1, data, rows, ne0, 0, 0); }
    else if (c->cnt_only) q80_code_impl(c, 0, data, rows, ne0, 0, 1);
    else if (c->stt) q80_code_impl(c, 0, data, rows, ne0, 1, 0); else q80_code_impl(c, 0, data, rows, ne0, 0, 0);
}

/* ---------------- F32, F16, BF16 ---------------- */
/* Floats as 4-bit symbols from static tables. The high 16 bits (all of BF16 and F16; sign,
   exponent and top mantissa bits of F32) are four nibbles a b c d, high first: a given the
   previous value's a, b given a and the previous b, c given a b, d given c only: sign,
   exponent and the top of the mantissa get the full context, the low mantissa bits (close to
   random in trained weights) only a little, so their tables stay small. F32's low 16 bits
   are close to random in trained weights (or all zero when converted from BF16), so each of
   their nibbles only sees the one before it. Contexts: */
#define FL_HI 544                                    /* 16 + 256 + 256 + 16 */
#define FL_LO 64                                     /* F32 low half: each nibble given the one before */
HOT int fl_nib(Coder *c, const int dec, const int ss, const int cn, size_t ctx, int v) {
    if (cn) { c->cnt[ctx * 16 + v]++; return v; }
    (void)ss;
    return ssym(c, dec, c->stt + ctx * 16, 16, v);
}
HOT uint32_t fl_half(Coder *c, const int dec, const int ss, const int cn, uint32_t h, int *pa, int *pb) {
    int a = fl_nib(c, dec, ss, cn, (size_t)*pa, (int)(h >> 12) & 15);
    int b = fl_nib(c, dec, ss, cn, 16 + (size_t)((a << 4) | *pb), (int)(h >> 8) & 15);
    int cc = fl_nib(c, dec, ss, cn, 272 + (size_t)((a << 4) | b), (int)(h >> 4) & 15);
    int d = fl_nib(c, dec, ss, cn, 528 + (size_t)cc, (int)h & 15);
    *pa = a; *pb = b;
    return (uint32_t)((a << 12) | (b << 8) | (cc << 4) | d);
}
HOT uint32_t fl_low(Coder *c, const int dec, const int ss, const int cn, uint32_t l, int dprev) {
    const size_t B = FL_HI;
    int a = fl_nib(c, dec, ss, cn, B + (size_t)dprev, (int)(l >> 12) & 15);
    int b = fl_nib(c, dec, ss, cn, B + 16 + (size_t)a, (int)(l >> 8) & 15);
    int cc = fl_nib(c, dec, ss, cn, B + 32 + (size_t)b, (int)(l >> 4) & 15);
    int d = fl_nib(c, dec, ss, cn, B + 48 + (size_t)cc, (int)l & 15);
    return (uint32_t)((a << 12) | (b << 8) | (cc << 4) | d);
}
HOT void float_code_impl(Coder *c, const int dec, uint8_t *data, uint64_t n, uint32_t type, const int ss, const int cn) {
    int pa = 0, pb = 0;
    for (uint64_t i = 0; i < n; i++) {
        if (type == T_F32) {
            uint8_t *p = data + i * 4; uint32_t v = 0;
            if (!dec) memcpy(&v, p, 4);
            uint32_t h = fl_half(c, dec, ss, cn, v >> 16, &pa, &pb);
            uint32_t l = fl_low(c, dec, ss, cn, v & 0xFFFF, (int)(h & 15));
            if (dec) { v = (h << 16) | l; memcpy(p, &v, 4); }
        } else {
            uint8_t *p = data + i * 2;
            uint32_t h = fl_half(c, dec, ss, cn, dec ? 0 : rd16(p), &pa, &pb);
            if (dec) wr16(p, (uint16_t)h);
        }
    }
}
static void float_code(Coder *c, uint8_t *data, uint64_t n, uint32_t type) {
    if (c->dec) { if (!c->stt) die("corrupt file"); float_code_impl(c, 1, data, n, type, 1, 0); }
    else if (c->cnt_only) float_code_impl(c, 0, data, n, type, 0, 1);
    else { if (!c->stt) die("internal: float segment without tables"); float_code_impl(c, 0, data, n, type, 1, 0); }
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
HOT void bit_update_fast(Bit *b, int bit) {               /* KT-like, but adapts to changing data */
    uint32_t k = RATE[b->n]; int p = b->p;
    int up = (int)(((uint32_t)(65536 - p) * k) >> 16), dn = (int)(((uint32_t)p * k) >> 16);
    p = bit ? p - dn : p + up;
    p = p < 64 ? 64 : p;
    p = p > 65472 ? 65472 : p;
    b->p = (uint16_t)p; b->n = (uint16_t)(b->n + (b->n < 20));
}
#define NIN 8
static Bit *bits_new_aligned(size_t n) {
    void *p = NULL; if (posix_memalign(&p, 64, n * sizeof(Bit))) die("out of memory");
    bits_init(p, n); return p;
}
/* bucket of 16 counters (one cache line) for a context and the bits of the current nibble */
static inline uint32_t bucket(uint32_t h, int hb) { return ((h * 0x85EBCA6Bu) >> (32 - (hb - 4))) << 4; }
static int bytes_hb(uint64_t n) { int hb = 16; while (hb < 23 && ((uint64_t)1 << hb) < n * 2) hb++; return hb; }
HOT void bytes_code_impl(Coder *c, const int dec, uint8_t *data, uint64_t n) {
    int hb = bytes_hb(n);
    Bit *o0 = bits_new(256), *o1 = bits_new(65536);
    Bit *h[4]; for (int i = 0; i < 4; i++) h[i] = bits_new_aligned((size_t)1 << hb);
    int32_t *W = xmalloc(256 * 3 * NIN * sizeof *W);
    for (int i = 0; i < 256 * 3 * NIN; i++) W[i] = 19661;      /* 0.3 */
    uint64_t hist = 0;
    int mb = hb - 2; uint32_t *mt = calloc((size_t)1 << mb, sizeof *mt); if (!mt) die("out of memory");
    Bit *mm = bits_new(64 * 2); uint64_t ptr = 0; int mlen = 0;
    for (uint64_t i = 0; i < n; i++) {
        int byte = dec ? 0 : data[i], node = 1;
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
            int bit = code_bit(c, dec, p0, (byte >> k) & 1);
            int err = (bit << 12) - p1;
            for (int m = 0; m < NIN; m++) w[m] += (st[m] * err) >> 10;
            for (int m = 0; m < 6; m++) bit_update_fast(b[m], bit);
            bit_update_fast(bm, bit);
            node = (node << 1) | bit;
        }
        byte = node - 256;
        if (dec) data[i] = (uint8_t)byte;
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
MODEL(bytes_code, uint8_t *data X uint64_t n, data X n)

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
        case T_Q4K: kq_code(c, p, s->size / 144, 4, (s->model & H_SYM) != 0); break;
        case T_Q5K: kq_code(c, p, s->size / 176, 5, (s->model & H_SYM) != 0); break;
        case T_Q6K: q6k_code(c, p, s->rows, s->ne0, (s->model & 255) == M_LATTICE, (s->model & H_SYM) != 0); break;
        case T_Q8_0: q80_code(c, p, s->rows, s->ne0); break;
        case T_BYTES: bytes_code(c, p, s->size); break;
        default: float_code(c, p, s->rows * s->ne0, s->type);
    }
}

/* ---------------- static value tables ---------------- */
/* types with static value tables (K-quants and Q8_0): index, symbols per context, contexts */
#define NSTATIC 7
static int kq_index(uint32_t type) {
    return type == T_Q4K ? 0 : type == T_Q5K ? 1 : type == T_Q6K ? 2 : type == T_Q8_0 ? 3 :
           type == T_BF16 ? 4 : type == T_F16 ? 5 : type == T_F32 ? 6 : -1;
}
static const int KQ_SYM[NSTATIC] = {16, 32, 64, 16, 16, 16, 16},
                 KQ_CTX[NSTATIC] = {32 * 36, 64 * 36, Q6_CTX, Q8_CTX * 17, FL_HI, FL_HI, FL_HI + FL_LO};
/* segments coded with static tables: K-quants, not the lattice, not raw */
static int is_static(const Seg *s) { return kq_index(s->type) >= 0 && (s->model & 255) == M_NORMAL; }

/* frequencies summing to SM; every seen symbol gets at least 1, unseen ones 0 */
static void quantize(const uint64_t *n, int k, uint16_t *f) {
    uint64_t N = 0; for (int i = 0; i < k; i++) N += n[i];
    for (int i = 0; i < k; i++) f[i] = 0;
    if (!N) return;
    int sum = 0;
    for (int i = 0; i < k; i++) if (n[i]) {
        uint64_t q = (n[i] * (uint64_t)SM + N / 2) / N; f[i] = (uint16_t)(q ? q : 1); sum += f[i];
    }
    while (sum != SM) {                                  /* fix the total on the largest entries */
        int b = -1;
        for (int i = 0; i < k; i++) if (n[i] && (sum < SM || f[i] > 1) && (b < 0 || f[i] > f[b])) b = i;
        if (sum < SM) { f[b]++; sum++; } else { f[b]--; sum--; }
    }
}
static void cum_row(const uint16_t *f, int k, int16_t *U) { int a = 0; for (int i = 0; i < k; i++) { a += f[i]; U[i] = (int16_t)a; } }
static void uniform_row(int k, uint16_t *f) { for (int i = 0; i < k; i++) f[i] = (uint16_t)(SM / k); }

/* one seed per K-quant type: the counts of all its static segments together */
typedef struct {
    int k, nctx;
    uint64_t *agg;          /* pass 1: summed counts (compressor only) */
    uint16_t *f;            /* frequencies per context; a row of zeros = context unused */
    uint8_t *used;
} Seed;
static void seed_alloc(Seed *sd, int t) {
    sd->k = KQ_SYM[t]; sd->nctx = KQ_CTX[t];
    sd->f = calloc((size_t)sd->nctx * sd->k, 2); sd->used = calloc(sd->nctx, 1);
    if (!sd->f || !sd->used) die("out of memory");
}
static void seed_free(Seed *sd) { free(sd->agg); free(sd->f); free(sd->used); memset(sd, 0, sizeof *sd); }

/* small integers 0..SM: zero flag, then Elias gamma with adaptive bits per position i */
typedef struct { Bit *z, *e; } IntM;
static IntM intm_new(int k) { IntM m = { bits_new((size_t)k * 2), bits_new((size_t)k * 14) }; return m; }
static void intm_free(IntM *m) { free(m->z); free(m->e); }
static int code_mag(Coder *c, int dec, IntM *m, int i, int v) {    /* v in 1..SM */
    int e = 0;
    if (!dec) while ((v >> (e + 1)) != 0) e++;
    int ee = 0;
    while (ee < 13 && cbit(c, dec, &m->e[i * 14 + ee], ee < e)) ee++;
    e = ee;
    int r = dec ? 1 : v;
    if (dec) { for (int b = e - 1; b >= 0; b--) r = (r << 1) | cdirect(c, dec, 0); }
    else for (int b = e - 1; b >= 0; b--) cdirect(c, dec, (v >> b) & 1);
    return r;
}
static int code_uint(Coder *c, int dec, IntM *m, int i, int v) {   /* v in 0..SM */
    if (!cbit(c, dec, &m->z[i * 2], v != 0)) return 0;
    return code_mag(c, dec, m, i, v);
}
static int code_sint(Coder *c, int dec, IntM *m, int i, int v) {   /* v in -SM..SM */
    if (!cbit(c, dec, &m->z[i * 2], v != 0)) return 0;
    int neg = cbit(c, dec, &m->z[i * 2 + 1], v < 0);
    int a = code_mag(c, dec, m, i, v < 0 ? -v : v);
    return neg ? -a : a;
}
/* the seed tables of all types, coded once per file */
static void code_seeds(Coder *c, int dec, Seed sd[NSTATIC]) {
    Bit pres[NSTATIC]; bits_init(pres, NSTATIC);
    for (int t = 0; t < NSTATIC; t++) {
        int has = cbit(c, dec, &pres[t], sd[t].f != NULL);
        if (!has) continue;
        if (dec) seed_alloc(&sd[t], t);
        int k = sd[t].k; IntM m = intm_new(k); Bit um[2]; bits_init(um, 2); int pu = 0;
        for (int x = 0; x < sd[t].nctx; x++) {
            int u = cbit(c, dec, &um[pu], sd[t].used[x]); pu = u; sd[t].used[x] = (uint8_t)u;
            uint16_t *f = sd[t].f + (size_t)x * k;
            if (!u) continue;
            int sum = 0;
            for (int i = 0; i < k; i++) { int v = code_uint(c, dec, &m, i, f[i]); if (v > SM) die("corrupt file"); f[i] = (uint16_t)v; sum += v; }
            if (sum != SM) die("corrupt file");
        }
        intm_free(&m);
    }
}
/* one segment's tables, at the start of its stream: per context the seed has, "unused",
   "use the seed" or "own frequencies as differences from the seed". Fills the cumulative
   tables U for coding (unused contexts: the seed; contexts the seed lacks: uniform). */
static void code_tables(Coder *c, int dec, const Seed *sd, const uint16_t *own, const uint32_t *ownN, int16_t *U) {
    int k = sd->k; IntM m = intm_new(k); Bit um[2], om[2], so; bits_init(um, 2); bits_init(om, 2); bits_init(&so, 1);
    int pu = 0, po = 0; uint16_t f[64];
    /* first a flag: the whole segment uses the seed as it is, with no table section (small
       segments, or ones like the seed); the encoder picks whichever its estimate makes cheaper */
    int seed_only = 0;
    if (!dec) {
        double es_all = 0, sec = 0;
        for (int x = 0; x < sd->nctx; x++) {
            if (!sd->used[x] || !ownN[x]) { sec += sd->used[x] ? 0.2 : 0; continue; }
            const uint16_t *fs = sd->f + (size_t)x * k, *fo = own + (size_t)x * k; double es = 0, eo = 0;
            for (int i = 0; i < k; i++) {
                double nn = (double)fo[i] * ownN[x] / SM;
                if (fo[i]) { if (!fs[i]) es = 1e300; else es += nn * log2((double)SM / fs[i]); eo += nn * log2((double)SM / fo[i]); }
                int d = (int)fo[i] - (int)fs[i], a = d < 0 ? -d : d, e = 0; while ((a >> (e + 1)) != 0) e++;
                eo += d ? 2.0 * e + 3 : 0.3;
            }
            es_all += es; sec += 1.2 + (eo < es ? eo : es);
        }
        seed_only = es_all <= sec;
    }
    seed_only = cbit(c, dec, &so, seed_only);
    if (seed_only) {
        for (int x = 0; x < sd->nctx; x++) {
            if (sd->used[x]) memcpy(f, sd->f + (size_t)x * k, (size_t)k * 2); else uniform_row(k, f);
            cum_row(f, k, U + (size_t)x * k);
        }
        intm_free(&m);
        return;
    }
    for (int x = 0; x < sd->nctx; x++) {
        const uint16_t *fs = sd->f + (size_t)x * k;
        if (!sd->used[x]) { uniform_row(k, f); cum_row(f, k, U + (size_t)x * k); continue; }
        int u = cbit(c, dec, &um[pu], !dec && ownN[x] > 0); pu = u;
        int useown = 0;
        if (u) {
            if (!dec) {                                  /* estimated cost of each choice, in bits */
                const uint16_t *fo = own + (size_t)x * k; double es = 0, eo = 0;
                for (int i = 0; i < k; i++) {
                    double nn = (double)fo[i] * ownN[x] / SM;
                    if (fo[i]) { if (!fs[i]) es = 1e300; else es += nn * log2((double)SM / fs[i]); eo += nn * log2((double)SM / fo[i]); }
                    int d = (int)fo[i] - (int)fs[i], a = d < 0 ? -d : d, e = 0; while ((a >> (e + 1)) != 0) e++;
                    eo += d ? 2.0 * e + 3 : 0.3;
                }
                useown = eo < es;
            }
            useown = cbit(c, dec, &om[po], useown); po = useown;
        }
        if (useown) {
            int sum = 0;
            for (int i = 0; i < k; i++) {
                int v = (int)fs[i] + code_sint(c, dec, &m, i, dec ? 0 : (int)own[(size_t)x * k + i] - (int)fs[i]);
                if (v < 0 || v > SM) die("corrupt file");
                f[i] = (uint16_t)v; sum += v;
            }
            if (sum != SM) die("corrupt file");
        } else memcpy(f, fs, (size_t)k * 2);
        cum_row(f, k, U + (size_t)x * k);
    }
    intm_free(&m);
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
static Seg *build_segments(uint64_t n, uint64_t seg_target, uint64_t nt, uint32_t *types, uint64_t *offs,
                           uint64_t *ne0s, uint64_t *nes, int *nseg);

/* ---------------- safetensors ----------------
   u64 header length H, then H bytes of JSON: {"name": {"dtype": "BF16", "shape": [rows, cols],
   "data_offsets": [start, end]}, ..., "__metadata__": {...}}; tensor data follows, offsets relative
   to its start, row-major (the last dimension is contiguous). Only what is needed is parsed; a
   header that does not parse makes the file plain bytes. */
typedef struct { const uint8_t *p; size_t n, i; int bad; } Js;
static void js_ws(Js *j) { while (j->i < j->n && (j->p[j->i] == ' ' || j->p[j->i] == '\t' || j->p[j->i] == '\n' || j->p[j->i] == '\r')) j->i++; }
static int js_eat(Js *j, char ch) { js_ws(j); if (j->i < j->n && j->p[j->i] == (uint8_t)ch) { j->i++; return 1; } return 0; }
static void js_str(Js *j, char *out, size_t cap) {      /* a string; copied (truncated) into out if given */
    size_t k = 0;
    if (!js_eat(j, '"')) { j->bad = 1; return; }
    while (j->i < j->n && j->p[j->i] != '"') {
        if (j->p[j->i] == '\\') j->i++;
        if (j->i < j->n && out && k + 1 < cap) out[k++] = (char)j->p[j->i];
        j->i++;
    }
    if (j->i >= j->n) { j->bad = 1; return; }
    j->i++; if (out) out[k] = 0;
}
static uint64_t js_uint(Js *j) {
    js_ws(j); uint64_t v = 0; int any = 0;
    while (j->i < j->n && j->p[j->i] >= '0' && j->p[j->i] <= '9') {
        if (v > (UINT64_MAX - 9) / 10) { j->bad = 1; return 0; }
        v = v * 10 + (j->p[j->i++] - '0'); any = 1;
    }
    if (!any) j->bad = 1;
    return v;
}
static void js_skip(Js *j, int depth) {                  /* any value */
    js_ws(j); if (j->i >= j->n || depth > 64) { j->bad = 1; return; }
    uint8_t ch = j->p[j->i];
    if (ch == '"') { js_str(j, NULL, 0); return; }
    if (ch == '{' || ch == '[') {
        uint8_t close = ch == '{' ? '}' : ']'; j->i++;
        if (js_eat(j, (char)close)) return;
        for (;;) {
            if (ch == '{') { js_str(j, NULL, 0); if (!js_eat(j, ':')) { j->bad = 1; return; } }
            js_skip(j, depth + 1); if (j->bad) return;
            if (js_eat(j, ',')) continue;
            if (js_eat(j, (char)close)) return;
            j->bad = 1; return;
        }
    }
    while (j->i < j->n && !strchr(",}] \t\r\n", j->p[j->i])) j->i++;   /* number, true, false, null */
}
static Seg *safetensors_segments(const uint8_t *p, size_t n, uint64_t seg_target, int *nseg) {
    uint64_t H; memcpy(&H, p, 8);
    if (H < 2 || H > n - 8) return NULL;
    Js j = {p + 8, H, 0, 0};
    uint64_t data = 8 + H, cap = 64, nt = 0;
    uint32_t *types = xmalloc(cap * 4); uint64_t *offs = xmalloc(cap * 8), *ne0s = xmalloc(cap * 8), *nes = xmalloc(cap * 8);
    if (!js_eat(&j, '{')) j.bad = 1;
    if (!j.bad && !js_eat(&j, '}')) for (;;) {
        char name[16]; js_str(&j, name, sizeof name); if (j.bad || !js_eat(&j, ':')) { j.bad = 1; break; }
        if (!strcmp(name, "__metadata__")) js_skip(&j, 0);
        else {
            char dt[16] = ""; uint64_t ne = 1, e0 = 1, a = 0, b = 0; int have_off = 0;
            if (!js_eat(&j, '{')) { j.bad = 1; break; }
            if (!js_eat(&j, '}')) for (;;) {
                char key[16]; js_str(&j, key, sizeof key); if (j.bad || !js_eat(&j, ':')) { j.bad = 1; break; }
                if (!strcmp(key, "dtype")) js_str(&j, dt, sizeof dt);
                else if (!strcmp(key, "shape")) {
                    if (!js_eat(&j, '[')) { j.bad = 1; break; }
                    if (!js_eat(&j, ']')) for (;;) {
                        uint64_t d = js_uint(&j); e0 = d;               /* the last dimension is contiguous */
                        if (__builtin_mul_overflow(ne, d, &ne)) j.bad = 1;
                        if (j.bad) break;
                        if (js_eat(&j, ',')) continue;
                        if (!js_eat(&j, ']')) j.bad = 1;
                        break;
                    }
                } else if (!strcmp(key, "data_offsets")) {
                    if (!js_eat(&j, '[')) { j.bad = 1; break; }
                    a = js_uint(&j); if (!js_eat(&j, ',')) j.bad = 1; b = js_uint(&j); if (!js_eat(&j, ']')) j.bad = 1;
                    have_off = 1;
                } else js_skip(&j, 0);
                if (j.bad) break;
                if (js_eat(&j, ',')) continue;
                if (!js_eat(&j, '}')) j.bad = 1;
                break;
            }
            if (j.bad) break;
            uint32_t t = !strcmp(dt, "F32") ? T_F32 : !strcmp(dt, "F16") ? T_F16 : !strcmp(dt, "BF16") ? T_BF16 : 0xFFFFFFFFu;
            int ti = type_index(t);
            if (have_off && ti >= 0 && b >= a && b - a == ne * (uint64_t)TYPES[ti].bs && a <= n - data && b - a <= n - data - a) {
                if (nt == cap) { cap *= 2; types = realloc(types, cap * 4); offs = realloc(offs, cap * 8);
                                 ne0s = realloc(ne0s, cap * 8); nes = realloc(nes, cap * 8);
                                 if (!types || !offs || !ne0s || !nes) die("out of memory"); }
                types[nt] = t; offs[nt] = data + a; ne0s[nt] = e0 ? e0 : 1; nes[nt] = ne; nt++;
            }
        }
        if (j.bad) break;
        if (js_eat(&j, ',')) continue;
        if (!js_eat(&j, '}')) j.bad = 1;
        break;
    }
    if (j.bad) { free(types); free(offs); free(ne0s); free(nes); return NULL; }
    /* overlapping tensors would fail later: drop such a header to the byte model instead */
    for (uint64_t x = 0; x < nt; x++) for (uint64_t y = x + 1; y < nt && nt < 20000; y++) {
        uint64_t ex = offs[x] + nes[x] * TYPES[type_index(types[x])].bs, ey = offs[y] + nes[y] * TYPES[type_index(types[y])].bs;
        if (offs[x] < ey && offs[y] < ex) { free(types); free(offs); free(ne0s); free(nes); return NULL; }
    }
    return build_segments(n, seg_target, nt, types, offs, ne0s, nes, nseg);
}

static Seg *find_segments(const uint8_t *p, size_t n, uint64_t seg_target, int *nseg) {
    Rd r = {p, n, 0};
    if (n >= 10 && memcmp(p, "GGUF", 4) && p[8] == '{') {   /* safetensors */
        Seg *s = safetensors_segments(p, n, seg_target, nseg);
        if (s) return s;
    }
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
    uint64_t *offs = xmalloc(nt * 8 + 8), *ne0s = xmalloc(nt * 8 + 8), *nes = xmalloc(nt * 8 + 8);
    uint32_t *types = xmalloc(nt * 4 + 4);
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
    for (uint64_t i = 0; i < nt; i++) if (__builtin_add_overflow(data, offs[i], &offs[i])) die("tensor beyond end of file");
    return build_segments(n, seg_target, nt, types, offs, ne0s, nes, nseg);
}
/* tensors (type, absolute offset, first dimension, element count) -> segments; every byte not in
   a supported tensor goes to byte-model segments. Takes ownership of the four arrays. */
static Seg *build_segments(uint64_t n, uint64_t seg_target, uint64_t nt, uint32_t *types, uint64_t *offs,
                           uint64_t *ne0s, uint64_t *nes, int *nseg) {
    Seg *s = NULL; int cnt = 0, cap = 0;
    for (uint64_t i = 0; i < nt; i++) {
        int ti = type_index(types[i]); if (ti < 0) continue;
        uint64_t e0 = ne0s[i], ne = nes[i]; int vpb = TYPES[ti].vpb, bs = TYPES[ti].bs;
        if (!e0 || !ne || e0 % vpb || ne % e0) continue;
        uint64_t rows = ne / e0, rowb = e0 / vpb * bs, off = offs[i], tb;
        if (__builtin_mul_overflow(rows, rowb, &tb) || off > n || tb > n - off) die("tensor beyond end of file");
        uint64_t rps = seg_target / rowb; if (rps < 1) rps = 1;
        for (uint64_t r0 = 0; r0 < rows; r0 += rps) {
            uint64_t nr = rows - r0 < rps ? rows - r0 : rps;
            if (cnt == cap) { cap = cap * 2 + 64; s = realloc(s, cap * sizeof *s); if (!s) die("out of memory"); }
            s[cnt] = (Seg){off + r0 * rowb, nr * rowb, e0, nr, types[i], M_NORMAL}; cnt++;
        }
    }
    free(offs); free(ne0s); free(nes); free(types);
    if (cnt) qsort(s, cnt, sizeof *s, cmp_seg);
    for (int i = 1; i < cnt; i++) if (s[i].off < s[i - 1].off + s[i - 1].size) die("overlapping tensors");
    int nt_seg = cnt; uint64_t at = 0;
    for (int i = 0; i <= nt_seg; i++) {
        uint64_t end = i < nt_seg ? s[i].off : n;
        /* metadata in 2 MiB pieces, coded in parallel: costs about 3% of the metadata's coded size
           (each piece learns from scratch), but a model's metadata (10+ MiB of tokenizer data) is
           otherwise one segment on one thread, the longest of the run on small models and on
           machines with many cores */
        uint64_t bt = (uint64_t)2 << 20;
        for (uint64_t a = at; a < end; a += bt) {
            uint64_t len = end - a < bt ? end - a : bt;
            if (cnt == cap) { cap = cap * 2 + 64; s = realloc(s, cap * sizeof *s); if (!s) die("out of memory"); }
            s[cnt++] = (Seg){a, len, len, 1, T_BYTES, M_NORMAL};
        }
        if (i < nt_seg) at = s[i].off + s[i].size;
    }
    if (cnt) qsort(s, cnt, sizeof *s, cmp_seg);
    *nseg = cnt; return s;
}

/* ---------------- parallel driver ---------------- */
/* container: magic | u64 orig size | u32 nseg
              | nseg x (u64 off, u64 ne0, u64 rows, u32 type | model << 16, u64 clen, u64 coff)
              | raw bytes between segments | coded segments at their coff */
#define ENTRY 44
#define HDR 36                                   /* magic, size, segments, seed offset, seed length */
typedef struct {
    int mode;                          /* 0 compress, 1 decompress, 2 test */
    Seg *seg; int n; int next, failed, done; int *order;
    const uint8_t *in; int fd;
    uint64_t *clen, *coff;
    uint64_t end; pthread_mutex_t mu; double t0;
    double tsum[16], tmax; int tmax_seg;           /* thread-seconds per type, slowest segment */
    /* memory: a segment starts only if its estimated need fits in the budget, and (live) the
       device still has that much plus the reserve free; one segment can always run */
    uint64_t *need, used, peak, budget, reserve; int live, inflight, max_inflight; char *taken;
    int hsym;                                      /* -H: block headers as symbols */
    int phase;                                     /* compress: 1 = count (Q6_K: also try the lattice), 2 = code everything else */
    char *skip; int ntodo;                         /* segments this run leaves alone */
    Seed seed[NSTATIC], dseed[NSTATIC];                        /* seed tables: built (compress), decoded (test, decompress) */
    uint16_t **ownf; uint32_t **ownN; uint64_t *held;   /* per segment, from pass 1 to pass 2 */
    pthread_cond_t cv;
} Job;

static void pwrite_all(int fd, const void *p, uint64_t n, uint64_t at) {
    const uint8_t *q = p;
    while (n) {
        ssize_t w = pwrite(fd, q, n > (1u << 30) ? (1u << 30) : n, (off_t)at);
        if (w <= 0) die("write failed (disk full?)");
        q += w; n -= (uint64_t)w; at += (uint64_t)w;
    }
}
/* ---------------- memory ---------------- */
static uint64_t cgroup_value(const char *path) {         /* cgroup v2 file; 0 = none or "max" */
    FILE *f = fopen(path, "r"); if (!f) return 0;
    char b[64] = {0}; if (!fgets(b, sizeof b, f)) b[0] = 0; fclose(f);
    return (b[0] >= '0' && b[0] <= '9') ? strtoull(b, NULL, 10) : 0;
}
static uint64_t mem_total(void) {                        /* physical memory (or container limit); 0 = unknown */
    uint64_t t = 0;
#if defined(__APPLE__)
    size_t l = sizeof t; if (sysctlbyname("hw.memsize", &t, &l, NULL, 0)) t = 0;
#else
    long pg = sysconf(_SC_PAGESIZE), np = sysconf(_SC_PHYS_PAGES);
    if (pg > 0 && np > 0) t = (uint64_t)pg * (uint64_t)np;
#endif
    uint64_t cg = cgroup_value("/sys/fs/cgroup/memory.max");
    return cg && (!t || cg < t) ? cg : t;
}
/* memory that can be taken now without swapping or pushing out other apps; 0 = unknown */
static uint64_t mem_avail(void) {
    uint64_t a = 0;
#if defined(__linux__)                                    /* includes Android */
    FILE *f = fopen("/proc/meminfo", "r");
    if (f) {
        char line[128]; uint64_t avail = 0, fr = 0, cached = 0, v;
        while (fgets(line, sizeof line, f)) {
            if (sscanf(line, "MemAvailable: %" SCNu64, &v) == 1) avail = v;
            else if (sscanf(line, "MemFree: %" SCNu64, &v) == 1) fr = v;
            else if (sscanf(line, "Cached: %" SCNu64, &v) == 1) cached = v;
        }
        fclose(f);
        a = (avail ? avail : fr + cached) * 1024;          /* kernels before 3.14 lack MemAvailable */
    }
#elif defined(__APPLE__)
    static mach_port_t host = MACH_PORT_NULL;
    if (host == MACH_PORT_NULL) host = mach_host_self();
    vm_statistics64_data_t vs; mach_msg_type_number_t cnt = HOST_VM_INFO64_COUNT;
    if (host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&vs, &cnt) == KERN_SUCCESS)
        a = ((uint64_t)vs.free_count + vs.inactive_count + vs.purgeable_count) * (uint64_t)vm_page_size;
#elif defined(__FreeBSD__) || defined(__DragonFly__)
    u_int fr = 0, ina = 0; size_t l = sizeof fr; long pg = sysconf(_SC_PAGESIZE);
    if (!sysctlbyname("vm.stats.vm.v_free_count", &fr, &l, NULL, 0)) {
        l = sizeof ina; if (sysctlbyname("vm.stats.vm.v_inactive_count", &ina, &l, NULL, 0)) ina = 0;
        a = ((uint64_t)fr + ina) * (uint64_t)pg;
    }
#endif
#if defined(_SC_AVPHYS_PAGES)
    if (!a) { long pg = sysconf(_SC_PAGESIZE), np = sysconf(_SC_AVPHYS_PAGES);
              if (pg > 0 && np > 0) a = (uint64_t)pg * (uint64_t)np; }
#endif
    uint64_t lim = cgroup_value("/sys/fs/cgroup/memory.max");    /* containers: the limit counts */
    if (lim) { uint64_t cur = cgroup_value("/sys/fs/cgroup/memory.current");
               uint64_t room = lim > cur ? lim - cur : 0; if (!a || room < a) a = room; }
    return a;
}
/* memory the coder allocates for one segment (its context tables) */
static uint64_t model_bytes(const Seg *s) {
    const uint64_t B = sizeof(Bit);
    switch (s->type) {
        case T_Q4K: return B * (4 * 65536 + 64 * 64 + 4096 * 64) + 2 * ((uint64_t)32 * 36 << 4);
        case T_Q5K: return B * (4 * 65536 + 64 * 64 + 4096 * 64) + 2 * ((uint64_t)64 * 36 << 5);
        case T_Q6K: return B * (3 * 65536 + 33 * 64 + 9 * 64 + 66 * 128 + 33 * 8) + 2 * (uint64_t)Q6_CTX * 64 + 16 * s->ne0;
        case T_Q8_0: return B * (2 * 65536 + (uint64_t)Q8_CTX * 256) + 8 * s->ne0;   /* bit tree: upper bound */
        case T_BYTES: { int hb = bytes_hb(s->size);
                        return B * ((4ull << hb) + 256 + 65536 + 128) + (4ull << (hb - 2)) + 256 * 3 * NIN * 4; }
        default: return B * (4 + 3 * 64 * 256 + 512);
    }
}
/* peak memory of one segment: tables plus buffers. Compress: the coded output (Q6_K may code
   twice and keep the smaller). Test: output plus the decoded copy. Decompress: the decoded copy. */
static uint64_t seg_need(const Seg *s, int mode, int phase) {
    uint64_t z = s->size, slack = 1u << 20, mb = model_bytes(s);
    int t = kq_index(s->type);
    uint64_t tab = t >= 0 ? (uint64_t)KQ_CTX[t] * KQ_SYM[t] * 2 : 0;          /* one table set, int16 */
    if (mode == 1) return (s->model & 255) == M_RAW ? slack : slack + mb + z + (is_static(s) ? tab : 0);
    if (t < 0) return slack + mb + (mode == 2 ? 2 : 1) * z;
    if (phase == 1) return slack + mb + 2 * tab + (s->type == T_Q6K ? 2 * z : 0);   /* counts; Q6_K may try the lattice */
    return slack + mb + tab + (mode == 2 ? 2 * (z + tab) : z);
}
static uint64_t held_bytes(const Seg *s) { int t = kq_index(s->type); return (uint64_t)KQ_CTX[t] * (KQ_SYM[t] * 2 + 4); }
#define MiB(x) ((double)(x) / 1048576.0)
/* Fix the memory budget from what the device has free now, keeping a reserve for the system
   and other apps: 15% of RAM, at least 512 MiB, at most 2 GiB. limit (-m) replaces all this. */
static int plan_memory(Job *j, int threads, uint64_t limit) {
    j->need = xmalloc((j->n + 1) * sizeof *j->need); j->taken = calloc(j->n + 1, 1);
    if (!j->taken) die("out of memory");
    uint64_t mx = 0; for (int i = 0; i < j->n; i++) { j->need[i] = seg_need(&j->seg[i], j->mode, j->phase); if (j->need[i] > mx) mx = j->need[i]; }
    const uint64_t base = 32u << 20;                     /* foldz itself: index, tables, stacks */
    if (threads > j->n && j->n > 0) threads = j->n;
    if (threads < 1) threads = 1;
    if (limit) {
        j->budget = limit > base ? limit - base : 0; j->live = 0;
        printf("memory: limit %.0f MiB (-m)\n", MiB(limit));
    } else {
        uint64_t total = mem_total(), avail = mem_avail();
        j->reserve = total ? total / 100 * 15 : 0;
        if (j->reserve < (512u << 20)) j->reserve = 512u << 20;
        if (j->reserve > (2048ull << 20)) j->reserve = 2048ull << 20;
        if (avail) {
            j->budget = avail > j->reserve + base ? avail - j->reserve - base : 0; j->live = 1;
            printf("memory: %.0f MiB free, keeping %.0f MiB for the device, foldz may use %.0f MiB\n",
                   MiB(avail), MiB(j->reserve), MiB(j->budget));
        } else {
            j->budget = UINT64_MAX; j->live = 0;
            printf("memory: free memory unknown on this system, not limited (set a limit with -m MiB)\n");
        }
    }
    uint64_t fit = j->budget / (mx ? mx : 1); if (fit > (uint64_t)threads) fit = threads;
    printf("%d threads, largest segment needs about %.0f MiB", threads, MiB(mx));
    if (fit >= 1) printf(", at least %d at once\n", (int)fit);
    else printf("\n  note: that is more than the budget, so it will run alone%s\n",
                j->mode == 1 ? "" : " (a smaller -s lowers it)");
    return threads;
}
/* next segment to code: the most expensive one that fits in memory now; -1 when all are taken */
static int claim(Job *j) {
    pthread_mutex_lock(&j->mu);
    for (;;) {
        uint64_t avail = j->live && j->inflight ? mem_avail() : 0;
        int left = 0;
        for (int qi = 0; qi < j->n; qi++) {
            if (j->taken[qi]) continue;
            left = 1; int i = j->order[qi]; uint64_t need = j->need[i];
            int fits = j->inflight == 0 ||                               /* never stall: one always runs */
                       (j->used + need <= j->budget && (!avail || avail >= need + j->reserve));
            if (!fits) continue;
            j->taken[qi] = 1; j->used += need; j->inflight++;
            if (j->used > j->peak) j->peak = j->used;
            if (j->inflight > j->max_inflight) j->max_inflight = j->inflight;
            pthread_mutex_unlock(&j->mu);
            return i;
        }
        if (!left) { pthread_mutex_unlock(&j->mu); return -1; }
        pthread_cond_wait(&j->cv, &j->mu);              /* woken when a running segment finishes */
    }
}
/* a coded segment: store raw if not smaller, then write it (compress) or check it (test) */
static void finish_coded(Job *j, int i, Seg *s, Coder *c, const uint8_t *src) {
    const uint8_t *outp = c->buf;
    if (c->pos >= s->size) { c->pos = s->size; outp = src; s->model = M_RAW; }   /* never larger: store raw */
    j->clen[i] = c->pos;
    if (j->mode == 0) {
        pthread_mutex_lock(&j->mu); uint64_t at = j->end; j->end += c->pos; pthread_mutex_unlock(&j->mu);
        pwrite_all(j->fd, outp, c->pos, at); j->coff[i] = at;
    } else if (s->model != M_RAW) {
        uint8_t *buf = xmalloc(s->size); int16_t *U = NULL;
        Coder d; dec_init(&d, c->buf, c->pos);
        if (is_static(s)) { int t = kq_index(s->type); U = xmalloc((size_t)KQ_CTX[t] * KQ_SYM[t] * 2);
                            code_tables(&d, 1, &j->dseed[t], NULL, NULL, U); d.stt = U; }
        code_seg(&d, s, buf);
        if (memcmp(buf, src, s->size)) __atomic_store_n(&j->failed, 1, __ATOMIC_RELAXED);
        free(buf); free(U);
    }
    free(c->buf); c->buf = NULL;
}
static void *worker(void *arg) {
    Job *j = arg;
    for (;;) {
        int i = claim(j);
        if (i < 0) break;
        Seg *s = &j->seg[i]; Coder c; double ts = now();
        int t = kq_index(s->type);
        if (j->mode != 1) {
            const uint8_t *src = j->in + s->off;
            uint32_t xf = j->hsym && t >= 0 && t < 3 ? H_SYM : 0;
            if (j->phase == 2 && t >= 0) {                   /* static tables, then the values */
                size_t tn = (size_t)KQ_CTX[t] * KQ_SYM[t];
                int16_t *U = xmalloc(tn * 2);
                enc_init(&c, s->size); code_tables(&c, 0, &j->seed[t], j->ownf[i], j->ownN[i], U);
                c.stt = U; code_seg(&c, s, (uint8_t *)src); enc_flush(&c); c.stt = NULL;
                free(j->ownf[i]); free(j->ownN[i]); j->ownf[i] = NULL; j->ownN[i] = NULL;
                pthread_mutex_lock(&j->mu); j->used -= j->held[i]; pthread_mutex_unlock(&j->mu);
                free(U);
                finish_coded(j, i, s, &c, src);
            } else if (t >= 0) {                             /* pass 1: count (Q6_K may pick the lattice now) */
                size_t tn = (size_t)KQ_CTX[t] * KQ_SYM[t];
                uint32_t *cnt = calloc(tn, 4); if (!cnt) die("out of memory");
                int lattice = 0;
                s->model = M_NORMAL | xf;
                if (s->type == T_Q6K && lattice_sample(src, s->size / 210) < 20) {
                    enc_init(&c, s->size); c.cnt = cnt; code_seg(&c, s, (uint8_t *)src); enc_flush(&c); c.cnt = NULL;
                    Coder c2; s->model = M_LATTICE | xf;
                    enc_init(&c2, s->size); code_seg(&c2, s, (uint8_t *)src); enc_flush(&c2);
                    if (c2.pos < c.pos) { free(c.buf); c = c2; lattice = 1; }
                    else { free(c2.buf); free(c.buf); s->model = M_NORMAL | xf; }
                } else {
                    Coder cc; memset(&cc, 0, sizeof cc); cc.cnt = cnt; cc.cnt_only = 1;
                    code_seg(&cc, s, (uint8_t *)src);
                }
                if (lattice) { free(cnt); finish_coded(j, i, s, &c, src); }
                else {
                    int k = KQ_SYM[t], nx = KQ_CTX[t];
                    uint16_t *f = xmalloc(tn * 2); uint32_t *N = xmalloc((size_t)nx * 4); uint64_t row[64];
                    for (int x = 0; x < nx; x++) { uint64_t sum = 0;
                        for (int q = 0; q < k; q++) { row[q] = cnt[(size_t)x * k + q]; sum += row[q]; }
                        N[x] = (uint32_t)sum; quantize(row, k, f + (size_t)x * k); }
                    pthread_mutex_lock(&j->mu);
                    if (!j->seed[t].agg) { j->seed[t].agg = calloc(tn, 8); if (!j->seed[t].agg) die("out of memory"); }
                    for (size_t q = 0; q < tn; q++) j->seed[t].agg[q] += cnt[q];
                    j->ownf[i] = f; j->ownN[i] = N; j->held[i] = held_bytes(s);
                    j->used += j->held[i]; if (j->used > j->peak) j->peak = j->used;
                    pthread_mutex_unlock(&j->mu);
                    free(cnt);
                }
            } else {
                s->model = M_NORMAL;
                enc_init(&c, s->size); code_seg(&c, s, (uint8_t *)src); enc_flush(&c);
                finish_coded(j, i, s, &c, src);
            }
        } else if (s->model == M_RAW) {                      /* straight from the input, no buffer */
            if (j->clen[i] != s->size) die("corrupt file");
            pwrite_all(j->fd, j->in + j->coff[i], s->size, s->off);
        } else {
            uint8_t *buf = xmalloc(s->size); int16_t *U = NULL;
            dec_init(&c, j->in + j->coff[i], j->clen[i]);
            if (is_static(s)) {
                if (!j->dseed[t].f) die("corrupt file");
                U = xmalloc((size_t)KQ_CTX[t] * KQ_SYM[t] * 2);
                code_tables(&c, 1, &j->dseed[t], NULL, NULL, U); c.stt = U;
            }
            code_seg(&c, s, buf);
            pwrite_all(j->fd, buf, s->size, s->off); free(buf); free(U);
        }
        double dt = now() - ts;
        pthread_mutex_lock(&j->mu);
        j->used -= j->need[i]; j->inflight--; pthread_cond_broadcast(&j->cv);
        j->tsum[type_index(s->type)] += dt;
        if (dt > j->tmax) { j->tmax = dt; j->tmax_seg = i; }
        pthread_mutex_unlock(&j->mu);
        int d = __atomic_add_fetch(&j->done, 1, __ATOMIC_RELAXED);
        fprintf(stderr, "\r  %s%d/%d segments, %.0f s", j->phase == 1 ? "pass 1: " : j->phase == 2 ? "pass 2: " : "",
                d, j->ntodo, now() - j->t0);
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
    j->ntodo = 0;
    for (int qi = 0; qi < j->n; qi++) { j->taken[qi] = j->skip && j->skip[j->order[qi]]; j->ntodo += !j->taken[qi]; }
    j->next = 0; j->done = 0; j->t0 = now(); pthread_mutex_init(&j->mu, NULL); pthread_cond_init(&j->cv, NULL);
    pthread_t th[256]; if (threads > 256) threads = 256; if (threads < 1) threads = 1;
    if (threads > j->n && j->n > 0) threads = j->n;
    for (int i = 0; i < threads; i++) pthread_create(&th[i], NULL, worker, j);
    for (int i = 0; i < threads; i++) pthread_join(th[i], NULL);
    if (j->ntodo) fprintf(stderr, "\n");
    free(j->order);
}
static void time_report(Job *j, double wall, int threads) {
    double tot = 0; for (int k = 0; k < NTYPES; k++) tot += j->tsum[k];
    printf("  time: %.0f s on %d threads, %.0f thread-seconds of work (%.0f%% use)\n   ", wall, threads, tot,
           100.0 * tot / (wall * threads));
    for (int k = 0; k < NTYPES; k++) if (j->tsum[k] > 0) printf(" %s %.1f s", TYPES[k].name, j->tsum[k]);
    Seg *m = &j->seg[j->tmax_seg];
    printf("\n    slowest segment: %.1f s (%s, %.1f MiB)\n", j->tmax, TYPES[type_index(m->type)].name, m->size / 1048576.0);
    printf("  memory: peak about %.0f MiB for segments, up to %d at once\n", MiB(j->peak), j->max_inflight);
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
        nraw += s[i].model == M_RAW; if (s[i].type == T_Q6K) { n6++; nlat += (s[i].model & 255) == M_LATTICE; }
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
static void cmd_compress(const char *in, const char *out, int threads, uint64_t segb, int test, uint64_t limit, int hsym) {
    size_t n; const uint8_t *p = map_file(in, &n);
    int ns; Seg *s = find_segments(p, n, segb, &ns);
    printf("%s: %.1f MiB, %d segments to code\n", in, n / 1048576.0, ns);
    Job j; memset(&j, 0, sizeof j);
    j.mode = test ? 2 : 0; j.seg = s; j.n = ns; j.in = p; j.fd = -1; j.hsym = hsym; j.phase = 1;
    if (hsym) printf("-H: block headers as symbols (faster to decompress, slightly larger)\n");
    threads = plan_memory(&j, threads, limit);
    j.clen = calloc(ns + 1, 8); j.coff = calloc(ns + 1, 8);
    j.ownf = calloc(ns + 1, sizeof *j.ownf); j.ownN = calloc(ns + 1, sizeof *j.ownN); j.held = calloc(ns + 1, 8);
    j.skip = calloc(ns + 1, 1);
    if (!j.clen || !j.coff || !j.ownf || !j.ownN || !j.held || !j.skip) die("out of memory");
    uint64_t seg_total = 0; for (int i = 0; i < ns; i++) seg_total += s[i].size;
    uint64_t hdr = HDR + (uint64_t)ns * ENTRY;
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
    for (int i = 0; i < ns; i++) j.skip[i] = kq_index(s[i].type) < 0;   /* metadata and floats wait for pass 2 */
    double t0 = now(); run(&j, threads);                 /* pass 1: counts (and lattice segments, finished) */
    /* seeds: one table set per K-quant type, from the counts of all its static segments */
    size_t seed_cap = 4096;
    for (int t = 0; t < NSTATIC; t++) if (j.seed[t].agg) {
        uint64_t *agg = j.seed[t].agg; seed_alloc(&j.seed[t], t); j.seed[t].agg = agg;
        int k = j.seed[t].k;
        for (int x = 0; x < j.seed[t].nctx; x++) {
            uint64_t sum = 0; for (int q = 0; q < k; q++) sum += agg[(size_t)x * k + q];
            j.seed[t].used[x] = sum > 0;
            quantize(agg + (size_t)x * k, k, j.seed[t].f + (size_t)x * k);
        }
        free(agg); j.seed[t].agg = NULL;
        seed_cap += (size_t)j.seed[t].nctx * k * 4;
    }
    Coder sc; enc_init(&sc, seed_cap); code_seeds(&sc, 0, j.seed); enc_flush(&sc);
    if (sc.pos > sc.cap) die("internal: seed buffer too small");
    uint64_t seed_off = 0, seed_len = sc.pos;
    if (!test) { seed_off = j.end; pwrite_all(j.fd, sc.buf, sc.pos, seed_off); j.end += sc.pos; }
    else {                                               /* check the seeds' own round trip */
        Coder d; dec_init(&d, sc.buf, sc.pos); code_seeds(&d, 1, j.dseed);
        for (int t = 0; t < NSTATIC; t++) {
            if (!j.seed[t].f != !j.dseed[t].f) j.failed = 1;
            else if (j.seed[t].f && (memcmp(j.seed[t].f, j.dseed[t].f, (size_t)j.seed[t].nctx * j.seed[t].k * 2) ||
                                     memcmp(j.seed[t].used, j.dseed[t].used, j.seed[t].nctx))) j.failed = 1;
        }
    }
    free(sc.buf);
    /* pass 2: static K-quant segments */
    j.phase = 2;
    for (int i = 0; i < ns; i++) { j.skip[i] = kq_index(s[i].type) >= 0 && j.ownf[i] == NULL; j.need[i] = seg_need(&s[i], j.mode, 2); }
    run(&j, threads);
    double t1 = now();
    for (int t = 0; t < NSTATIC; t++) { seed_free(&j.seed[t]); seed_free(&j.dseed[t]); }
    uint64_t ctotal = seed_len; for (int i = 0; i < ns; i++) ctotal += j.clen[i];
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
    memcpy(h, MAGIC, 8); memcpy(h + 8, &v, 8); memcpy(h + 16, &c32, 4); memcpy(h + 20, &seed_off, 8); memcpy(h + 28, &seed_len, 8);
    for (int i = 0; i < ns; i++) {
        uint8_t *e = h + HDR + (uint64_t)i * ENTRY; uint32_t tm = s[i].type | (s[i].model << 16);
        memcpy(e, &s[i].off, 8); memcpy(e + 8, &s[i].ne0, 8); memcpy(e + 16, &s[i].rows, 8);
        memcpy(e + 24, &tm, 4); memcpy(e + 28, &j.clen[i], 8); memcpy(e + 36, &j.coff[i], 8);
    }
    pwrite_all(j.fd, h, hdr, 0); close(j.fd); free(h);
}

static void cmd_decompress(const char *in, const char *out, int threads, uint64_t limit) {
    size_t n; const uint8_t *p = map_file(in, &n);
    if (n < HDR) die("not a foldz file");
    if (memcmp(p, MAGIC, 8)) {
        if (!memcmp(p, "FOLDZ", 5) && p[7] == '\n')
            fprintf(stderr, "foldz: file format %.2s, but foldz " FOLDZ_VERSION " reads only format 14: "
                            "use the foldz version that made it\n", (const char *)p + 5), exit(1);
        die("not a foldz file");
    }
    uint64_t orig; uint32_t ns; memcpy(&orig, p + 8, 8); memcpy(&ns, p + 16, 4);
    uint64_t seed_off, seed_len; memcpy(&seed_off, p + 20, 8); memcpy(&seed_len, p + 28, 8);
    if (seed_off > n || seed_len > n - seed_off) die("corrupt header");
    uint64_t pos = HDR; if ((uint64_t)ns * ENTRY > n - pos) die("corrupt header");
    Seg *s = calloc(ns + 1, sizeof *s); uint64_t *clen = calloc(ns + 1, 8), *coff = calloc(ns + 1, 8);
    for (uint32_t i = 0; i < ns; i++) {
        const uint8_t *e = p + pos; uint32_t tm;
        memcpy(&s[i].off, e, 8); memcpy(&s[i].ne0, e + 8, 8); memcpy(&s[i].rows, e + 16, 8);
        memcpy(&tm, e + 24, 4); memcpy(&clen[i], e + 28, 8); memcpy(&coff[i], e + 36, 8);
        s[i].type = tm & 0xFFFF; s[i].model = tm >> 16;
        if (type_index(s[i].type) < 0 || !s[i].ne0) die("corrupt header");
        {   uint32_t md = s[i].model & 255, fl = s[i].model >> 8, kq = s[i].type == T_Q4K || s[i].type == T_Q5K || s[i].type == T_Q6K;
            if (md != M_NORMAL && md != M_RAW && !(md == M_LATTICE && s[i].type == T_Q6K)) die("corrupt header");
            if (fl & ~1u) die("corrupt header");
            if (fl && (!kq || md == M_RAW)) die("corrupt header"); }
        s[i].size = seg_bytes(&s[i]); pos += ENTRY;
        if (coff[i] > n || clen[i] > n - coff[i] || s[i].off > orig || s[i].size > orig - s[i].off) die("corrupt header");
    }
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
    { Coder d; dec_init(&d, p + seed_off, seed_len); code_seeds(&d, 1, j.dseed); }
    threads = plan_memory(&j, threads, limit);
    double t0 = now(); run(&j, threads); double t1 = now();
    if (ftruncate(fd, (off_t)orig)) die("cannot size output");
    close(fd);
    for (int t = 0; t < NSTATIC; t++) seed_free(&j.dseed[t]);
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
        printf("foldz " FOLDZ_VERSION " (file format 14)\n"); return 0;
    }
    if (argc < 3) {
        fprintf(stderr, "usage: foldz c in [out.foldz] [-t threads] [-s segment_MiB] [-m MiB] [-H] [-f]\n"
                        "       foldz d in.foldz [out] [-t threads] [-m MiB] [-f]\n"
                        "       foldz t in [-t threads] [-s segment_MiB] [-m MiB] [-H]   (round-trip test, no files)\n"
                        "       foldz v                                    (version)\n"
                        "  out defaults to in + .foldz (c) or in without .foldz (d); -f overwrites an existing file\n"
                        "  -H (c, t): block headers as symbols: faster to decompress, about 0.05%% larger\n"
                        "  defaults: all cores, 48 MiB segments; memory: what is free, minus a reserve for the\n"
                        "  device (15%% of RAM, 512 MiB to 2 GiB); -m sets foldz's memory limit instead\n");
        return 1;
    }
    int threads = (int)sysconf(_SC_NPROCESSORS_ONLN), force = 0; uint64_t segb = 48, limit = 0; int hsym = 0;
    const char *pos[2] = {NULL, NULL}; int np = 0;
    for (int i = 2; i < argc; i++) {                         /* options anywhere, in any order */
        if (!strcmp(argv[i], "-t") || !strcmp(argv[i], "-s") || !strcmp(argv[i], "-m")) {
            if (i + 1 >= argc) die("option needs a value");
            long long v = atoll(argv[++i]); char o = argv[i - 1][1];
            if (o == 't') threads = (int)v;
            else if (o == 's') segb = (uint64_t)(v < 1 ? 1 : v);
            else { if (v < 64) die("-m needs at least 64 (MiB)"); limit = (uint64_t)v << 20; }
        } else if (!strcmp(argv[i], "-f")) force = 1;
        else if (!strcmp(argv[i], "-H")) hsym = 1;
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
        if (cmd == 'c') cmd_compress(pos[0], out, threads, segb, 0, limit, hsym); else cmd_decompress(pos[0], out, threads, limit);
        free(out);
    } else if (cmd == 't' && argv[1][1] == 0) {
        if (pos[1]) die("t takes no output name");
        cmd_compress(pos[0], NULL, threads, segb, 1, limit, hsym);
    } else die("unknown command (use c, d, t or v)");
    return 0;
}
