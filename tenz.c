/*
 * tenz 0.23 - lossless compressor for GGUF models (file format 23)
 *
 * Successor of foldz (versions 0.9 to 0.15; 0.16 was never released). The numbers continue:
 * foldz files need the foldz version that made them.
 *
 * The version and the file format share one number: tenz 0.23 writes and reads format 23
 * (magic "TENZ23"). Until 1.0 the format may change between versions; a file is only
 * guaranteed to decompress with the version that made it. Check with: tenz v
 *
 * Works on GGUF and safetensors files (and, with the byte model alone, any other file): tensors of the supported types are coded with
 * context models learned from the data itself; every other byte (metadata,
 * tensor index, padding) is coded with a byte-level context-mixing model. GGUF tensors
 * of other types are stored as they are (the byte model gains little on them and is slow). Decompression gives back the identical file, byte for
 * byte. Nothing depends on the model's name or architecture.
 *
 * Supported types and models
 *   Q4_K, Q5_K  values: whole symbols, context: zero point, extremes seen
 *               (0 and max), position; with -q (quick) the zero point alone, decoded by a
 *               full slot -> symbol table (about 35% faster, about 0.6 to 0.8 points less saved). Header: d, dmin (fp16, as nibbles), scales, mins
 *               (high 2 bits, low 4 bits). Whole blocks in 4 rANS lanes.
 *   Q6_K        values: context = column offset and width (running mean and
 *               spread of the column, in units of the sub-block's step),
 *               position, low values seen (whole symbols). Header: d (fp16, as nibbles),
 *               scales (two nibbles each). Whole blocks in 4 rANS lanes, in chunks of
 *               2048 blocks. Or, if smaller, the AWQ lattice (its own static tables,
 *               same lanes; lattice steps given their column, errors, as symbols)
 *               (models converted from 4-bit AWQ/GPTQ checkpoints).
 *   Q8_0        values: context = column sign bias, extremes seen, average
 *               magnitude so far in the block (two 4-bit symbols each). Header: d (fp16,
 *               as nibbles, given the previous block's d). Whole blocks in 4 rANS lanes.
 *   Q4_0        values: context = column sign bias, q = 0 seen, average |q - 8| so far,
 *               position. Header: d (fp16, as nibbles). Whole blocks in 4 rANS lanes.
 *   F32, F16, BF16  sign, exponent (context: previous exponent), mantissa
 *               (top bits given the exponent).
 *   GGUF metadata  string arrays (tokens, merges) in pieces of about 1 MiB: the lengths with a
 *               small model, the strings with the byte model; arrays of 32-bit values
 *               (token types, scores) with the float model; the rest with the byte model.
 * Safety: every segment whose coded size would not be smaller is stored raw,
 * so the output is never larger than the input (apart from a small index).
 *
 * Coder: every tensor type is coded with interleaved rANS (4 lanes, decoded without divisions;
 * symbols found by a branch-free vector search) from static tables: Q4_K, Q5_K and Q6_K values
 * as whole symbols, Q4_0 values as 4-bit symbols, Q8_0 values as two 4-bit symbols (high, then
 * low given high), F32, F16 and BF16 values as 4-bit symbols (each 16 bits as four, each given
 * the ones above). The compressor first counts every segment, builds one seed table per type
 * for the whole file, and each segment then stores, per context, either "use the seed" or its
 * own table as differences from the seed. Decoding needs no learning, so it is faster than
 * compression. The tables themselves and the byte model use a range coder (LZMA style) with
 * adaptive bits (probabilities updated like the KT estimator). Big tensors are split into
 * segments that are coded independently, in parallel. Contexts use only exact or IEEE-754
 * arithmetic (no libm functions except exact ones), so files decode identically on any
 * platform.
 *
 * Build:  gcc -O2 -pthread -ffp-contract=off -o tenz tenz.c -lm
 *         (clang, e.g. Termux: same flags). Requires IEEE double arithmetic without excess
 *         precision or fused multiply-add, so that every platform computes the same contexts:
 *         64-bit x86, ARM and RISC-V are fine; 32-bit x86 needs -msse2 -mfpmath=sse.
 *         The build stops with an error where this can't be guaranteed (x87 maths, -ffast-math).
 *         -DTENZ_LUT decodes Q5_K, Q6_K by lookup instead of search, -DTENZ_SCALAR without vectors:
 *         both give identical output, for comparison.
 * Platforms: Linux, Android (Termux), macOS, BSD. Windows: use WSL (needs mmap and pwrite).
 * Usage:  tenz c model.gguf model.tenz [-t threads] [-s segment_MiB] [-m MiB] [-q]
 *         tenz d model.tenz model.gguf [-t threads] [-m MiB]
 *         tenz t model.gguf [-t threads] [-s segment_MiB] [-m MiB] [-q]   (round trip in memory)
 * Memory: segments start only while their estimated memory fits in what the device has free,
 *         keeping a reserve for the system and other apps; -m sets a fixed limit instead.
 */
#define _FILE_OFFSET_BITS 64
#define _GNU_SOURCE
#include <float.h>
#if defined(_WIN32) && !defined(__CYGWIN__)
#error "tenz needs POSIX mmap and pwrite: build it under WSL (or Cygwin)"
#endif
#if defined(__FAST_MATH__)
#error "do not build tenz with -ffast-math: contexts must be computed identically everywhere"
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

#define TENZ_VERSION "0.23"
#define TENZ_FORMAT "23"                               /* the number in MAGIC */
#define KQ_LANES 4                     /* rANS lanes of Q4_K and Q5_K (8 was measured: no faster) */
static const char MAGIC[8] = {'T','E','N','Z','2','3','\n', 0};    /* file format 23 */
/* T_M32: a GGUF metadata array of 32-bit values, coded as F32; T_STRS: whole records of a GGUF
   string array; T_BYTES: everything else outside the tensors */
enum { T_F32 = 0, T_F16 = 1, T_Q4_0 = 2, T_Q8_0 = 8, T_Q4K = 12, T_Q5K = 13, T_Q6K = 14, T_BF16 = 30,
       T_M32 = 0xFFFD, T_STRS = 0xFFFE, T_BYTES = 0xFFFF };
enum { M_NORMAL = 0, M_LATTICE = 1, M_QUICK = 2, M_RAW = 255 };   /* M_QUICK: Q4_K, Q5_K with -q */
static inline uint32_t fl_type(uint32_t t) { return t == T_M32 ? T_F32 : t; }   /* the float model's type */

typedef struct { uint32_t id; int vpb, bs; const char *name; } TypeInfo;
static const TypeInfo TYPES[] = {
    {T_Q4K, 256, 144, "Q4_K"}, {T_Q5K, 256, 176, "Q5_K"}, {T_Q6K, 256, 210, "Q6_K"},
    {T_Q4_0, 32, 18, "Q4_0"}, {T_Q8_0, 32, 34, "Q8_0"}, {T_BF16, 1, 2, "BF16"}, {T_F16, 1, 2, "F16"}, {T_F32, 1, 4, "F32"},
    {T_M32, 1, 4, "kv32"},               /* metadata arrays of 32-bit values (token types, scores) */
    {T_STRS, 1, 1, "strs"},              /* metadata string arrays (tokens, merges) */
    {T_BYTES, 1, 1, "other"},            /* metadata, tensor index, padding, unsupported tensor types */
};
#define NTYPES (int)(sizeof TYPES / sizeof TYPES[0])
static int type_index(uint32_t id) { for (int i = 0; i < NTYPES; i++) if (TYPES[i].id == id) return i; return -1; }

__attribute__((noreturn)) static void die(const char *m) { fprintf(stderr, "tenz: %s\n", m); exit(1); }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static double thread_cpu(void) {                   /* this thread's CPU time; 0 where there is no such clock */
#ifdef CLOCK_THREAD_CPUTIME_ID
    struct timespec t; if (!clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t)) return t.tv_sec + t.tv_nsec * 1e-9;
#endif
    return 0;
}
static void *xmalloc(size_t n) { void *p = malloc(n ? n : 1); if (!p) die("out of memory (try fewer threads: -t 2)"); return p; }

/* ---------------- adaptive bit model ---------------- */
typedef struct { uint16_t p; uint16_t n; } Bit;      /* p = P(bit = 0) * 65536 */
#define NLIM 1022
static uint32_t RATE[NLIM + 1];
static double LOG2T[66];
static void q40_init(void);
static void q6l_init(void);
static void q80_init(void);
static void init_tables(void) {
    q40_init(); q6l_init(); q80_init();
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
/* ---- static tables ----
   A context holds the cumulative counts of its n symbols: U[i] = count of symbols 0..i, so
   U[n-1] is the total, SM = 2^12: the coder needs no division by a total and no update, and
   zero-frequency symbols are never chosen. The searches work on 8 counts at a time with the
   GCC/Clang generic vector type, which the compiler turns into SSE2 or AVX on x86, NEON on ARM,
   or scalar code where there is none. Only integer compares and adds are used, so every CPU
   produces the same bytes. TENZ_SCALAR (or a compiler without the extension) selects plain loops
   that give identical output. */
#if !defined(TENZ_SCALAR) && (defined(__GNUC__) || defined(__clang__))
#define SYM_VEC 1
typedef int16_t V8 __attribute__((vector_size(16)));
static const V8 IOTA8 = {0, 1, 2, 3, 4, 5, 6, 7};
#endif
#define SM_BITS 12
#define SM (1 << SM_BITS)

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
static inline uint64_t rd64(const uint8_t *p) { uint64_t v = 0; for (int i = 7; i >= 0; i--) v = (v << 8) | p[i]; return v; }
static inline void wr64(uint8_t *p, uint64_t v) { for (int i = 0; i < 8; i++) { p[i] = (uint8_t)v; v >>= 8; } }
static inline int sgn16(uint16_t d) { return (d & 0x7FFF) == 0 ? 0 : ((d >> 15) ? -1 : 1); }
static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : v > hi ? hi : v; }

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
        if (16 * LOG2T[2 * m + 1] >= best) break;      /* the cost's floor rises with m: no larger m can win */
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
#define W_C 0.70710678118654752
/* Q6_K column statistics in float (exact IEEE operations, so still identical everywhere): twice
   as many per vector as double, half the memory */
typedef float Q6F;
#define Q6F_MAX FLT_MAX
#define Q6_CTX 9600                    /* 25 offset x 8 width x 16 pos x 3 extremes */

/* ---------------- F32, F16, BF16 ---------------- */
/* Floats as 4-bit symbols from static tables, coded with rANS in FL_LANES interleaved lanes:
   value i belongs to lane i % FL_LANES, and each lane carries its own state and its own history,
   so the lanes are independent chains the CPU can work on at once. Decoding a symbol is a mask,
   a table search, a multiply and an add: no division, unlike the range coder.
   The high 16 bits (all of BF16 and F16; sign, exponent and top mantissa bits of F32) are four
   nibbles a b c d, high first: a given the lane's previous a, b given a and the lane's previous
   b, c given a b, d given c. F32's low 16 bits: each nibble given the one before. */
#define FL_HI 544                                    /* 16 + 256 + 256 + 16 */
#define FL_LO 64
#define FL_LANES 4
#define RANS_L (1u << 16)                            /* rANS state in [2^16, 2^32), 16-bit output words */
/* refill a decoder state below RANS_L with the next 16-bit word, without a branch (whether a word is
   needed is data-dependent and mispredicts); past the end of the input it reads zeros, as before */
static const uint8_t RANS_ZERO2[2] = {0, 0};
#ifdef TENZ_BRANCH_REFILL
#define RANS_REFILL(ST, P, END) do { if ((ST) < RANS_L && (P) + 2 <= (END)) { uint16_t w16; memcpy(&w16, (P), 2); (P) += 2; (ST) = ((ST) << 16) | w16; } } while (0)
#else
#define RANS_REFILL(ST, P, END) do {                                                              \
    int ok_ = (P) + 2 <= (END); uint32_t need_ = ((ST) < RANS_L) & (uint32_t)ok_;                 \
    const uint8_t *src_ = ok_ ? (P) : RANS_ZERO2; uint16_t w16; memcpy(&w16, src_, 2);           \
    uint32_t sh_ = need_ << 4;                                                                    \
    (ST) = ((ST) << sh_) | (w16 & (0u - need_));                                                  \
    (P) += 2 * need_; } while (0)
#endif
static inline int fl_nsym(uint32_t type) { return type == T_F32 ? 8 : 4; }
static inline uint32_t fl_load(const uint8_t *data, uint64_t i, uint32_t type) {
    if (type == T_F32) { uint32_t v; memcpy(&v, data + i * 4, 4); return v; }
    return rd16(data + i * 2);
}
/* contexts and symbols of value v (its high half h; the low half for F32), lane history pa pb */
static inline void fl_syms(uint32_t type, uint32_t v, int pa, int pb, int *ctx, int *sym) {
    uint32_t h = type == T_F32 ? v >> 16 : v;
    int a = (int)(h >> 12) & 15, b = (int)(h >> 8) & 15, c = (int)(h >> 4) & 15, d = (int)h & 15;
    sym[0] = a; ctx[0] = pa;
    sym[1] = b; ctx[1] = 16 + ((a << 4) | pb);
    sym[2] = c; ctx[2] = 272 + ((a << 4) | b);
    sym[3] = d; ctx[3] = 528 + c;
    if (type == T_F32) {
        int la = (int)(v >> 12) & 15, lb = (int)(v >> 8) & 15, lc = (int)(v >> 4) & 15, ld = (int)v & 15;
        sym[4] = la; ctx[4] = FL_HI + d;
        sym[5] = lb; ctx[5] = FL_HI + 16 + la;
        sym[6] = lc; ctx[6] = FL_HI + 32 + lb;
        sym[7] = ld; ctx[7] = FL_HI + 48 + lc;
    }
}
/* pass 1: count */
static void float_count(uint32_t *cnt, const uint8_t *data, uint64_t n, uint32_t type) {
    int pa[FL_LANES] = {0}, pb[FL_LANES] = {0}, ctx[8], sym[8], ns = fl_nsym(type);
    for (uint64_t i = 0; i < n; i++) {
        int ln = (int)(i % FL_LANES);
        fl_syms(type, fl_load(data, i, type), pa[ln], pb[ln], ctx, sym);
        for (int k = 0; k < ns; k++) cnt[(size_t)ctx[k] * 16 + sym[k]]++;
        pa[ln] = sym[0]; pb[ln] = sym[1];
    }
}
/* encode: symbols in reverse order, words written backwards; out = lane states, then the words.
   Returns the length, or 0 if it does not fit in cap. U: cumulative tables (sum SM). */
static size_t float_encode(const uint8_t *data, uint64_t n, uint32_t type, const int16_t *U, uint8_t *out, size_t cap) {
    size_t head = 4 * FL_LANES;
    if (cap < head + 2) return 0;
    size_t nw = (cap - head) / 2; uint16_t *w = xmalloc(nw * 2 + 2), *wp = w + nw;
    uint32_t x[FL_LANES]; int ctx[8], sym[8], ns = fl_nsym(type);
    for (int k = 0; k < FL_LANES; k++) x[k] = RANS_L;
    for (uint64_t i = n; i-- > 0;) {
        int ln = (int)(i % FL_LANES), pa = 0, pb = 0;
        if (i >= FL_LANES) { int c2[8], s2[8]; fl_syms(type, fl_load(data, i - FL_LANES, type), 0, 0, c2, s2); pa = s2[0]; pb = s2[1]; }
        fl_syms(type, fl_load(data, i, type), pa, pb, ctx, sym);
        uint32_t st = x[ln];
        for (int k = ns - 1; k >= 0; k--) {
            const int16_t *T = U + (size_t)ctx[k] * 16;
            uint32_t start = sym[k] ? (uint32_t)T[sym[k] - 1] : 0, f = (uint32_t)T[sym[k]] - start;
            if (!f) die("internal: float symbol without frequency");
            if ((uint64_t)st >= ((uint64_t)(RANS_L >> SM_BITS) << 16) * f) {   /* 64-bit: f = SM gives 2^32 */ if (wp == w) { free(w); return 0; } *--wp = (uint16_t)st; st >>= 16; }
            st = ((st / f) << SM_BITS) + (st % f) + start;
        }
        x[ln] = st;
    }
    size_t words = (size_t)(w + nw - wp);
    for (int k = 0; k < FL_LANES; k++) memcpy(out + 4 * k, &x[k], 4);
    memcpy(out + head, wp, words * 2);
    free(w);
    return head + words * 2;
}
static inline int fl_search(const int16_t *T, uint32_t slot) {    /* number of cumulative counts <= slot */
#ifdef SYM_VEC
    V8 vv = IOTA8 * 0 + (int16_t)slot, acc = IOTA8 * 0, u0, u1;
    memcpy(&u0, T, 16); memcpy(&u1, T + 8, 16); acc -= (V8)(u0 <= vv); acc -= (V8)(u1 <= vv);
    uint64_t a2[2]; memcpy(a2, &acc, 16);
    return (int)(((a2[0] + a2[1]) * 0x0001000100010001ull) >> 48);
#else
    int sy = 0; for (int i = 0; i < 16; i++) sy += T[i] <= (int16_t)slot; return sy;
#endif
}
/* decode; corrupt input gives wrong values, never reads outside in[0..len) */
HOT void float_decode(const uint8_t *in, size_t len, uint64_t n, uint32_t type, const int16_t *U, uint8_t *data) {
    size_t head = 4 * FL_LANES;
    if (len < head) die("corrupt file");
    uint32_t x[FL_LANES]; int pa[FL_LANES] = {0}, pb[FL_LANES] = {0};
    for (int k = 0; k < FL_LANES; k++) memcpy(&x[k], in + 4 * k, 4);
    const uint8_t *p = in + head, *end = in + len;
    #define FL_DSYM(CTX, OUT) do {                                                              \
        const int16_t *T = U + (size_t)(CTX) * 16; uint32_t slot = st & (SM - 1);                \
        int sy = fl_search(T, slot);                                                            \
        uint32_t s0 = sy ? (uint32_t)T[sy - 1] : 0, f = (uint32_t)T[sy] - s0;                  \
        st = f * (st >> SM_BITS) + slot - s0;                                                     \
        RANS_REFILL(st, p, end);                                                                \
        (OUT) = sy; } while (0)
    /* one value of lane L: its state and history in locals, so the 4 lanes are independent chains */
    #define FL_DVAL(L, I) do {                                                                  \
        uint32_t st = st##L; int a, b, cc, d;                                                   \
        FL_DSYM(pa##L, a); FL_DSYM(16 + ((a << 4) | pb##L), b);                                 \
        FL_DSYM(272 + ((a << 4) | b), cc); FL_DSYM(528 + cc, d);                                \
        uint32_t h = (uint32_t)((a << 12) | (b << 8) | (cc << 4) | d); pa##L = a; pb##L = b;   \
        if (type == T_F32) {                                                                    \
            int la, lb, lc, ld;                                                                 \
            FL_DSYM(FL_HI + d, la); FL_DSYM(FL_HI + 16 + la, lb); FL_DSYM(FL_HI + 32 + lb, lc); FL_DSYM(FL_HI + 48 + lc, ld); \
            uint32_t v = (h << 16) | (uint32_t)((la << 12) | (lb << 8) | (lc << 4) | ld);        \
            memcpy(data + (I) * 4, &v, 4);                                                      \
        } else wr16(data + (I) * 2, (uint16_t)h);                                               \
        st##L = st; } while (0)
    uint32_t st0 = x[0], st1 = x[1], st2 = x[2], st3 = x[3];
    int pa0 = 0, pa1 = 0, pa2 = 0, pa3 = 0, pb0 = 0, pb1 = 0, pb2 = 0, pb3 = 0;
    (void)pa; (void)pb;
    uint64_t i = 0;
    for (; i + 4 <= n; i += 4) { FL_DVAL(0, i); FL_DVAL(1, i + 1); FL_DVAL(2, i + 2); FL_DVAL(3, i + 3); }
    if (i < n) { FL_DVAL(0, i); i++; }
    if (i < n) { FL_DVAL(1, i); i++; }
    if (i < n) { FL_DVAL(2, i); i++; }
    #undef FL_DVAL
    #undef FL_DSYM
}
static void float_code(Coder *c, uint8_t *data, uint64_t n, uint32_t type) {
    if (!c->dec && c->cnt_only) float_count(c->cnt, data, n, type);
    else die("internal: floats are coded with float_encode / float_decode");
}

/* ---------------- Q4_K and Q5_K in rANS lanes ---------------- */
/* Whole blocks are coded with rANS in 4 interleaved lanes from static tables, like the floats;
   only the components differ. One stream, 4 states, in groups of 4 blocks:
   - headers: block b belongs to lane b % 4, and each lane has its own history (the lane's
     previous block), so the 4 headers of a group are 4 independent chains of 40 symbols:
       d, dmin   fp16 as four nibbles a b c d, high first: a given the lane's previous a, b given
                 a and the lane's previous b, c given a b, d given c (as the floats)
       scale     6 bits as high 2 + low 4: high given the previous scale, low given both
       min       6 bits as high 2 + low 4: high given its scale and the previous min, low given
                 the high part, the previous min and the scale (coarser)
     ("previous" scale and min: the sub-block before; for the first, the lane's previous block)
   - values: contexts are the zero point of the sub-block, 0 and max seen so far in it, and the
     position. They restart at every 32-value sub-block, so lane L takes sub-blocks L and 4 + L
     of a block: value i of 4 sub-blocks at once.
   Table rows: the value contexts first, then KH_CTX header contexts (16 symbols used). */
enum { KH_D = 0, KH_M = FL_HI, KH_SH = 2 * FL_HI, KH_SL = KH_SH + 64, KH_MH = KH_SL + 256, KH_ML = KH_MH + 1024,
       KH_CTX = KH_ML + 512, KH_N = 40 };
static inline int kq_vctx(int bits) { return (bits == 4 ? 32 : 64) * 36; }
static inline int kq_search(const int16_t *T, int k, uint32_t slot) {   /* number of cumulative counts <= slot */
#ifdef SYM_VEC
    V8 vv = IOTA8 * 0 + (int16_t)slot, acc = IOTA8 * 0;
    for (int o = 0; o < k; o += 8) { V8 u; memcpy(&u, T + o, 16); acc -= (V8)(u <= vv); }
    uint64_t a2[2]; memcpy(a2, &acc, 16);
    return (int)(((a2[0] + a2[1]) * 0x0001000100010001ull) >> 48);
#else
    int sy = 0; for (int i = 0; i < k; i++) sy += T[i] <= (int16_t)slot; return sy;
#endif
}
/* rANS decoding by lookup (decoder only, files unchanged): each row of k symbols gets k buckets
   of SM / k slots holding the symbol of the bucket's first slot; the decoder steps forward past
   the boundaries before its slot (usually none). Exact for any row summing to SM, which every
   table does (checked when read). Row x at L + x * k, as in U. -DTENZ_NO_LUT to compare. */
#ifndef Q6_PF
#define Q6_PF 2                     /* Q6_K decoder: prefetch distance in positions, 0 = off */
#endif
#ifndef Q6_LUTB
#define Q6_LUTB 64                  /* Q6_K: buckets per row (-DQ6_LUTB=128 or 256: fewer steps, more cache) */
#endif
static uint8_t *lut_build(const int16_t *U, size_t nrows, int k, int nb) {   /* nb buckets per row */
    int sh = SM_BITS - __builtin_ctz((unsigned)nb);
    uint8_t *L = xmalloc(nrows * (size_t)nb);
    for (size_t x = 0; x < nrows; x++) {
        const int16_t *T = U + x * k; uint8_t *Lx = L + x * nb; int s = 0;
        for (int b = 0; b < nb; b++) { while (T[s] <= (b << sh)) s++; Lx[b] = (uint8_t)s; }
    }
    return L;
}
static inline int lut_find(const int16_t *T, const uint8_t *Lrow, int sh, uint32_t slot) {
    int s = Lrow[slot >> sh];
    while ((uint32_t)T[s] <= slot) s++;          /* stops by T[k-1] = SM > slot */
    return s;
}
/* Measured (0.22): the branch-free vector search over the whole row decodes Q5_K and Q6_K about 40%
   faster than the lookup, whose step loop mispredicts and whose tables add to the cache misses
   (the decoder cannot run ahead of its own symbols, so every miss is waited for). So the search is
   the default; -DTENZ_LUT selects the lookup. Both give identical output. */
#if !defined(TENZ_LUT) && !defined(TENZ_NO_LUT)
#define TENZ_NO_LUT
#endif
#ifdef TENZ_NO_LUT
#define LUT_FIND(T_, LP_, SH, K, SLOT) kq_search(T_, K, SLOT)
#else
#define LUT_FIND(T_, LP_, SH, K, SLOT) lut_find(T_, LP_, SH, SLOT)
#endif
static void kq_zb(const uint8_t *B, int bits, int *zb) {             /* the sub-blocks' zero points */
    int sc[8], mn[8], zmax = bits == 4 ? 31 : 63; k_unpack_scales(B + 4, sc, mn);
    float df = h2f(rd16(B)), mf = h2f(rd16(B + 2));
    for (int j = 0; j < 8; j++) {
        float step = df * (float)sc[j];
        float z = step > 0 ? mf * (float)mn[j] / step : 0.0f;
        zb[j] = (z == z && z < 1e6f) ? clampi((int)rintf(z * 2.0f), 0, zmax) : 0;
    }
}
/* contexts of a block's 256 values, in block order (q: the unpacked values) */
static void kq_val_ctx(const uint8_t *B, int bits, const uint8_t *q, uint16_t *ctx, int quick) {
    int zb[8], maxv = (1 << bits) - 1; kq_zb(B, bits, zb);
    for (int s = 0; s < 8; s++) {
        int n0 = 0, nm = 0;
        for (int i = 0; i < 32; i++) {
            int idx = s * 32 + i;
            ctx[idx] = (uint16_t)(quick ? zb[s] * 36 : (zb[s] * 9 + (n0 < 2 ? n0 : 2) * 3 + (nm < 2 ? nm : 2)) * 4 + (i >> 3));
            n0 += q[idx] == 0; nm += q[idx] == maxv;
        }
    }
}
static inline void kq_f16_syms(uint16_t h, uint16_t ph, int base, uint16_t *ctx, uint8_t *sym) {
    int a = h >> 12, b = (h >> 8) & 15, c = (h >> 4) & 15, d = h & 15, pa = ph >> 12, pb = (ph >> 8) & 15;
    sym[0] = (uint8_t)a; ctx[0] = (uint16_t)(base + pa);
    sym[1] = (uint8_t)b; ctx[1] = (uint16_t)(base + 16 + ((a << 4) | pb));
    sym[2] = (uint8_t)c; ctx[2] = (uint16_t)(base + 272 + ((a << 4) | b));
    sym[3] = (uint8_t)d; ctx[3] = (uint16_t)(base + 528 + c);
}
/* the KH_N header symbols of block B and their contexts (0 .. KH_CTX-1), in coding order;
   P: the lane's previous block (4 blocks back), or NULL */
static void kq_hdr_syms(const uint8_t *B, const uint8_t *P, uint16_t *ctx, uint8_t *sym) {
    int sc[8], mn[8], psc = 0, pmn = 0;
    k_unpack_scales(B + 4, sc, mn);
    if (P) { int s2[8], m2[8]; k_unpack_scales(P + 4, s2, m2); psc = s2[7]; pmn = m2[7]; }
    kq_f16_syms(rd16(B), P ? rd16(P) : 0, KH_D, ctx, sym);
    kq_f16_syms(rd16(B + 2), P ? rd16(P + 2) : 0, KH_M, ctx + 4, sym + 4);
    for (int j = 0; j < 8; j++) {
        int o = 8 + 4 * j, sh = sc[j] >> 4, mh = mn[j] >> 4;
        sym[o]     = (uint8_t)sh;            ctx[o]     = (uint16_t)(KH_SH + psc);
        sym[o + 1] = (uint8_t)(sc[j] & 15);  ctx[o + 1] = (uint16_t)(KH_SL + psc * 4 + sh);
        sym[o + 2] = (uint8_t)mh;            ctx[o + 2] = (uint16_t)(KH_MH + sc[j] * 16 + (pmn >> 2));
        sym[o + 3] = (uint8_t)(mn[j] & 15);  ctx[o + 3] = (uint16_t)(KH_ML + (mh * 16 + (pmn >> 2)) * 8 + (sc[j] >> 3));
        psc = sc[j]; pmn = mn[j];
    }
}
#define KQ_H (8 / KQ_LANES)            /* sub-blocks per lane in a block */
/* the decoder's zero / max counts: state min(n0, 2) * 3 + min(nm, 2), after a value that is
   neither (0), zero (1) or max (2) */
static const uint8_t KQ_STEP[9][3] = {{0,3,1},{1,4,2},{2,5,2},{3,6,4},{4,7,5},{5,8,5},{6,6,7},{7,7,8},{8,8,8}};
/* pass 1: count */
static void kq_count(uint32_t *cnt, const uint8_t *data, uint64_t nblocks, int bits, int quick) {
    int bs = bits == 4 ? 144 : 176, hb = kq_vctx(bits);
    for (uint64_t b = 0; b < nblocks; b++) {
        const uint8_t *B = data + b * bs; uint8_t q[256], hs[KH_N]; uint16_t ctx[256], hc[KH_N];
        kq_hdr_syms(B, b >= KQ_LANES ? B - KQ_LANES * bs : NULL, hc, hs);
        for (int n = 0; n < KH_N; n++) cnt[((size_t)(hb + hc[n]) << bits) + hs[n]]++;
        k_unpack_vals(B, bits, q); kq_val_ctx(B, bits, q, ctx, quick);
        for (int i = 0; i < 256; i++) cnt[((size_t)ctx[i] << bits) + q[i]]++;
    }
}
/* one rANS encoder step: symbol sy of table row T into state *x; 0 if the word buffer is full */
static inline int rans_put(uint32_t *x, const int16_t *T, int sy, uint16_t **wp, const uint16_t *w) {
    uint32_t start = sy ? (uint32_t)T[sy - 1] : 0, f = (uint32_t)T[sy] - start, st = *x;
    if (!f) die("internal: K-quant symbol without frequency");
    if ((uint64_t)st >= ((uint64_t)(RANS_L >> SM_BITS) << 16) * f) { if (*wp == w) return 0; *--*wp = (uint16_t)st; st >>= 16; }
    *x = ((st / f) << SM_BITS) + (st % f) + start;
    return 1;
}
/* encode: everything in the reverse of the decoder's order, words written backwards;
   out = KQ_LANES lane states, then the words. Returns the length, or 0 if it does not fit in cap.
   Groups of KQ_LANES blocks: block b's header in lane b % KQ_LANES (history: the lane's previous
   block); values: lane L takes sub-blocks L, L + KQ_LANES, ... of each block, value i of all
   lanes at once. */
static size_t kq_lanes_encode(const uint8_t *data, uint64_t nblocks, int bits, int quick, const int16_t *U, uint8_t *out, size_t cap) {
    int bs = bits == 4 ? 144 : 176, k = 1 << bits, hb = kq_vctx(bits); size_t head = 4 * KQ_LANES;
    if (cap < head + 2) return 0;
    size_t nw = (cap - head) / 2; uint16_t *w = xmalloc(nw * 2 + 2), *wp = w + nw;
    uint32_t x[KQ_LANES]; for (int L = 0; L < KQ_LANES; L++) x[L] = RANS_L;
    for (uint64_t g = (nblocks + KQ_LANES - 1) / KQ_LANES; g-- > 0;) {
        uint64_t b0 = g * KQ_LANES; int nb = nblocks - b0 < KQ_LANES ? (int)(nblocks - b0) : KQ_LANES;
        for (int bi = nb - 1; bi >= 0; bi--) {             /* values, last block of the group first */
            const uint8_t *B = data + (b0 + bi) * bs; uint8_t q[256]; uint16_t ctx[256];
            k_unpack_vals(B, bits, q); kq_val_ctx(B, bits, q, ctx, quick);
            for (int h = KQ_H - 1; h >= 0; h--) for (int i = 31; i >= 0; i--) for (int L = KQ_LANES - 1; L >= 0; L--) {
                int idx = (KQ_LANES * h + L) * 32 + i;
                if (!rans_put(&x[L], U + (size_t)ctx[idx] * k, q[idx], &wp, w)) { free(w); return 0; }
            }
        }
        uint16_t hc[KQ_LANES][KH_N]; uint8_t hs[KQ_LANES][KH_N];   /* headers: symbol n of each block, lane = block */
        for (int L = 0; L < nb; L++) { const uint8_t *B = data + (b0 + L) * bs; kq_hdr_syms(B, b0 ? B - KQ_LANES * bs : NULL, hc[L], hs[L]); }
        for (int n = KH_N - 1; n >= 0; n--) for (int L = nb - 1; L >= 0; L--)
            if (!rans_put(&x[L], U + (size_t)(hb + hc[L][n]) * k, hs[L][n], &wp, w)) { free(w); return 0; }
    }
    size_t words = (size_t)(w + nw - wp);
    for (int L = 0; L < KQ_LANES; L++) memcpy(out + 4 * L, &x[L], 4);
    memcpy(out + head, wp, words * 2); free(w);
    return head + words * 2;
}
/* decode; corrupt input gives wrong values, never reads outside in[0..len) or the tables.
   Lt: Q5_K's value rows by lookup (with -DTENZ_LUT; NULL for Q4_K, whose 16-symbol rows are searched) */
HOT void kq_lanes_decode_impl(const uint8_t *in, size_t len, uint8_t *data, uint64_t nblocks, const int bits, const int quick, const int16_t *U, const uint8_t *Lt) {
    const int bs = bits == 4 ? 144 : 176, k = 1 << bits, maxv = k - 1, zmax = bits == 4 ? 31 : 63;
    const int16_t *UH = U + (size_t)kq_vctx(bits) * k;        /* header rows */
    const size_t head = 4 * KQ_LANES;
    if (len < head) die("corrupt file");
    uint32_t x[KQ_LANES]; for (int L = 0; L < KQ_LANES; L++) memcpy(&x[L], in + 4 * L, 4);
    const uint8_t *p = in + head, *end = in + len;
    #define KQ_GET(L, ROW, KK, OUT) do {                                                         \
        const int16_t *T_ = (ROW); uint32_t st_ = x[L], slot_ = st_ & (SM - 1); int sy_ = kq_search(T_, KK, slot_); \
        uint32_t s0_ = sy_ ? (uint32_t)T_[sy_ - 1] : 0, f_ = (uint32_t)T_[sy_] - s0_;            \
        st_ = f_ * (st_ >> SM_BITS) + slot_ - s0_;                                               \
        RANS_REFILL(st_, p, end);                                                                \
        x[L] = st_; (OUT) = sy_; } while (0)
    #define KQ_GETV(L, ROW, OUT) do {                                                            \
        size_t o_ = (size_t)(ROW) * k; const int16_t *T_ = U + o_;                               \
        uint32_t st_ = x[L], slot_ = st_ & (SM - 1);                                             \
        int sy_ = bits == 5 ? LUT_FIND(T_, Lt + o_, 7, 32, slot_) : kq_search(T_, k, slot_);     \
        uint32_t s0_ = sy_ ? (uint32_t)T_[sy_ - 1] : 0, f_ = (uint32_t)T_[sy_] - s0_;            \
        st_ = f_ * (st_ >> SM_BITS) + slot_ - s0_;                                               \
        RANS_REFILL(st_, p, end);                                                                \
        x[L] = st_; (OUT) = sy_; } while (0)
    #define KQ_HDR(L, CTX, MASK, OUT) do { int s_; KQ_GET(L, UH + (size_t)(CTX) * k, 16, s_); (OUT) = s_ & (MASK); } while (0)
    #define KQ_F16(BASE, PA, PB, OUT) do { int a[KQ_LANES] = {0}, b_[KQ_LANES] = {0}, c_[KQ_LANES] = {0}, d_[KQ_LANES] = {0}; \
        for (int L = 0; L < nb; L++) KQ_HDR(L, (BASE) + PA[L], 15, a[L]);                        \
        for (int L = 0; L < nb; L++) KQ_HDR(L, (BASE) + 16 + ((a[L] << 4) | PB[L]), 15, b_[L]);  \
        for (int L = 0; L < nb; L++) KQ_HDR(L, (BASE) + 272 + ((a[L] << 4) | b_[L]), 15, c_[L]); \
        for (int L = 0; L < nb; L++) KQ_HDR(L, (BASE) + 528 + c_[L], 15, d_[L]);                 \
        for (int L = 0; L < nb; L++) { OUT[L] = (uint16_t)((a[L] << 12) | (b_[L] << 8) | (c_[L] << 4) | d_[L]); PA[L] = a[L]; PB[L] = b_[L]; } \
        } while (0)
    /* quick: one value row per zero point, so each gets a full slot -> symbol table */
    uint8_t *FS = NULL;
    if (quick) {
        FS = xmalloc((size_t)(zmax + 1) * SM);
        for (int z = 0; z <= zmax; z++) { const int16_t *T = U + (size_t)z * 36 * k; uint8_t *F = FS + (size_t)z * SM; int sy = 0;
            for (int sl = 0; sl < SM; sl++) { while (sy < k - 1 && T[sy] <= sl) sy++; F[sl] = (uint8_t)sy; } }
    }
    #define KQ_GETF(L, Z, OUT) do {                                                              \
        const int16_t *T_ = U + (size_t)(Z) * 36 * k; uint32_t st_ = x[L], slot_ = st_ & (SM - 1); \
        int sy_ = FS[(size_t)(Z) * SM + slot_];                                                  \
        uint32_t s0_ = sy_ ? (uint32_t)T_[sy_ - 1] : 0, f_ = (uint32_t)T_[sy_] - s0_;            \
        st_ = f_ * (st_ >> SM_BITS) + slot_ - s0_;                                               \
        RANS_REFILL(st_, p, end);                                                                \
        x[L] = st_; (OUT) = sy_; } while (0)
    int da[KQ_LANES] = {0}, db[KQ_LANES] = {0}, ma[KQ_LANES] = {0}, mb[KQ_LANES] = {0}, psc[KQ_LANES] = {0}, pmn[KQ_LANES] = {0};
    for (uint64_t b0 = 0; b0 < nblocks; b0 += KQ_LANES) {
        const int nb = nblocks - b0 < KQ_LANES ? (int)(nblocks - b0) : KQ_LANES;
        uint16_t d[KQ_LANES], dm[KQ_LANES]; int sc[KQ_LANES][8], mn[KQ_LANES][8];
        KQ_F16(KH_D, da, db, d); KQ_F16(KH_M, ma, mb, dm);
        for (int j = 0; j < 8; j++) {
            int sh[KQ_LANES] = {0}, sl[KQ_LANES] = {0}, mh[KQ_LANES] = {0}, ml[KQ_LANES] = {0};
            for (int L = 0; L < nb; L++) KQ_HDR(L, KH_SH + psc[L], 3, sh[L]);
            for (int L = 0; L < nb; L++) { KQ_HDR(L, KH_SL + psc[L] * 4 + sh[L], 15, sl[L]); sc[L][j] = (sh[L] << 4) | sl[L]; }
            for (int L = 0; L < nb; L++) KQ_HDR(L, KH_MH + sc[L][j] * 16 + (pmn[L] >> 2), 3, mh[L]);
            for (int L = 0; L < nb; L++) { KQ_HDR(L, KH_ML + (mh[L] * 16 + (pmn[L] >> 2)) * 8 + (sc[L][j] >> 3), 15, ml[L]);
                                           mn[L][j] = (mh[L] << 4) | ml[L]; psc[L] = sc[L][j]; pmn[L] = mn[L][j]; }
        }
        for (int bi = 0; bi < nb; bi++) {
            uint8_t *B = data + (b0 + bi) * bs; uint8_t q[256]; int zb[8];
            wr16(B, d[bi]); wr16(B + 2, dm[bi]); k_pack_scales(B + 4, sc[bi], mn[bi]);
            float df = h2f(d[bi]), mf = h2f(dm[bi]);
            for (int j = 0; j < 8; j++) {                     /* as kq_zb */
                float step = df * (float)sc[bi][j];
                float z = step > 0 ? mf * (float)mn[bi][j] / step : 0.0f;
                zb[j] = (z == z && z < 1e6f) ? clampi((int)rintf(z * 2.0f), 0, zmax) : 0;
            }
            if (quick) for (int h = 0; h < KQ_H; h++) {
                int z[KQ_LANES];
                for (int L = 0; L < KQ_LANES; L++) z[L] = zb[KQ_LANES * h + L];
                for (int i = 0; i < 32; i++) for (int L = 0; L < KQ_LANES; L++) {
                    int sy; KQ_GETF(L, z[L], sy); q[(KQ_LANES * h + L) * 32 + i] = (uint8_t)sy;
                }
            }
            else for (int h = 0; h < KQ_H; h++) {
                /* per lane: c = min(n0, 2) * 3 + min(nm, 2), stepped by a table; row = zb * 36 + c * 4 + i / 8 */
                int c[KQ_LANES] = {0}, zr[KQ_LANES];
                for (int L = 0; L < KQ_LANES; L++) zr[L] = zb[KQ_LANES * h + L] * 36;
                for (int i = 0; i < 32; i++) for (int L = 0; L < KQ_LANES; L++) {
                    int idx = (KQ_LANES * h + L) * 32 + i, sy;
                    KQ_GETV(L, zr[L] + c[L] * 4 + (i >> 3), sy);
                    q[idx] = (uint8_t)sy; c[L] = KQ_STEP[c[L]][sy == 0 ? 1 : sy == maxv ? 2 : 0];
                }
            }
            k_pack_vals(B, bits, q);
        }
    }
    free(FS);
    #undef KQ_GETF
    #undef KQ_F16
    #undef KQ_HDR
    #undef KQ_GETV
    #undef KQ_GET
}
static void kq_lanes_decode(const uint8_t *in, size_t len, uint8_t *data, uint64_t nblocks, int bits, int quick, const int16_t *U, const uint8_t *Lt) {
    if (bits == 4) { if (quick) kq_lanes_decode_impl(in, len, data, nblocks, 4, 1, U, Lt); else kq_lanes_decode_impl(in, len, data, nblocks, 4, 0, U, Lt); }
    else { if (quick) kq_lanes_decode_impl(in, len, data, nblocks, 5, 1, U, Lt); else kq_lanes_decode_impl(in, len, data, nblocks, 5, 0, U, Lt); }
}
static void kq_code(Coder *c, uint8_t *data, uint64_t nblocks, int bits, int quick) {
    if (!c->dec && c->cnt_only) kq_count(c->cnt, data, nblocks, bits, quick);
    else die("internal: Q4_K and Q5_K are coded with kq_lanes_encode / kq_lanes_decode");
}

/* ---------------- Q4_0 in rANS lanes ---------------- */
/* block: d(2) + 16 bytes, value i in the low nibble of byte i and value 16 + i in the high one;
   w = d * (q - 8), and the quantizer maps the block's largest |w| to q = 0.
   Blocks in groups of 4, block b in lane b % 4 (like the Q4_K headers), one rANS stream:
   - d: fp16 as four nibbles with the float contexts, the lane's previous d as history
   - values: the column's share of positive weights so far (flipped when d < 0), whether
     q = 0 was seen in the block, the average |q - 8| so far, the position. The column counts are
     those at the start of the group and are updated after it, so the 4 lanes stay independent.
   Table rows: Q40_VCTX value contexts, then FL_HI header contexts. */
#define Q40_VCTX (8 * 2 * 8 * 4)
/* The column's share of positive weights, (P + 1) / (r + 2), in 1/32 steps by one multiplication
   with a per-row reciprocal, then 8 buckets with edges near .3 .4 .45 .5 .55 .6 .7 (within 1/64),
   times 64: its part of the context. Flipped when d < 0. */
static uint16_t Q40_CB[33];                     /* bucket of each 1/32 step, by its midpoint; filled by q40_init */
static inline uint32_t q40_inv(uint64_t r) { return (uint32_t)((32ull << 26) / (r + 2)); }   /* share * 32 * 2^26 */
static inline int q40_cb(uint32_t pc, uint32_t inv, int ds) {
    int cb = Q40_CB[((uint64_t)(pc + 1) * inv) >> 26];
    return ds < 0 ? 448 - cb : cb;
}
/* the rest: seen * 32 + av * 4 + i / 8, av = min(7, 2 sa / i) from a table by (i, sa) */
static uint8_t Q40_AV[32][256];
static void q40_init(void) {
    static const int E[7] = {30, 40, 45, 50, 55, 60, 70};
    for (int k = 0; k <= 32; k++) { int b = 0; for (int e = 0; e < 7; e++) b += 100 * (2 * k + 1) >= 64 * E[e]; Q40_CB[k] = (uint16_t)(b * 64); }
    for (int i = 0; i < 32; i++) for (int sa = 0; sa < 256; sa++) {
        int av = i ? 2 * sa / i : 0; if (av > 7) av = 7;
        Q40_AV[i][sa] = (uint8_t)(av * 4 + (i >> 3));
    }
}
static inline int q40_ctx1(const uint32_t *P, uint64_t col, uint32_t inv, int ds, int seen, int sa, int i) {
    return q40_cb(P[col], inv, ds) + seen * 32 + Q40_AV[i][sa];
}
static inline void q40_unpack(const uint8_t *B, uint8_t *q) { for (int j = 0; j < 16; j++) { q[j] = B[2 + j] & 15; q[16 + j] = B[2 + j] >> 4; } }
/* block b's value contexts, from the column counts P */
static void q40_vctx(const uint32_t *P, uint64_t b, uint64_t bpr, const uint8_t *q, int ds, uint16_t *ctx) {
    uint64_t c0 = b % bpr * 32; int seen = 0, sa = 0; uint32_t inv = q40_inv(b / bpr);
    for (int i = 0; i < 32; i++) {
        ctx[i] = (uint16_t)q40_ctx1(P, c0 + i, inv, ds, seen, sa, i);
        seen |= q[i] == 0; sa += q[i] < 8 ? 8 - q[i] : q[i] - 8;
    }
}
/* add (dir 1) or remove (dir -1) block b's values in the column counts */
static inline void q40_cols(uint32_t *P, uint64_t b, uint64_t bpr, const uint8_t *q, int ds, int dir) {
    uint32_t *Pc = P + b % bpr * 32;
    for (int i = 0; i < 32; i++) Pc[i] += (uint32_t)(dir * (((int)q[i] - 8) * ds > 0));
}
static void q40_count(uint32_t *cnt, const uint8_t *data, uint64_t rows, uint64_t ne0) {
    uint64_t bpr = ne0 / 32, nb = rows * bpr; uint32_t *P = calloc(ne0, 4); if (!P) die("out of memory");
    for (uint64_t b0 = 0; b0 < nb; b0 += 4) {
        int n = nb - b0 < 4 ? (int)(nb - b0) : 4; uint8_t q[4][32];
        for (int L = 0; L < n; L++) {
            uint64_t b = b0 + L; const uint8_t *B = data + b * 18; uint16_t hc[4], ctx[32]; uint8_t hs[4];
            kq_f16_syms(rd16(B), b >= 4 ? rd16(B - 72) : 0, Q40_VCTX, hc, hs);
            for (int k = 0; k < 4; k++) cnt[((size_t)hc[k] << 4) + hs[k]]++;
            q40_unpack(B, q[L]); q40_vctx(P, b, bpr, q[L], sgn16(rd16(B)), ctx);
            for (int i = 0; i < 32; i++) cnt[((size_t)ctx[i] << 4) + q[L][i]]++;
        }
        for (int L = 0; L < n; L++) q40_cols(P, b0 + L, bpr, q[L], sgn16(rd16(data + (b0 + L) * 18)), 1);
    }
    free(P);
}
/* encode: everything in the reverse of the decoder's order, words written backwards. The column
   counts are run to the end first, then taken back group by group. 0 if it does not fit in cap. */
static size_t q40_encode(const uint8_t *data, uint64_t rows, uint64_t ne0, const int16_t *U, uint8_t *out, size_t cap) {
    size_t head = 16;
    if (cap < head + 2) return 0;
    uint64_t bpr = ne0 / 32, nb = rows * bpr; uint32_t *P = calloc(ne0, 4); if (!P) die("out of memory");
    for (uint64_t b = 0; b < nb; b++) { uint8_t q[32]; q40_unpack(data + b * 18, q); q40_cols(P, b, bpr, q, sgn16(rd16(data + b * 18)), 1); }
    size_t nw = (cap - head) / 2; uint16_t *w = xmalloc(nw * 2 + 2), *wp = w + nw;
    uint32_t x[4] = {RANS_L, RANS_L, RANS_L, RANS_L};
    for (uint64_t g = (nb + 3) / 4; g-- > 0;) {
        uint64_t b0 = g * 4; int n = nb - b0 < 4 ? (int)(nb - b0) : 4;
        uint8_t q[4][32], hs[4][4]; uint16_t ctx[4][32], hc[4][4]; int ds[4];
        for (int L = 0; L < n; L++) { const uint8_t *B = data + (b0 + L) * 18; q40_unpack(B, q[L]); ds[L] = sgn16(rd16(B)); q40_cols(P, b0 + L, bpr, q[L], ds[L], -1); }
        for (int L = 0; L < n; L++) {
            const uint8_t *B = data + (b0 + L) * 18;
            q40_vctx(P, b0 + L, bpr, q[L], ds[L], ctx[L]);
            kq_f16_syms(rd16(B), b0 ? rd16(B - 72) : 0, Q40_VCTX, hc[L], hs[L]);
        }
        for (int i = 31; i >= 0; i--) for (int L = n - 1; L >= 0; L--)
            if (!rans_put(&x[L], U + (size_t)ctx[L][i] * 16, q[L][i], &wp, w)) { free(w); free(P); return 0; }
        for (int k = 3; k >= 0; k--) for (int L = n - 1; L >= 0; L--)
            if (!rans_put(&x[L], U + (size_t)hc[L][k] * 16, hs[L][k], &wp, w)) { free(w); free(P); return 0; }
    }
    size_t words = (size_t)(w + nw - wp);
    for (int L = 0; L < 4; L++) memcpy(out + 4 * L, &x[L], 4);
    memcpy(out + head, wp, words * 2); free(w); free(P);
    return head + words * 2;
}
typedef struct { uint32_t x[4]; const uint8_t *p, *end; } RansDec;
#define RD_GET(D, L, ROW, OUT) do {                                                                  \
    const int16_t *T_ = (ROW); uint32_t st_ = (D)->x[L], slot_ = st_ & (SM - 1); int sy_ = kq_search(T_, 16, slot_); \
    uint32_t s0_ = sy_ ? (uint32_t)T_[sy_ - 1] : 0, f_ = (uint32_t)T_[sy_] - s0_;                    \
    st_ = f_ * (st_ >> SM_BITS) + slot_ - s0_;                                                       \
    RANS_REFILL(st_, (D)->p, (D)->end);                                                              \
    (D)->x[L] = st_; (OUT) = sy_; } while (0)
/* one group of n blocks (n = 4 except at the end; inlined for each) */
static inline __attribute__((always_inline)) void q40_dgroup(RansDec *D, uint8_t *data, uint64_t b0, const int n, uint64_t bpr,
                                                             uint32_t *P, const int16_t *U, int *pa, int *pb) {
    const int16_t *UH = U + (size_t)Q40_VCTX * 16;
    int a[4] = {0}, bn[4] = {0}, cn[4] = {0}, dn[4] = {0}, ds[4], seen[4] = {0}, sa[4] = {0};
    uint64_t c0[4]; uint32_t inv[4]; uint8_t q[4][32]; uint16_t d[4];
    for (int L = 0; L < n; L++) RD_GET(D, L, UH + (size_t)pa[L] * 16, a[L]);
    for (int L = 0; L < n; L++) RD_GET(D, L, UH + (size_t)(16 + ((a[L] << 4) | pb[L])) * 16, bn[L]);
    for (int L = 0; L < n; L++) RD_GET(D, L, UH + (size_t)(272 + ((a[L] << 4) | bn[L])) * 16, cn[L]);
    for (int L = 0; L < n; L++) RD_GET(D, L, UH + (size_t)(528 + cn[L]) * 16, dn[L]);
    for (int L = 0; L < n; L++) {
        d[L] = (uint16_t)((a[L] << 12) | (bn[L] << 8) | (cn[L] << 4) | dn[L]); pa[L] = a[L]; pb[L] = bn[L];
        ds[L] = sgn16(d[L]); inv[L] = q40_inv((b0 + L) / bpr); c0[L] = (b0 + L) % bpr * 32;
    }
    for (int i = 0; i < 32; i++) for (int L = 0; L < n; L++) {
        int sy; RD_GET(D, L, U + (size_t)q40_ctx1(P, c0[L] + i, inv[L], ds[L], seen[L], sa[L], i) * 16, sy);
        q[L][i] = (uint8_t)sy; seen[L] |= sy == 0; sa[L] += sy < 8 ? 8 - sy : sy - 8;
    }
    for (int L = 0; L < n; L++) {
        uint8_t *B = data + (b0 + L) * 18; wr16(B, d[L]);
        for (int j = 0; j < 16; j++) B[2 + j] = (uint8_t)(q[L][j] | (q[L][16 + j] << 4));
        q40_cols(P, b0 + L, bpr, q[L], ds[L], 1);
    }
}
/* decode; corrupt input gives wrong values, never reads outside in[0..len) */
HOT void q40_decode(const uint8_t *in, size_t len, uint8_t *data, uint64_t rows, uint64_t ne0, const int16_t *U) {
    if (len < 16) die("corrupt file");
    RansDec D; for (int L = 0; L < 4; L++) memcpy(&D.x[L], in + 4 * L, 4);
    D.p = in + 16; D.end = in + len;
    uint64_t bpr = ne0 / 32, nb = rows * bpr, b0 = 0; uint32_t *P = calloc(ne0, 4); if (!P) die("out of memory");
    int pa[4] = {0}, pb[4] = {0};
    for (; b0 + 4 <= nb; b0 += 4) q40_dgroup(&D, data, b0, 4, bpr, P, U, pa, pb);
    if (b0 < nb) q40_dgroup(&D, data, b0, (int)(nb - b0), bpr, P, U, pa, pb);
    free(P);
}
static void q40_code(Coder *c, uint8_t *data, uint64_t rows, uint64_t ne0) {
    if (!c->dec && c->cnt_only) q40_count(c->cnt, data, rows, ne0);
    else die("internal: Q4_0 is coded with q40_encode / q40_decode");
}

/* ---------------- Q8_0 in rANS lanes ---------------- */
/* block: d(2) + 32 int8; w = d * q, the quantizer maps the block's largest |w| to +-127.
   Like Q4_0: blocks in groups of 4, block b in lane b % 4, one rANS stream.
   - d: fp16 as four nibbles a b c e given the previous block's d (block b - 1; 0 at the segment's
     start): a given that d's high byte, b given a and that byte, c given a b, e given b c. Neighbouring
     blocks have close scales, so the previous block predicts d well (0.20's adaptive model had the
     high byte given the previous high byte). The group's 4 headers become a short serial chain
     (16 of its 272 symbols); the values stay 4 independent chains.
   - values: u = q + 128 as a high nibble given the context, then a low nibble given the context
     and the high one. Context: the column's share of positive weights so far (Q4_0's 8 buckets,
     flipped when d < 0), extremes |q| >= 127 seen in the block (0, 1, 2+), the average |q| so far
     in steps of 8 (16 buckets, from a table: no division). Column counts are those at the start of
     the group, so the 4 lanes stay independent.
   Table rows: Q8_CTX high-nibble rows, Q8_CTX * 16 low-nibble rows, then Q8H_CTX header rows. */
#define Q8_CTX (8 * 3 * 16)
#define Q8_VROWS (Q8_CTX * 17)
enum { Q8H_A = 0, Q8H_B = 256, Q8H_C = Q8H_B + 4096, Q8H_D = Q8H_C + 256, Q8H_CTX = Q8H_D + 256 };
static uint8_t Q8_AV[32][497];                  /* min(15, floor(sa / (8 i))) by (i, sa / 8): 0.20's (sa / i) >> 3 */
static void q80_init(void) {
    for (int i = 1; i < 32; i++) for (int s = 0; s < 497; s++) { int a = s / i; Q8_AV[i][s] = (uint8_t)(a < 15 ? a : 15); }
}
static inline int q80_ctx1(const uint32_t *P, uint64_t col, uint32_t inv, int ds, int nx, int sa, int i) {
    return ((q40_cb(P[col], inv, ds) >> 6) * 3 + (nx < 2 ? nx : 2)) * 16 + Q8_AV[i][sa >> 3];
}
/* d's 4 symbols and their table rows; pd: the previous block's d */
static inline void q80_hdr_syms(uint16_t d, uint16_t pd, uint16_t *ctx, uint8_t *sym) {
    int a = d >> 12, b = (d >> 8) & 15, c = (d >> 4) & 15, e = d & 15, ph = pd >> 8;
    sym[0] = (uint8_t)a; ctx[0] = (uint16_t)(Q8_VROWS + Q8H_A + ph);
    sym[1] = (uint8_t)b; ctx[1] = (uint16_t)(Q8_VROWS + Q8H_B + ((a << 8) | ph));
    sym[2] = (uint8_t)c; ctx[2] = (uint16_t)(Q8_VROWS + Q8H_C + ((a << 4) | b));
    sym[3] = (uint8_t)e; ctx[3] = (uint16_t)(Q8_VROWS + Q8H_D + ((b << 4) | c));
}
static void q80_vctx(const uint32_t *P, uint64_t b, uint64_t bpr, const uint8_t *B, int ds, uint16_t *ctx) {
    uint64_t c0 = b % bpr * 32; uint32_t inv = q40_inv(b / bpr); int nx = 0, sa = 0;
    for (int i = 0; i < 32; i++) {
        int v = (int)(int8_t)B[2 + i];
        ctx[i] = (uint16_t)q80_ctx1(P, c0 + i, inv, ds, nx, sa, i);
        nx += v >= 127 || v <= -127; sa += v < 0 ? -v : v;
    }
}
static inline void q80_cols(uint32_t *P, uint64_t b, uint64_t bpr, const uint8_t *B, int ds, int dir) {
    uint32_t *Pc = P + b % bpr * 32;
    for (int i = 0; i < 32; i++) Pc[i] += (uint32_t)(dir * ((int)(int8_t)B[2 + i] * ds > 0));
}
static void q80_count(uint32_t *cnt, const uint8_t *data, uint64_t rows, uint64_t ne0) {
    uint64_t bpr = ne0 / 32, nb = rows * bpr; uint32_t *P = calloc(ne0, 4); if (!P) die("out of memory");
    for (uint64_t b0 = 0; b0 < nb; b0 += 4) {
        int n = nb - b0 < 4 ? (int)(nb - b0) : 4;
        for (int L = 0; L < n; L++) {
            uint64_t b = b0 + L; const uint8_t *B = data + b * 34; uint16_t hc[4], ctx[32]; uint8_t hs[4];
            q80_hdr_syms(rd16(B), b ? rd16(B - 34) : 0, hc, hs);
            for (int k = 0; k < 4; k++) cnt[((size_t)hc[k] << 4) + hs[k]]++;
            q80_vctx(P, b, bpr, B, sgn16(rd16(B)), ctx);
            for (int i = 0; i < 32; i++) {
                int u = (int)(int8_t)B[2 + i] + 128;
                cnt[((size_t)ctx[i] << 4) + (u >> 4)]++;
                cnt[((size_t)(Q8_CTX + ctx[i] * 16 + (u >> 4)) << 4) + (u & 15)]++;
            }
        }
        for (int L = 0; L < n; L++) { const uint8_t *B = data + (b0 + L) * 34; q80_cols(P, b0 + L, bpr, B, sgn16(rd16(B)), 1); }
    }
    free(P);
}
/* encode: the reverse of the decoder's order (per group: each block's d, block by block, then the
   values), words written backwards; column counts run to the end first, then taken back group by
   group. 0 if it does not fit in cap. */
static size_t q80_encode(const uint8_t *data, uint64_t rows, uint64_t ne0, const int16_t *U, uint8_t *out, size_t cap) {
    size_t head = 16;
    if (cap < head + 2) return 0;
    uint64_t bpr = ne0 / 32, nb = rows * bpr; uint32_t *P = calloc(ne0, 4); if (!P) die("out of memory");
    for (uint64_t b = 0; b < nb; b++) { const uint8_t *B = data + b * 34; q80_cols(P, b, bpr, B, sgn16(rd16(B)), 1); }
    size_t nw = (cap - head) / 2; uint16_t *w = xmalloc(nw * 2 + 2), *wp = w + nw;
    uint32_t x[4] = {RANS_L, RANS_L, RANS_L, RANS_L};
    for (uint64_t g = (nb + 3) / 4; g-- > 0;) {
        uint64_t b0 = g * 4; int n = nb - b0 < 4 ? (int)(nb - b0) : 4;
        uint16_t ctx[4][32], hc[4][4]; uint8_t hs[4][4]; int ds[4];
        for (int L = 0; L < n; L++) { const uint8_t *B = data + (b0 + L) * 34; ds[L] = sgn16(rd16(B)); q80_cols(P, b0 + L, bpr, B, ds[L], -1); }
        for (int L = 0; L < n; L++) {
            const uint8_t *B = data + (b0 + L) * 34;
            q80_vctx(P, b0 + L, bpr, B, ds[L], ctx[L]);
            q80_hdr_syms(rd16(B), b0 + L ? rd16(B - 34) : 0, hc[L], hs[L]);
        }
        for (int i = 31; i >= 0; i--) {                     /* decoder: all lanes' high, then all lanes' low */
            for (int L = n - 1; L >= 0; L--) {
                int u = (int)(int8_t)data[(b0 + L) * 34 + 2 + i] + 128;
                if (!rans_put(&x[L], U + ((size_t)(Q8_CTX + ctx[L][i] * 16 + (u >> 4)) << 4), u & 15, &wp, w)) goto full;
            }
            for (int L = n - 1; L >= 0; L--) {
                int u = (int)(int8_t)data[(b0 + L) * 34 + 2 + i] + 128;
                if (!rans_put(&x[L], U + ((size_t)ctx[L][i] << 4), u >> 4, &wp, w)) goto full;
            }
        }
        for (int L = n - 1; L >= 0; L--) for (int k = 3; k >= 0; k--)   /* decoder: block by block, a b c e */
            if (!rans_put(&x[L], U + ((size_t)hc[L][k] << 4), hs[L][k], &wp, w)) goto full;
    }
    {
        size_t words = (size_t)(w + nw - wp);
        for (int L = 0; L < 4; L++) memcpy(out + 4 * L, &x[L], 4);
        memcpy(out + head, wp, words * 2); free(w); free(P);
        return head + words * 2;
    }
full:
    free(w); free(P); return 0;
}
static inline __attribute__((always_inline)) void q80_dgroup(RansDec *D, uint8_t *data, uint64_t b0, const int n, uint64_t bpr,
                                                             uint32_t *P, const int16_t *U, uint16_t *pd) {
    const int16_t *UH = U + (size_t)Q8_VROWS * 16;
    int ds[4] = {0}, nx[4] = {0}, sa[4] = {0};
    uint64_t c0[4] = {0}; uint32_t inv[4] = {0}; int8_t q[4][32];
    for (int L = 0; L < n; L++) {                           /* d, block by block: each given the one before */
        int ph = *pd >> 8, a, b, c, e;
        RD_GET(D, L, UH + (size_t)(Q8H_A + ph) * 16, a);
        RD_GET(D, L, UH + (size_t)(Q8H_B + ((a << 8) | ph)) * 16, b);
        RD_GET(D, L, UH + (size_t)(Q8H_C + ((a << 4) | b)) * 16, c);
        RD_GET(D, L, UH + (size_t)(Q8H_D + ((b << 4) | c)) * 16, e);
        uint16_t d = (uint16_t)((a << 12) | (b << 8) | (c << 4) | e); *pd = d;
        wr16(data + (b0 + L) * 34, d);
        ds[L] = sgn16(d); inv[L] = q40_inv((b0 + L) / bpr); c0[L] = (b0 + L) % bpr * 32;
    }
    for (int i = 0; i < 32; i++) {
        int ctx[4], hi[4];
        for (int L = 0; L < n; L++) { ctx[L] = q80_ctx1(P, c0[L] + i, inv[L], ds[L], nx[L], sa[L], i); RD_GET(D, L, U + (size_t)ctx[L] * 16, hi[L]); }
        for (int L = 0; L < n; L++) {
            int lo; RD_GET(D, L, U + (size_t)(Q8_CTX + ctx[L] * 16 + hi[L]) * 16, lo);
            int v = ((hi[L] << 4) | lo) - 128; q[L][i] = (int8_t)v;
            nx[L] += v >= 127 || v <= -127; sa[L] += v < 0 ? -v : v;
        }
    }
    for (int L = 0; L < n; L++) {
        uint8_t *B = data + (b0 + L) * 34; memcpy(B + 2, q[L], 32);
        q80_cols(P, b0 + L, bpr, B, ds[L], 1);
    }
}
/* decode; corrupt input gives wrong values, never reads outside in[0..len) */
HOT void q80_decode(const uint8_t *in, size_t len, uint8_t *data, uint64_t rows, uint64_t ne0, const int16_t *U) {
    if (len < 16) die("corrupt file");
    RansDec D; for (int L = 0; L < 4; L++) memcpy(&D.x[L], in + 4 * L, 4);
    D.p = in + 16; D.end = in + len;
    uint64_t bpr = ne0 / 32, nb = rows * bpr, b0 = 0; uint32_t *P = calloc(ne0, 4); if (!P) die("out of memory");
    uint16_t pd = 0;
    for (; b0 + 4 <= nb; b0 += 4) q80_dgroup(&D, data, b0, 4, bpr, P, U, &pd);
    if (b0 < nb) q80_dgroup(&D, data, b0, (int)(nb - b0), bpr, P, U, &pd);
    free(P);
}
static void q80_code(Coder *c, uint8_t *data, uint64_t rows, uint64_t ne0) {
    if (!c->dec && c->cnt_only) q80_count(c->cnt, data, rows, ne0);
    else die("internal: Q8_0 is coded with q80_encode / q80_decode");
}

/* ---------------- Q6_K in rANS lanes ---------------- */
/* The normal Q6_K model (not the lattice) with static tables in 4 rANS lanes, like Q4_K: blocks
   in groups of 4, block b's header in lane b % 4 (history: the lane's previous block), then each
   block's 256 values, lane L taking sub-blocks L, 4 + L, 8 + L and 12 + L (the only context
   inside a block, values <= 1 seen, restarts at every 16-value sub-block).
   - header: d (fp16) as four nibbles with the float contexts; each scale u = int8 + 128 as a
     high nibble given the previous scale and a low nibble given the high one and the previous
     scale ("previous": the sub-block before; for the first, the lane's previous block's last)
   - values: the column contexts (running mean and spread of the column in the rows above, in
     units of the sub-block's step), position, values <= 1 seen.
   The column sums are floats, so the encoder cannot take them back to run in reverse: the
   segment is cut into chunks of Q6_CHUNK blocks, each its own rANS stream (u32 word count, 4
   lane states, the words). The encoder computes a chunk's contexts forward, then codes it
   backwards. Table rows: Q6_CTX value contexts, then Q6H_CTX header contexts (16 symbols used). */
enum { Q6H_CTX = FL_HI + 256 + 4096, Q6H_N = 36 };    /* header rows (after the value rows), header symbols */
#define Q6_CHUNK 2048                                   /* blocks per rANS stream, a multiple of 4 */
/* the 36 header symbols of block B and their table rows, in coding order; P: the lane's previous block */
static void q6_hdr_syms(const uint8_t *B, const uint8_t *P, int base, uint16_t *ctx, uint8_t *sym) {
    kq_f16_syms(rd16(B + 208), P ? rd16(P + 208) : 0, base, ctx, sym);
    int up = P ? (int)(int8_t)P[192 + 15] + 128 : 128;
    for (int k = 0; k < 16; k++) {
        int u = (int)(int8_t)B[192 + k] + 128, hi = u >> 4;
        sym[4 + 2 * k] = (uint8_t)hi;       ctx[4 + 2 * k] = (uint16_t)(base + FL_HI + up);
        sym[5 + 2 * k] = (uint8_t)(u & 15); ctx[5 + 2 * k] = (uint16_t)(base + FL_HI + 256 + hi * 256 + up);
        up = u;
    }
}
/* the context bases of a block's 256 values (without the values <= 1 seen), from the column sums
   of the rows above (r of them): the column's mean in units of the sub-block's step (offset) and
   its variance in units of the step squared (width), with the divisions turned into
   multiplications by 1/r and 1/step (still exact IEEE operations) */
static void q6_bases(const Q6F *SW, const Q6F *SW2, uint64_t r, uint64_t bb, const Q6F *D, int *base) {
    if (r == 0) { for (int i = 0; i < 256; i++) base[i] = (12 * 8 + 7) * 48 + (i & 15) * 3; return; }
    Q6F ir = (Q6F)1 / (Q6F)r;
    for (int sb = 0; sb < 16; sb++) {
        Q6F Ds = D[sb]; const Q6F *S1 = SW + bb * 256 + sb * 16, *S2 = SW2 + bb * 256 + sb * 16;
        int *bs = base + sb * 16;
        if (Ds == 0 || !isfinite(Ds)) { for (int p = 0; p < 16; p++) bs[p] = (12 * 8 + 7) * 48 + p * 3; continue; }
        Q6F iD = (Q6F)1 / Ds, iD2 = iD * iD;
        /* branch-free, so the compiler can vectorize it; gives the same buckets as before:
           offset: a non-finite t takes the fallback 12 through t = 0; width: r2 >= W_C * 2^(j+2)
           (exact scalings of the 1/sqrt 2 constant) is width_bucket(r2) >= j, for finite r2 */
        for (int p = 0; p < 16; p++) {
            Q6F m = S1[p] * ir, t = m * iD;
            Q6F tc = t < -13 ? -13 : t > 13 ? 13 : t;
            tc = t - t == 0 ? tc : 0;                                   /* NaN, inf: 0 */
            int o = (int)(tc + (Q6F)25.5) - 25;
            int offb = (o < -12 ? -12 : o > 12 ? 12 : o) + 12;
            Q6F r2 = (S2[p] * ir - m * m) * iD2;
            int fin = r2 <= Q6F_MAX;                                     /* false for NaN and inf */
            int widb = ((r2 >= (Q6F)(W_C * 8)) + (r2 >= (Q6F)(W_C * 16)) + (r2 >= (Q6F)(W_C * 32)) + (r2 >= (Q6F)(W_C * 64)) +
                              (r2 >= (Q6F)(W_C * 128)) + (r2 >= (Q6F)(W_C * 256)) + (r2 >= (Q6F)(W_C * 512)));
            widb = fin ? widb : 0;
            bs[p] = (offb * 8 + widb) * 48 + p * 3;
        }
    }
}
static inline void q6_steps(const uint8_t *B, Q6F *D) {
    Q6F dd = (Q6F)h2f(rd16(B + 208));
    for (int k = 0; k < 16; k++) D[k] = dd * (Q6F)(int)(int8_t)B[192 + k];   /* signed step, exact */
}
static inline void q6_cols(Q6F *SW, Q6F *SW2, uint64_t bb, const Q6F *D, const uint8_t *q) {
    /* W is finite exactly when the step is (|q - 32| <= 32): test once per sub-block */
    for (int sb = 0; sb < 16; sb++) {
        Q6F Ds = D[sb]; if (!isfinite(Ds)) continue;
        Q6F *S1 = SW + bb * 256 + sb * 16, *S2 = SW2 + bb * 256 + sb * 16; const uint8_t *qs = q + sb * 16;
        for (int p = 0; p < 16; p++) { Q6F W = ((int)qs[p] - 32) * Ds; S1[p] += W; S2[p] += W * W; }
    }
}
/* block b, forward: header symbols and rows, values and rows (block order), then the column sums */
static void q6_fwd_block(const uint8_t *data, uint64_t b, uint64_t bpr, Q6F *SW, Q6F *SW2,
                         uint16_t *hc, uint8_t *hs, uint16_t *vc, uint8_t *q) {
    const uint8_t *B = data + b * 210; Q6F D[16]; int base[256];
    q6_hdr_syms(B, b >= 4 ? B - 4 * 210 : NULL, Q6_CTX, hc, hs);
    q6_unpack(B, q); q6_steps(B, D);
    q6_bases(SW, SW2, b / bpr, b % bpr, D, base);
    for (int sb = 0; sb < 16; sb++) {
        int nx = 0;
        for (int pos = 0; pos < 16; pos++) { int i = sb * 16 + pos; vc[i] = (uint16_t)(base[i] + (nx < 2 ? nx : 2)); nx += q[i] <= 1; }
    }
    q6_cols(SW, SW2, b % bpr, D, q);
}
static void q6_count(uint32_t *cnt, const uint8_t *data, uint64_t rows, uint64_t ne0) {
    uint64_t bpr = ne0 / 256, nb = rows * bpr;
    Q6F *SW = calloc(ne0, sizeof *SW), *SW2 = calloc(ne0, sizeof *SW2); if (!SW || !SW2) die("out of memory");
    for (uint64_t b = 0; b < nb; b++) {
        uint16_t hc[Q6H_N], vc[256]; uint8_t hs[Q6H_N], q[256];
        q6_fwd_block(data, b, bpr, SW, SW2, hc, hs, vc, q);
        for (int n = 0; n < Q6H_N; n++) cnt[((size_t)hc[n] << 6) + hs[n]]++;
        for (int i = 0; i < 256; i++) cnt[((size_t)vc[i] << 6) + q[i]]++;
    }
    free(SW); free(SW2);
}
/* encode: chunk by chunk, each forward for its contexts, then backwards into its own stream.
   Returns the length, or 0 if it does not fit in cap. */
static size_t q6_encode(const uint8_t *data, uint64_t rows, uint64_t ne0, const int16_t *U, uint8_t *out, size_t cap) {
    uint64_t bpr = ne0 / 256, nb = rows * bpr; size_t o = 0;
    Q6F *SW = calloc(ne0, sizeof *SW), *SW2 = calloc(ne0, sizeof *SW2); if (!SW || !SW2) die("out of memory");
    uint16_t *hc = xmalloc((size_t)Q6_CHUNK * Q6H_N * 2), *vc = xmalloc((size_t)Q6_CHUNK * 256 * 2);
    uint8_t *hs = xmalloc((size_t)Q6_CHUNK * Q6H_N), *vs = xmalloc((size_t)Q6_CHUNK * 256);
    size_t nw = (size_t)Q6_CHUNK * (256 + Q6H_N) + 8; uint16_t *w = xmalloc(nw * 2);
    for (uint64_t c0 = 0; c0 < nb; c0 += Q6_CHUNK) {
        int n = nb - c0 < Q6_CHUNK ? (int)(nb - c0) : Q6_CHUNK;
        for (int k = 0; k < n; k++)
            q6_fwd_block(data, c0 + k, bpr, SW, SW2, hc + (size_t)k * Q6H_N, hs + (size_t)k * Q6H_N, vc + (size_t)k * 256, vs + (size_t)k * 256);
        uint16_t *wp = w + nw; uint32_t x[4] = {RANS_L, RANS_L, RANS_L, RANS_L};
        for (int g0 = (n - 1) / 4 * 4; g0 >= 0; g0 -= 4) {
            int ng = n - g0 < 4 ? n - g0 : 4;
            for (int bi = ng - 1; bi >= 0; bi--) {             /* values, last block of the group first */
                const uint16_t *C = vc + (size_t)(g0 + bi) * 256; const uint8_t *S = vs + (size_t)(g0 + bi) * 256;
                for (int g = 3; g >= 0; g--) for (int pos = 15; pos >= 0; pos--) for (int L = 3; L >= 0; L--) {
                    int i = (4 * g + L) * 16 + pos;
                    if (!rans_put(&x[L], U + ((size_t)C[i] << 6), S[i], &wp, w)) die("internal: Q6_K word buffer");
                }
            }
            for (int k = Q6H_N - 1; k >= 0; k--) for (int L = ng - 1; L >= 0; L--) {
                size_t h = (size_t)(g0 + L) * Q6H_N + k;
                if (!rans_put(&x[L], U + ((size_t)hc[h] << 6), hs[h], &wp, w)) die("internal: Q6_K word buffer");
            }
        }
        uint32_t words = (uint32_t)(w + nw - wp);
        if (o + 20 + (size_t)words * 2 > cap) { o = 0; break; }
        memcpy(out + o, &words, 4); for (int L = 0; L < 4; L++) memcpy(out + o + 4 + 4 * L, &x[L], 4);
        memcpy(out + o + 20, wp, (size_t)words * 2); o += 20 + (size_t)words * 2;
    }
    free(SW); free(SW2); free(hc); free(vc); free(hs); free(vs); free(w);
    return o;
}
#define RD_GETK(D, L, ROW, K, OUT) do {                                                              \
    const int16_t *T_ = (ROW); uint32_t st_ = (D)->x[L], slot_ = st_ & (SM - 1); int sy_ = kq_search(T_, K, slot_); \
    uint32_t s0_ = sy_ ? (uint32_t)T_[sy_ - 1] : 0, f_ = (uint32_t)T_[sy_] - s0_;                    \
    st_ = f_ * (st_ >> SM_BITS) + slot_ - s0_;                                                       \
    RANS_REFILL(st_, (D)->p, (D)->end);                                                              \
    (D)->x[L] = st_; (OUT) = sy_; } while (0)
/* a symbol from 64-symbol row ROW (Q6_K, both models) by lookup; U and Lt from the caller */
#define RD_GET64(D, L, ROW, OUT) do {                                                                \
    size_t o_ = (size_t)(ROW) << 6; const int16_t *T_ = U + o_;                                      \
    uint32_t st_ = (D)->x[L], slot_ = st_ & (SM - 1); int sy_ = LUT_FIND(T_, Lt + ((size_t)(ROW) * Q6_LUTB), SM_BITS - __builtin_ctz(Q6_LUTB), 64, slot_); \
    uint32_t s0_ = sy_ ? (uint32_t)T_[sy_ - 1] : 0, f_ = (uint32_t)T_[sy_] - s0_;                    \
    st_ = f_ * (st_ >> SM_BITS) + slot_ - s0_;                                                       \
    RANS_REFILL(st_, (D)->p, (D)->end);                                                              \
    (D)->x[L] = st_; (OUT) = sy_; } while (0)
/* the headers of a group of n blocks, lane by lane interleaved, written into the blocks; UH: header rows */
static inline __attribute__((always_inline)) void q6_dhdr(RansDec *Dc, uint8_t *data, uint64_t b0, const int n,
                                                          const int16_t *UH, int *pa, int *pb, int *pu) {
    int a[4] = {0}, bn[4] = {0}, cn[4] = {0}, dn[4] = {0}, sc[4][16];
    #define Q6H(L, ROW, OUT) do { int s_; RD_GETK(Dc, L, UH + ((size_t)(ROW) << 6), 16, s_); (OUT) = s_ & 15; } while (0)
    for (int L = 0; L < n; L++) Q6H(L, pa[L], a[L]);
    for (int L = 0; L < n; L++) Q6H(L, 16 + ((a[L] << 4) | pb[L]), bn[L]);
    for (int L = 0; L < n; L++) Q6H(L, 272 + ((a[L] << 4) | bn[L]), cn[L]);
    for (int L = 0; L < n; L++) Q6H(L, 528 + cn[L], dn[L]);
    for (int k = 0; k < 16; k++) {
        int hi[4] = {0};
        for (int L = 0; L < n; L++) Q6H(L, FL_HI + pu[L], hi[L]);
        for (int L = 0; L < n; L++) { int lo; Q6H(L, FL_HI + 256 + hi[L] * 256 + pu[L], lo); sc[L][k] = (hi[L] << 4) | lo; pu[L] = sc[L][k]; }
    }
    #undef Q6H
    for (int bi = 0; bi < n; bi++) {
        uint8_t *B = data + (b0 + bi) * 210;
        wr16(B + 208, (uint16_t)((a[bi] << 12) | (bn[bi] << 8) | (cn[bi] << 4) | dn[bi])); pa[bi] = a[bi]; pb[bi] = bn[bi];
        for (int k = 0; k < 16; k++) B[192 + k] = (uint8_t)(int8_t)(sc[bi][k] - 128);
    }
}
/* one group of n blocks: the headers, then the blocks' values */
static inline __attribute__((always_inline)) void q6_dgroup(RansDec *Dc, uint8_t *data, uint64_t b0, const int n, uint64_t bpr,
                                                            Q6F *SW, Q6F *SW2, const int16_t *U, int *pa, int *pb, int *pu, const uint8_t *Lt) {
    q6_dhdr(Dc, data, b0, n, U + ((size_t)Q6_CTX << 6), pa, pb, pu);
    for (int bi = 0; bi < n; bi++) {
        uint8_t *B = data + (b0 + bi) * 210; uint8_t q[256]; Q6F D[16]; int base[256];
        q6_steps(B, D);
        q6_bases(SW, SW2, (b0 + bi) / bpr, (b0 + bi) % bpr, D, base);
        for (int g = 0; g < 4; g++) {
            int nx[4] = {0};
            for (int pos = 0; pos < 16; pos++) for (int L = 0; L < 4; L++) {
                int i = (4 * g + L) * 16 + pos, sy;
#if Q6_PF > 0
                /* the next contexts are known but for nx: fetch the likely rows while this symbol
                   decodes, so the loads after it do not wait for memory (on phones the
                   tables do not fit in cache, and the decoder cannot run ahead of its own symbols) */
                if (pos + Q6_PF < 16) {
                    size_t r_ = (size_t)base[i + Q6_PF] + (nx[L] < 2 ? nx[L] : 2);
                    __builtin_prefetch(U + (r_ << 6)); __builtin_prefetch(U + (r_ << 6) + 32);
#ifndef TENZ_NO_LUT
                    __builtin_prefetch(Lt + r_ * Q6_LUTB);
#endif
                }
#endif
                RD_GET64(Dc, L, base[i] + (nx[L] < 2 ? nx[L] : 2), sy);
                q[i] = (uint8_t)sy; nx[L] += sy <= 1;
            }
        }
        q6_pack(B, q);
        q6_cols(SW, SW2, (b0 + bi) % bpr, D, q);
    }
}
/* decode; corrupt input gives wrong values, never reads outside in[0..len) */
HOT void q6_decode(const uint8_t *in, size_t len, uint8_t *data, uint64_t rows, uint64_t ne0, const int16_t *U, const uint8_t *Lt) {
    uint64_t bpr = ne0 / 256, nb = rows * bpr; const uint8_t *p = in, *end = in + len;
    Q6F *SW = calloc(ne0, sizeof *SW), *SW2 = calloc(ne0, sizeof *SW2); if (!SW || !SW2) die("out of memory");
    int pa[4] = {0}, pb[4] = {0}, pu[4] = {128, 128, 128, 128};
    for (uint64_t c0 = 0; c0 < nb; c0 += Q6_CHUNK) {
        uint64_t n = nb - c0 < Q6_CHUNK ? nb - c0 : Q6_CHUNK; uint32_t words;
        if (end - p < 20) die("corrupt file");
        memcpy(&words, p, 4); RansDec D; for (int L = 0; L < 4; L++) memcpy(&D.x[L], p + 4 + 4 * L, 4);
        if ((uint64_t)(end - p - 20) < (uint64_t)words * 2) die("corrupt file");
        D.p = p + 20; D.end = D.p + (size_t)words * 2; p = D.end;
        uint64_t b = c0;
        for (; b + 4 <= c0 + n; b += 4) q6_dgroup(&D, data, b, 4, bpr, SW, SW2, U, pa, pb, pu, Lt);
        if (b < c0 + n) q6_dgroup(&D, data, b, (int)(c0 + n - b), bpr, SW, SW2, U, pa, pb, pu, Lt);
    }
    free(SW); free(SW2);
}

/* ---------------- Q6_K, AWQ lattice, in rANS lanes ---------------- */
/* For Q6_K converted from 4-bit AWQ/GPTQ checkpoints or trained for quantization: in each 16-value
   sub-block, y = q - 32 = rint(k * span / m) + e (lattice_fit). Same headers and chunks as the
   normal model, its own static tables; lane L takes sub-blocks 4L .. 4L + 3 of each block, so a
   lane's previous sub-block is usually the one before. Per sub-block, in its lane: m given the
   lane's previous m, the span given m / 4; per value: k + 32 given 2 m + (k = -m seen), the
   column's share of positive k (as Q4_0's buckets) and its average |k| / m (4 buckets), both over
   the rows above; then e + 3 in 0..6 given m, or 7 and then q itself (escape). The number of
   symbols per block varies, so the encoder records a chunk's symbols in the decoder's order
   (lane, row, symbol) and codes them backwards. */
enum { L_KM = 0, L_ES = 33, L_K = 42, L_KN = 66 * 8 * 4, L_E = L_K + L_KN, L_ESC = L_E + 33, Q6L_N = L_ESC + 1 };
/* the columns over the rows above: how many k > 0, and the sum of |k| * 32 / m (Q6L_DIV) */
typedef struct { uint32_t *pos, *mag; uint64_t bpr; } LCol;
static uint8_t Q6L_DIV[33][33];
static void q6l_init(void) { for (int m = 1; m <= 32; m++) for (int k = 0; k <= 32; k++) Q6L_DIV[m][k] = (uint8_t)(k * 32 / m); }
/* per row: the share's reciprocal and the magnitude thresholds (avg > 10, 16, 24 <=> mag >= 11 r ...) */
typedef struct { uint32_t inv, t1, t2, t3; int r0; } LRow;
static inline LRow q6l_row(uint64_t r) { LRow w = {q40_inv(r), (uint32_t)(11 * r), (uint32_t)(17 * r), (uint32_t)(25 * r), r == 0}; return w; }
static inline int q6l_kctx(const LCol *C, uint64_t col, const LRow *w, int m, int seen) {
    uint32_t p = C->pos[col], g = C->mag[col];
    int cb = Q40_CB[((uint64_t)(p + 1) * w->inv) >> 26] >> 6, mb = w->r0 ? 1 : (g >= w->t1) + (g >= w->t2) + (g >= w->t3);
    return L_K + ((2 * m + seen) * 8 + cb) * 4 + mb;
}
static inline void q6l_col(LCol *C, uint64_t col, int k, int m) { C->pos[col] += k > 0; C->mag[col] += Q6L_DIV[m][k < 0 ? -k : k]; }
#define Q6L_CHUNK 512                                   /* blocks per stream, a multiple of 4 */
typedef struct { uint16_t row; uint8_t sym, lane; } Ev;
/* a group of n blocks from b0: its symbols in the decoder's order; lkm: each lane's previous m */
static size_t q6l_fwd_group(const uint8_t *data, uint64_t b0, int n, int *lkm, Ev *ev, LCol *C) {
    size_t ne = 0; uint16_t hc[4][Q6H_N]; uint8_t hs[4][Q6H_N];
    for (int L = 0; L < n; L++) { const uint8_t *B = data + (b0 + L) * 210; q6_hdr_syms(B, b0 + L >= 4 ? B - 4 * 210 : NULL, Q6L_N, hc[L], hs[L]); }
    for (int k = 0; k < Q6H_N; k++) for (int L = 0; L < n; L++) ev[ne++] = (Ev){hc[L][k], hs[L][k], (uint8_t)L};
    for (int bi = 0; bi < n; bi++) {
        uint8_t q[256]; q6_unpack(data + (b0 + bi) * 210, q);
        uint64_t b = b0 + bi, c0 = b % C->bpr * 256; LRow w = q6l_row(b / C->bpr);
        int kb[256], mb[16];
        for (int g = 0; g < 4; g++) {
            int km[4], es[4], kk[4][16], ee[4][16], seen[4] = {0};
            for (int L = 0; L < 4; L++) {
                int y[16]; for (int i = 0; i < 16; i++) y[i] = q[(4 * L + g) * 16 + i] - 32;
                lattice_fit(y, &km[L], &es[L], kk[L], ee[L]);
            }
            for (int L = 0; L < 4; L++) { ev[ne++] = (Ev){(uint16_t)(L_KM + lkm[L]), (uint8_t)km[L], (uint8_t)L}; lkm[L] = km[L]; }
            for (int L = 0; L < 4; L++) ev[ne++] = (Ev){(uint16_t)(L_ES + (km[L] >> 2)), (uint8_t)es[L], (uint8_t)L};
            for (int i = 0; i < 16; i++) {
                int esc[4];
                for (int L = 0; L < 4; L++) {
                    int sb = 4 * L + g;
                    ev[ne++] = (Ev){(uint16_t)q6l_kctx(C, c0 + sb * 16 + i, &w, km[L], seen[L]), (uint8_t)(kk[L][i] + 32), (uint8_t)L};
                    kb[sb * 16 + i] = kk[L][i]; mb[sb] = km[L];
                }
                for (int L = 0; L < 4; L++) {
                    int e = ee[L][i]; esc[L] = e < -3 || e > 3;
                    ev[ne++] = (Ev){(uint16_t)(L_E + km[L]), (uint8_t)(esc[L] ? 7 : e + 3), (uint8_t)L};
                }
                for (int L = 0; L < 4; L++) {
                    if (esc[L]) ev[ne++] = (Ev){L_ESC, q[(4 * L + g) * 16 + i], (uint8_t)L};
                    seen[L] |= kk[L][i] == -km[L];
                }
            }
        }
        for (int i = 0; i < 256; i++) q6l_col(C, c0 + i, kb[i], mb[i >> 4]);
    }
    return ne;
}
#define Q6L_EVMAX (Q6H_N * 4 + 4 * (16 * 2 + 256 * 3))  /* symbols per group of 4 blocks, at most */
static void q6l_count(uint32_t *cnt, const uint8_t *data, uint64_t rows, uint64_t ne0) {
    uint64_t nb = rows * (ne0 / 256); int lkm[4] = {0}; Ev *ev = xmalloc(Q6L_EVMAX * sizeof *ev);
    LCol C = {calloc(ne0, 4), calloc(ne0, 4), ne0 / 256}; if (!C.pos || !C.mag) die("out of memory");
    for (uint64_t b0 = 0; b0 < nb; b0 += 4) {
        size_t ne = q6l_fwd_group(data, b0, nb - b0 < 4 ? (int)(nb - b0) : 4, lkm, ev, &C);
        for (size_t e = 0; e < ne; e++) cnt[((size_t)ev[e].row << 6) + ev[e].sym]++;
    }
    free(ev); free(C.pos); free(C.mag);
}
static size_t q6l_encode(const uint8_t *data, uint64_t rows, uint64_t ne0, const int16_t *U, uint8_t *out, size_t cap) {
    uint64_t nb = rows * (ne0 / 256); size_t o = 0; int lkm[4] = {0};
    LCol C = {calloc(ne0, 4), calloc(ne0, 4), ne0 / 256}; if (!C.pos || !C.mag) die("out of memory");
    size_t evcap = (size_t)Q6L_CHUNK / 4 * Q6L_EVMAX; Ev *ev = xmalloc(evcap * sizeof *ev); uint16_t *w = xmalloc(evcap * 2 + 16);
    for (uint64_t c0 = 0; c0 < nb; c0 += Q6L_CHUNK) {
        uint64_t n = nb - c0 < Q6L_CHUNK ? nb - c0 : Q6L_CHUNK; size_t ne = 0;
        for (uint64_t b0 = c0; b0 < c0 + n; b0 += 4) ne += q6l_fwd_group(data, b0, c0 + n - b0 < 4 ? (int)(c0 + n - b0) : 4, lkm, ev + ne, &C);
        uint16_t *wp = w + evcap + 8; uint32_t x[4] = {RANS_L, RANS_L, RANS_L, RANS_L};
        for (size_t e = ne; e-- > 0;)
            if (!rans_put(&x[ev[e].lane], U + ((size_t)ev[e].row << 6), ev[e].sym, &wp, w)) die("internal: Q6_K lattice word buffer");
        uint32_t words = (uint32_t)(w + evcap + 8 - wp);
        if (o + 20 + (size_t)words * 2 > cap) { o = 0; break; }
        memcpy(out + o, &words, 4); for (int L = 0; L < 4; L++) memcpy(out + o + 4 + 4 * L, &x[L], 4);
        memcpy(out + o + 20, wp, (size_t)words * 2); o += 20 + (size_t)words * 2;
    }
    free(ev); free(w); free(C.pos); free(C.mag);
    return o;
}
/* decode one group: headers, then each block's sub-blocks, 4 lanes at once */
static inline __attribute__((always_inline)) void q6l_dgroup(RansDec *Dc, uint8_t *data, uint64_t b0, const int n,
                                                             const int16_t *U, int *pa, int *pb, int *pu, int *lkm, LCol *C, const uint8_t *Lt) {
    q6_dhdr(Dc, data, b0, n, U + ((size_t)Q6L_N << 6), pa, pb, pu);
    for (int bi = 0; bi < n; bi++) {
        uint8_t q[256]; uint64_t b = b0 + bi, c0 = b % C->bpr * 256; LRow w = q6l_row(b / C->bpr);
        for (int g = 0; g < 4; g++) {
            int km[4], es[4], seen[4] = {0}; double rr[4];
            for (int L = 0; L < 4; L++) {
                RD_GET64(Dc, L, L_KM + lkm[L], km[L]);
                if (km[L] < 1 || km[L] > 32) die("corrupt file");    /* the encoder never writes 0 or > 32 */
                lkm[L] = km[L];
            }
            for (int L = 0; L < 4; L++) { RD_GET64(Dc, L, L_ES + (km[L] >> 2), es[L]); rr[L] = (double)(es[L] ? es[L] : 32) / km[L]; }
            for (int i = 0; i < 16; i++) {
                int k[4], e[4];
                for (int L = 0; L < 4; L++) {
                    RD_GET64(Dc, L, q6l_kctx(C, c0 + (4 * L + g) * 16 + i, &w, km[L], seen[L]), k[L]); k[L] -= 32;
                }
                for (int L = 0; L < 4; L++) RD_GET64(Dc, L, L_E + km[L], e[L]);
                for (int L = 0; L < 4; L++) {
                    int v;
                    if (e[L] >= 7) RD_GET64(Dc, L, L_ESC, v);
                    else v = 32 + (int)rint(k[L] * rr[L]) + e[L] - 3;
                    q[(4 * L + g) * 16 + i] = (uint8_t)(v & 63); seen[L] |= k[L] == -km[L];
                    q6l_col(C, c0 + (4 * L + g) * 16 + i, k[L], km[L]);   /* each column once per block */
                }
            }
        }
        q6_pack(data + (b0 + bi) * 210, q);
    }
}
HOT void q6l_decode(const uint8_t *in, size_t len, uint8_t *data, uint64_t rows, uint64_t ne0, const int16_t *U, const uint8_t *Lt) {
    uint64_t nb = rows * (ne0 / 256); const uint8_t *p = in, *end = in + len;
    int pa[4] = {0}, pb[4] = {0}, pu[4] = {128, 128, 128, 128}, lkm[4] = {0};
    LCol C = {calloc(ne0, 4), calloc(ne0, 4), ne0 / 256}; if (!C.pos || !C.mag) die("out of memory");
    for (uint64_t c0 = 0; c0 < nb; c0 += Q6L_CHUNK) {
        uint64_t n = nb - c0 < Q6L_CHUNK ? nb - c0 : Q6L_CHUNK; uint32_t words;
        if (end - p < 20) die("corrupt file");
        memcpy(&words, p, 4); RansDec D; for (int L = 0; L < 4; L++) memcpy(&D.x[L], p + 4 + 4 * L, 4);
        if ((uint64_t)(end - p - 20) < (uint64_t)words * 2) die("corrupt file");
        D.p = p + 20; D.end = D.p + (size_t)words * 2; p = D.end;
        uint64_t b = c0;
        for (; b + 4 <= c0 + n; b += 4) q6l_dgroup(&D, data, b, 4, U, pa, pb, pu, lkm, &C, Lt);
        if (b < c0 + n) q6l_dgroup(&D, data, b, (int)(c0 + n - b), U, pa, pb, pu, lkm, &C, Lt);
    }
    free(C.pos); free(C.mag);
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

/* ---------------- GGUF string arrays ---------------- */
/* A run of whole records of a GGUF string array (tokens, merges): each a u64 length, then its
   bytes. The lengths, 8 bytes each and nearly all zero, are much of the metadata and cost the byte
   model full price, so they are coded first with a small model: the bit length of length + 1 given
   the previous one (6-bit tree), then the bits below the leading one, each given its bit length,
   position and the bit before. The strings, put together, then go to the byte model alone. The
   decoder reads lengths until they fill the segment, so no count is stored. */
HOT void strs_code_impl(Coder *c, const int dec, uint8_t *data, uint64_t n) {
    Bit *eb = bits_new(64 * 64), *mb = bits_new(64 * 64 * 2);
    uint64_t pos = 0, sbytes = 0; int pe = 0;
    while (pos < n) {
        if (n - pos < 8) die(dec ? "corrupt file" : "internal: string array");
        uint64_t room = n - pos - 8, L = 0, v = 1;
        if (!dec) {
            L = rd64(data + pos);
            if (L > room) die("internal: string array");
            v = L + 1;
        }
        int e1 = dec ? 0 : 63 - __builtin_clzll(v), node = 1;
        for (int i = 5; i >= 0; i--) node = (node << 1) | cbit(c, dec, &eb[pe * 64 + node], (e1 >> i) & 1);
        e1 = node - 64;
        uint64_t u = 1; int pb = 0;
        for (int i = e1 - 1; i >= 0; i--) {
            int bit = cbit(c, dec, &mb[((size_t)e1 * 64 + i) * 2 + pb], dec ? 0 : (int)((v >> i) & 1));
            u = (u << 1) | (uint64_t)bit; pb = bit;
        }
        if (dec) { L = u - 1; if (L > room) die("corrupt file"); wr64(data + pos, L); }
        pe = e1; pos += 8 + L; sbytes += L;
    }
    free(eb); free(mb);
    if (sbytes) {
        uint8_t *s = xmalloc(sbytes); uint64_t q = 0;
        if (!dec) for (pos = 0; pos < n;) { uint64_t L = rd64(data + pos); memcpy(s + q, data + pos + 8, L); q += L; pos += 8 + L; }
        bytes_code_impl(c, dec, s, sbytes);
        if (dec) for (pos = 0, q = 0; pos < n;) { uint64_t L = rd64(data + pos); memcpy(data + pos + 8, s + q, L); q += L; pos += 8 + L; }
        free(s);
    }
}
MODEL(strs_code, uint8_t *data X uint64_t n, data X n)

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
        case T_Q4K: kq_code(c, p, s->size / 144, 4, (s->model & 255) == M_QUICK); break;
        case T_Q5K: kq_code(c, p, s->size / 176, 5, (s->model & 255) == M_QUICK); break;
        case T_Q6K:
            if (!c->dec && c->cnt_only) q6_count(c->cnt, p, s->rows, s->ne0);
            else die("internal: Q6_K is coded with q6_encode / q6_decode");
            break;
        case T_Q8_0: q80_code(c, p, s->rows, s->ne0); break;
        case T_Q4_0: q40_code(c, p, s->rows, s->ne0); break;
        case T_BYTES: bytes_code(c, p, s->size); break;
        case T_STRS: strs_code(c, p, s->size); break;
        case T_M32: float_code(c, p, s->rows * s->ne0, T_F32); break;
        default: float_code(c, p, s->rows * s->ne0, s->type);
    }
}

/* ---------------- static value tables ---------------- */
/* types with static value tables (every tensor type, and metadata arrays of 32-bit values, which
   share F32's tables): index, symbols per context, contexts */
#define NSTATIC 9                                /* 8: Q6_K lattice segments */
static int kq_index(uint32_t type) {
    return type == T_Q4K ? 0 : type == T_Q5K ? 1 : type == T_Q6K ? 2 : type == T_Q8_0 ? 3 :
           type == T_BF16 ? 4 : type == T_F16 ? 5 : type == T_F32 || type == T_M32 ? 6 : type == T_Q4_0 ? 7 : -1;
}
static const int KQ_SYM[NSTATIC] = {16, 32, 64, 16, 16, 16, 16, 16, 64},
                 KQ_CTX[NSTATIC] = {32 * 36 + KH_CTX, 64 * 36 + KH_CTX, Q6_CTX + Q6H_CTX, Q8_CTX * 17 + Q8H_CTX, FL_HI, FL_HI, FL_HI + FL_LO, Q40_VCTX + FL_HI, Q6L_N + Q6H_CTX};
/* segments coded with static tables: tensor types, not raw */
static int is_static(const Seg *s) {
    int md = s->model & 255;
    return kq_index(s->type) >= 0 && (md == M_NORMAL || (md == M_LATTICE && s->type == T_Q6K) ||
                                      (md == M_QUICK && (s->type == T_Q4K || s->type == T_Q5K)));
}
/* the segment's table set: by type, and Q6_K lattice segments their own */
static int tab_index(const Seg *s) { return s->type == T_Q6K && (s->model & 255) == M_LATTICE ? 8 : kq_index(s->type); }
static int is_kq45(uint32_t type) { return type == T_Q4K || type == T_Q5K; }
/* static segments, all coded in rANS lanes: floats, K-quants (both Q6_K models), Q4_0, Q8_0 */
static int is_rans(const Seg *s) { return is_static(s); }
/* decoding by lookup: the types with rows of 32 or 64 symbols */
static int uses_lut(const Seg *s) {
#ifdef TENZ_NO_LUT
    (void)s; return 0;
#else
    return s->type == T_Q5K || s->type == T_Q6K;
#endif
}
/* bound on a rANS segment's table section: per context a few bits for the flags and at most a few
   bytes per symbol for the differences from the seed */
static uint64_t rans_tab_cap(const Seg *s) {
    int t = tab_index(s); uint64_t c = (uint64_t)KQ_CTX[t] * KQ_SYM[t] * 4 + 4096;
    return c < s->size ? c : s->size;
}

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

/* one seed per type: the counts of all its static segments together */
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
/* Metadata splitter. A GGUF metadata array (key type 9) at r: arrays of 32-bit values (u32, i32,
   f32) of at least M32_MIN entries become one T_M32 segment, coded with the float model; string
   arrays of at least STRS_MIN bytes become T_STRS segments of whole records, cut at record
   boundaries into pieces of about META_PIECE (the last one at least half that), so they are coded
   in parallel. Anything else is skipped and stays with the byte model. */
#define META_PIECE ((uint64_t)1 << 20)
#define STRS_MIN ((uint64_t)64 << 10)
#define M32_MIN 1024
static void meta_add(Seg **ex, int *nex, int *cap, Seg s) {
    if (*nex == *cap) { *cap = *cap * 2 + 16; *ex = realloc(*ex, (size_t)*cap * sizeof **ex); if (!*ex) die("out of memory"); }
    (*ex)[(*nex)++] = s;
}
static void meta_array(Rd *r, Seg **ex, int *nex, int *cap) {
    size_t at = r->pos;
    uint32_t et = ru32(r); uint64_t cnt = ru64(r); size_t p0 = r->pos;
    if ((et == 4 || et == 5 || et == 6) && cnt >= M32_MIN) {
        if (cnt > r->n / 4) die("bad array");
        need(r, cnt * 4); r->pos += cnt * 4;
        meta_add(ex, nex, cap, (Seg){p0, cnt * 4, cnt, 1, T_M32, M_NORMAL});
        return;
    }
    if (et == 8) {
        for (uint64_t i = 0; i < cnt; i++) { uint64_t l = ru64(r); need(r, l); r->pos += l; }   /* the end, checked */
        size_t end = r->pos;
        if (end - p0 >= STRS_MIN) {
            size_t q = p0, a = p0;
            for (uint64_t i = 0; i < cnt; i++) {
                q += 8 + rd64(r->p + q);
                if (q - a >= META_PIECE && end - q >= META_PIECE / 2) {
                    meta_add(ex, nex, cap, (Seg){a, q - a, q - a, 1, T_STRS, M_NORMAL}); a = q;
                }
            }
            if (end > a) meta_add(ex, nex, cap, (Seg){a, end - a, end - a, 1, T_STRS, M_NORMAL});
        }
        return;
    }
    r->pos = at; skip_val(r, 9, 0);
}
static int cmp_seg(const void *a, const void *b) {
    uint64_t x = ((const Seg *)a)->off, y = ((const Seg *)b)->off; return x < y ? -1 : x > y;
}
static Seg *build_segments(uint64_t n, uint64_t seg_target, uint64_t nt, uint32_t *types, uint64_t *offs,
                           uint64_t *ne0s, uint64_t *nes, Seg *ex, int nex, int *nseg);

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
    return build_segments(n, seg_target, nt, types, offs, ne0s, nes, NULL, 0, nseg);
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
    Seg *ex = NULL; int nex = 0, excap = 0;              /* metadata arrays split out of the byte model */
    for (uint64_t i = 0; i < nkv; i++) {
        uint64_t kl = ru64(&r); need(&r, kl);
        int is_align = kl == 17 && !memcmp(p + r.pos, "general.alignment", 17);
        r.pos += kl; uint32_t t = ru32(&r);
        if (is_align && t == 4) { align = ru32(&r); if (!align) align = 32; }
        else if (t == 9) meta_array(&r, &ex, &nex, &excap);
        else skip_val(&r, t, 0);
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
    return build_segments(n, seg_target, nt, types, offs, ne0s, nes, ex, nex, nseg);
}
/* tensors (type, absolute offset, first dimension, element count) -> segments; the metadata
   segments ex (from meta_array) are added as they are; every byte not in a supported tensor or a
   metadata segment goes to byte-model segments. Takes ownership of the four arrays and of ex. */
/* GGUF tensors of types tenz has no model for are stored as they are: the byte model gains little
   on quantized weights and would take hours on a large model. Bytes per GGML type, for the report. */
#define NGGML 40
static uint64_t g_unmodelled[NGGML + 1];               /* [NGGML]: type ids beyond the table */
static const char *ggml_name(int t) {
    static const char *N[NGGML] = {"F32", "F16", "Q4_0", "Q4_1", "type 4", "type 5", "Q5_0", "Q5_1", "Q8_0", "Q8_1",
        "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K", "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ1_S", "IQ4_NL", "IQ3_S",
        "IQ2_S", "IQ4_XS", "I8", "I16", "I32", "I64", "F64", "IQ1_M", "BF16", "type 31", "type 32", "type 33",
        "TQ1_0", "TQ2_0", "type 36", "type 37", "type 38", "MXFP4"};
    return t < NGGML ? N[t] : "other types";
}
typedef struct { uint64_t a, b; uint32_t t; } Span;
static int cmp_u64(const void *a, const void *b) { uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b; return x < y ? -1 : x > y; }
static int cmp_span(const void *a, const void *b) { return cmp_u64(&((const Span *)a)->a, &((const Span *)b)->a); }

static Seg *build_segments(uint64_t n, uint64_t seg_target, uint64_t nt, uint32_t *types, uint64_t *offs,
                           uint64_t *ne0s, uint64_t *nes, Seg *ex, int nex, int *nseg) {
    Seg *s = NULL; int cnt = 0, cap = 0;
    /* tensors of other types: from their offset to the next tensor's (their size needs no table) */
    memset(g_unmodelled, 0, sizeof g_unmodelled);
    uint64_t *so = xmalloc(nt * 8 + 8); memcpy(so, offs, nt * 8); qsort(so, nt, 8, cmp_u64);
    Span *R = xmalloc(nt * sizeof *R + sizeof *R); uint64_t nr = 0;
    for (uint64_t i = 0; i < nt; i++) {
        if (type_index(types[i]) >= 0 || offs[i] >= n) continue;
        uint64_t lo = 0, hi = nt;                          /* first offset above this one */
        while (lo < hi) { uint64_t m = (lo + hi) / 2; if (so[m] <= offs[i]) lo = m + 1; else hi = m; }
        R[nr++] = (Span){offs[i], lo < nt && so[lo] < n ? so[lo] : n, types[i]};
    }
    free(so); if (nr) qsort(R, nr, sizeof *R, cmp_span);
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
    for (int k = 0; k < nex; k++) {                      /* metadata segments, checked against the file */
        if (ex[k].off > n || ex[k].size > n - ex[k].off) die("metadata beyond end of file");
        if (cnt == cap) { cap = cap * 2 + 64; s = realloc(s, cap * sizeof *s); if (!s) die("out of memory"); }
        s[cnt++] = ex[k];
    }
    free(ex);
    free(offs); free(ne0s); free(nes); free(types);
    if (cnt) qsort(s, cnt, sizeof *s, cmp_seg);
    for (int i = 1; i < cnt; i++) if (s[i].off < s[i - 1].off + s[i - 1].size) die("overlapping tensors");
    int nt_seg = cnt; uint64_t at = 0, ri = 0;
    for (int i = 0; i <= nt_seg; i++) {
        uint64_t gend = i < nt_seg ? s[i].off : n;
        for (uint64_t ga = at; ga < gend;) {             /* the gap, without the unmodelled tensors */
            while (ri < nr && R[ri].b <= ga) ri++;
            uint64_t end = gend;
            if (ri < nr && R[ri].a < gend) {
                if (R[ri].a <= ga) {                         /* stored as is: left between the segments */
                    uint64_t e = R[ri].b < gend ? R[ri].b : gend;
                    g_unmodelled[R[ri].t < NGGML ? R[ri].t : NGGML] += e - ga; ga = e; continue;
                }
                end = R[ri].a;
            }
            uint64_t from = ga; ga = end;
            /* metadata in 2 MiB pieces, coded in parallel: costs about 3% of the metadata's coded size
               (each piece learns from scratch), but a model's metadata (10+ MiB of tokenizer data) is
               otherwise one segment on one thread, the longest of the run on small models and on
               machines with many cores */
            uint64_t bt = (uint64_t)2 << 20;
            for (uint64_t a = from; a < end; a += bt) {
                uint64_t len = end - a < bt ? end - a : bt;
                if (cnt == cap) { cap = cap * 2 + 64; s = realloc(s, cap * sizeof *s); if (!s) die("out of memory"); }
                s[cnt++] = (Seg){a, len, len, 1, T_BYTES, M_NORMAL};
            }
        }
        if (i < nt_seg) at = s[i].off + s[i].size;
    }
    free(R);
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
    double busy, cpu, wait_mem, wait_dev;          /* segment wall time, its CPU time, threads waiting: budget full, device low */
    double pwall[4], pbusy[4]; int nruns;          /* per run (pass): wall time, segment time */
    /* memory: a segment starts only if its estimated need fits in the budget, and (live) the
       device still has that much plus the reserve free; one segment can always run */
    uint64_t *need, used, peak, budget, reserve, free0; int live, inflight, max_inflight; char *taken;
    int quick;                                     /* compress -q: Q4_K and Q5_K in the quick model */
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
        case T_Q4K: case T_Q5K: return 0; /* static tables only: counted in seg_need */
        case T_Q4_0: case T_Q8_0: return 4 * s->ne0;   /* column counts */
        case T_Q6K: return B * (3 * 65536 + 33 * 64 + 9 * 64 + 66 * 128 + 33 * 8) + 2 * (uint64_t)(4096 * 16 + 66 * 128 + 33 * 72)  /* lattice */
                           + 2 * sizeof(Q6F) * s->ne0 + (uint64_t)Q6_CHUNK * (256 + Q6H_N) * 5;                                       /* rANS: column sums, chunk */
        case T_BYTES: { int hb = bytes_hb(s->size);
                        return B * ((4ull << hb) + 256 + 65536 + 128) + (4ull << (hb - 2)) + 256 * 3 * NIN * 4; }
        case T_STRS: { int hb = bytes_hb(s->size);            /* the byte model, the length model, the strings put together */
                       return B * ((4ull << hb) + 256 + 65536 + 128 + 3 * 64 * 64) + (4ull << (hb - 2)) + 256 * 3 * NIN * 4 + s->size; }
        default: return B * (4 + 3 * 64 * 256 + 512);
    }
}
/* peak memory of one segment: tables plus buffers. Compress: the coded output (Q6_K may code
   twice and keep the smaller). Test: output plus the decoded copy. Decompress: the decoded copy.
   Decoding Q5_K and Q6_K also holds a lookup table, half the size of a table set. */
static uint64_t seg_need(const Seg *s, int mode, int phase) {
    uint64_t z = s->size, slack = 1u << 20, mb = model_bytes(s);
    int t = tab_index(s);
    uint64_t tab = t >= 0 ? (uint64_t)KQ_CTX[t] * KQ_SYM[t] * 2 : 0;          /* one table set, int16 */
    uint64_t lut = !uses_lut(s) ? 0 : s->type == T_Q6K ? (uint64_t)KQ_CTX[t] * Q6_LUTB : tab / 2;
    if (mode == 1) return (s->model & 255) == M_RAW ? slack : slack + mb + z + (is_static(s) ? tab + lut : 0);
    if (t < 0) return slack + mb + (mode == 2 ? 2 : 1) * z;
    if (phase == 1) return slack + mb + 2 * tab + (s->type == T_Q6K ? 2 * z : 0);   /* counts; Q6_K may try the lattice */
    /* rANS: the table section, then the output and the word buffer (2 z); test: then the output and
       the decoded copy (2 z), the word buffer freed by then, and the lookup table */
    if (is_rans(s)) return slack + mb + 2 * tab + rans_tab_cap(s) + 2 * z + (mode == 2 ? lut : 0);
    return slack + mb + tab + (mode == 2 ? 2 * (z + tab) : z);
}
/* bits to code these counts with tables fitted to them (their entropy) */
static double count_bits(const uint32_t *cnt, int nctx, int k) {
    double bits = 0;
    for (int x = 0; x < nctx; x++) {
        const uint32_t *r = cnt + (size_t)x * k; uint64_t N = 0;
        for (int q = 0; q < k; q++) N += r[q];
        for (int q = 0; q < k; q++) if (r[q]) bits += r[q] * log2((double)N / r[q]);
    }
    return bits;
}
static uint64_t held_bytes(const Seg *s) { int t = tab_index(s); return (uint64_t)KQ_CTX[t] * (KQ_SYM[t] * 2 + 4); }
#define MiB(x) ((double)(x) / 1048576.0)
/* Fix the memory budget from what the device has free now, keeping a reserve for the system
   and other apps: 15% of RAM, at least 512 MiB, at most 2 GiB. limit (-m) replaces all this. */
static int plan_memory(Job *j, int threads, uint64_t limit) {
    j->need = xmalloc((j->n + 1) * sizeof *j->need); j->taken = calloc(j->n + 1, 1);
    if (!j->taken) die("out of memory");
    uint64_t mx = 0;                                     /* compress: the larger of both passes, for the message */
    for (int i = 0; i < j->n; i++) {
        j->need[i] = seg_need(&j->seg[i], j->mode, j->phase); if (j->need[i] > mx) mx = j->need[i];
        if (j->mode != 1) { uint64_t n2 = seg_need(&j->seg[i], j->mode, 2); if (n2 > mx) mx = n2; }
    }
    const uint64_t base = 32u << 20;                     /* tenz itself: index, tables, stacks */
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
            j->budget = avail > j->reserve + base ? avail - j->reserve - base : 0; j->live = 1; j->free0 = avail;
            printf("memory: %.0f MiB free, keeping %.0f MiB for the device, tenz may use %.0f MiB\n",
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
        /* The budget already keeps the reserve free. The live reading only lowers it when other apps
           have taken memory since the start: what the device lost beyond what tenz holds now, less
           some slack for page cache and estimates. (It used to require need + reserve free on top of
           the budget, which stalled threads whenever free memory swung by a few MiB.) */
        uint64_t avail = j->live && j->inflight ? mem_avail() : 0, cut = 0;
        if (avail) {
            uint64_t slack = j->budget / 8 + (64u << 20), held = avail + j->used;
            if (j->free0 > held + slack) cut = j->free0 - held - slack;
        }
        int left = 0;
        for (int qi = 0; qi < j->n; qi++) {
            if (j->taken[qi]) continue;
            left = 1; int i = j->order[qi]; uint64_t need = j->need[i];
            int fits = j->inflight == 0 ||                               /* never stall: one always runs */
                       j->used + need + cut <= j->budget;
            if (!fits) continue;
            j->taken[qi] = 1; j->used += need; j->inflight++;
            if (j->used > j->peak) j->peak = j->used;
            if (j->inflight > j->max_inflight) j->max_inflight = j->inflight;
            pthread_mutex_unlock(&j->mu);
            return i;
        }
        if (!left) { pthread_mutex_unlock(&j->mu); return -1; }
        double tw = now();
        pthread_cond_wait(&j->cv, &j->mu);              /* woken when a running segment finishes */
        if (cut) j->wait_dev += now() - tw; else j->wait_mem += now() - tw;   /* segments left, none fits now */
    }
}
/* a static rANS segment: u32 length of the table section, the table section (range coder),
   then the blocks or values (rANS) */
static void static_rans_decode(Job *j, Seg *s, const uint8_t *in, size_t len, uint8_t *out) {
    int t = tab_index(s); uint32_t tl;
    if (len < 4) die("corrupt file");
    memcpy(&tl, in, 4); if (tl > len - 4) die("corrupt file");
    if (!j->dseed[t].f) die("corrupt file");
    int16_t *U = xmalloc((size_t)KQ_CTX[t] * KQ_SYM[t] * 2);
    Coder d; dec_init(&d, in + 4, tl); code_tables(&d, 1, &j->dseed[t], NULL, NULL, U);
    uint8_t *Lt = uses_lut(s) ? lut_build(U, (size_t)KQ_CTX[t], KQ_SYM[t], s->type == T_Q6K ? Q6_LUTB : KQ_SYM[t]) : NULL;   /* every row sums to SM: checked above */
    const uint8_t *body = in + 4 + tl; size_t blen = len - 4 - tl;
    if (is_kq45(s->type)) {
        int bits = s->type == T_Q4K ? 4 : 5;
        kq_lanes_decode(body, blen, out, s->size / (bits == 4 ? 144 : 176), bits, (s->model & 255) == M_QUICK, U, Lt);
    } else if (s->type == T_Q4_0) q40_decode(body, blen, out, s->rows, s->ne0, U);
    else if (s->type == T_Q8_0) q80_decode(body, blen, out, s->rows, s->ne0, U);
    else if (s->type == T_Q6K) {
        if (t == 8) q6l_decode(body, blen, out, s->rows, s->ne0, U, Lt);
        else q6_decode(body, blen, out, s->rows, s->ne0, U, Lt);
    }
    else float_decode(body, blen, s->rows * s->ne0, fl_type(s->type), U, out);
    free(Lt); free(U);
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
        if (is_rans(s)) static_rans_decode(j, s, c->buf, c->pos, buf);
        else {
            Coder d; dec_init(&d, c->buf, c->pos);
            if (is_static(s)) { int t = tab_index(s); U = xmalloc((size_t)KQ_CTX[t] * KQ_SYM[t] * 2);
                                code_tables(&d, 1, &j->dseed[t], NULL, NULL, U); d.stt = U; }
            code_seg(&d, s, buf);
        }
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
        Seg *s = &j->seg[i]; Coder c; double ts = now(), tc = thread_cpu();
        int t = j->phase == 2 || j->mode == 1 ? tab_index(s) : kq_index(s->type);   /* pass 1 picks the lattice */
        if (j->mode != 1) {
            const uint8_t *src = j->in + s->off;
            if (j->phase == 2 && t >= 0) {                   /* static tables, then the values */
                size_t tn = (size_t)KQ_CTX[t] * KQ_SYM[t];
                int16_t *U = xmalloc(tn * 2);
                /* rANS segments: the range coder holds only the table section, far smaller than the
                   segment (bytes past its cap are counted, and the segment is then stored raw) */
                enc_init(&c, is_rans(s) ? rans_tab_cap(s) : s->size); code_tables(&c, 0, &j->seed[t], j->ownf[i], j->ownN[i], U);
                if (is_rans(s)) {                            /* table section, then everything else in rANS */
                    enc_flush(&c);
                    uint8_t *o = xmalloc(s->size + 16); size_t total = s->size;
                    if (c.pos <= c.cap && c.pos + 4 + 4 * FL_LANES + 2 < s->size) {
                        uint32_t tl = (uint32_t)c.pos; memcpy(o, &tl, 4); memcpy(o + 4, c.buf, tl);
                        size_t r = is_kq45(s->type)
                            ? kq_lanes_encode(src, s->size / (s->type == T_Q4K ? 144 : 176), s->type == T_Q4K ? 4 : 5, (s->model & 255) == M_QUICK, U, o + 4 + tl, s->size - 4 - tl)
                            : s->type == T_Q4_0 ? q40_encode(src, s->rows, s->ne0, U, o + 4 + tl, s->size - 4 - tl)
                            : s->type == T_Q8_0 ? q80_encode(src, s->rows, s->ne0, U, o + 4 + tl, s->size - 4 - tl)
                            : s->type == T_Q6K ? (t == 8 ? q6l_encode : q6_encode)(src, s->rows, s->ne0, U, o + 4 + tl, s->size - 4 - tl)
                            : float_encode(src, s->rows * s->ne0, fl_type(s->type), U, o + 4 + tl, s->size - 4 - tl);
                        if (r) total = 4 + tl + r;
                    }
                    free(c.buf); c.buf = o; c.pos = total; c.cap = s->size + 16;
                } else { c.stt = U; code_seg(&c, s, (uint8_t *)src); enc_flush(&c); c.stt = NULL; }
                free(j->ownf[i]); free(j->ownN[i]); j->ownf[i] = NULL; j->ownN[i] = NULL;
                pthread_mutex_lock(&j->mu); j->used -= j->held[i]; pthread_mutex_unlock(&j->mu);
                free(U);
                finish_coded(j, i, s, &c, src);
            } else if (t >= 0) {                             /* pass 1: count (Q6_K may pick the lattice now) */
                size_t tn = (size_t)KQ_CTX[t] * KQ_SYM[t];
                uint32_t *cnt = calloc(tn, 4); if (!cnt) die("out of memory");
                s->model = j->quick && (s->type == T_Q4K || s->type == T_Q5K) ? M_QUICK : M_NORMAL;
                { Coder cc; memset(&cc, 0, sizeof cc); cc.cnt = cnt; cc.cnt_only = 1; code_seg(&cc, s, (uint8_t *)src); }
                if (s->type == T_Q6K && lattice_sample(src, s->size / 210) < 20) {
                    /* the lattice too: the model whose counts have the lower entropy (both static) */
                    size_t tn2 = (size_t)KQ_CTX[8] * KQ_SYM[8];
                    uint32_t *cnt2 = calloc(tn2, 4); if (!cnt2) die("out of memory");
                    q6l_count(cnt2, src, s->rows, s->ne0);
                    if (count_bits(cnt2, KQ_CTX[8], KQ_SYM[8]) < count_bits(cnt, KQ_CTX[t], KQ_SYM[t])) {
                        free(cnt); cnt = cnt2; tn = tn2; t = 8; s->model = M_LATTICE;
                    } else free(cnt2);
                }
                {
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
            if (is_rans(s)) {
                static_rans_decode(j, s, j->in + j->coff[i], j->clen[i], buf);
                pwrite_all(j->fd, buf, s->size, s->off); free(buf);
                goto seg_done;
            }
            dec_init(&c, j->in + j->coff[i], j->clen[i]);
            if (is_static(s)) {
                if (!j->dseed[t].f) die("corrupt file");
                U = xmalloc((size_t)KQ_CTX[t] * KQ_SYM[t] * 2);
                code_tables(&c, 1, &j->dseed[t], NULL, NULL, U); c.stt = U;
            }
            code_seg(&c, s, buf);
            pwrite_all(j->fd, buf, s->size, s->off); free(buf); free(U);
        }
    seg_done:;
        double dt = now() - ts, dc = thread_cpu() - tc;
        pthread_mutex_lock(&j->mu);
        j->used -= j->need[i]; j->inflight--; pthread_cond_broadcast(&j->cv);
        j->tsum[type_index(s->type)] += dt; j->busy += dt; j->cpu += dc;
        if (dt > j->tmax) { j->tmax = dt; j->tmax_seg = i; }
        pthread_mutex_unlock(&j->mu);
        int d = __atomic_add_fetch(&j->done, 1, __ATOMIC_RELAXED);
        fprintf(stderr, "\r  %s%d/%d segments, %.0f s", j->phase == 1 ? "pass 1: " : j->phase == 2 ? "pass 2: " : "",
                d, j->ntodo, now() - j->t0);
    }
    return NULL;
}
/* expected work per byte, relative to Q4_K, so the longest segments start first and none is left
   running alone at the end. Measured on a phone (tenz t, 0.18): byte model 25x, lattice Q6_K 3x
   (before pass 1 decides: Q6_K 2x, as it may try the lattice), Q4_0 1.6x, Q6_K 1.1x, Q5_K 0.85x;
   Q8_0 in rANS lanes estimated 1.8x (2 symbols per byte, Q4_0 about 1.8): re-measure. String
   arrays estimated 14x: about half their bytes go to the byte model, the lengths cost little */
static int SORT_PASS1, SORT_QUICK;                          /* compress pass 1: whether Q6_K tries the lattice is not known yet */
static double seg_cost(const Seg *s) {
    if ((s->model & 255) == M_QUICK || (SORT_QUICK && (s->type == T_Q4K || s->type == T_Q5K))) return (double)s->size * 0.65;
    double w = s->type == T_BYTES ? 25 : s->type == T_STRS ? 14 : s->type == T_Q8_0 ? 1.8 : s->type == T_Q4_0 ? 1.6 : s->type == T_Q5K ? 0.85 :
               s->type != T_Q6K ? 1 : SORT_PASS1 ? 2 : (s->model & 255) == M_LATTICE ? 3 : 1.1;
    return (double)s->size * w;
}
static Seg *SORT_SEGS;
static int cmp_cost(const void *a, const void *b) {
    double x = seg_cost(&SORT_SEGS[*(const int *)a]), y = seg_cost(&SORT_SEGS[*(const int *)b]);
    return x > y ? -1 : x < y;
}
static void run(Job *j, int threads) {
    j->order = xmalloc((j->n + 1) * sizeof(int));
    for (int i = 0; i < j->n; i++) j->order[i] = i;
    SORT_SEGS = j->seg; SORT_PASS1 = j->mode != 1 && j->phase == 1; SORT_QUICK = j->quick; qsort(j->order, j->n, sizeof(int), cmp_cost);
    j->ntodo = 0;
    for (int qi = 0; qi < j->n; qi++) { j->taken[qi] = j->skip && j->skip[j->order[qi]]; j->ntodo += !j->taken[qi]; }
    j->next = 0; j->done = 0; j->t0 = now(); pthread_mutex_init(&j->mu, NULL); pthread_cond_init(&j->cv, NULL);
    double busy0 = j->busy;
    pthread_t th[256]; if (threads > 256) threads = 256; if (threads < 1) threads = 1;
    if (threads > j->n && j->n > 0) threads = j->n;
    for (int i = 0; i < threads; i++) pthread_create(&th[i], NULL, worker, j);
    for (int i = 0; i < threads; i++) pthread_join(th[i], NULL);
    if (j->ntodo) fprintf(stderr, "\n");
    if (j->nruns < 4) { j->pwall[j->nruns] = now() - j->t0; j->pbusy[j->nruns] = j->busy - busy0; j->nruns++; }
    free(j->order);
}
static void time_report(Job *j, double wall, int threads) {
    double tot = 0; for (int k = 0; k < NTYPES; k++) tot += j->tsum[k];
    printf("  time: %.0f s on %d threads, %.0f thread-seconds of work (%.0f%% use)\n   ", wall, threads, tot,
           100.0 * tot / (wall * threads));
    for (int k = 0; k < NTYPES; k++) if (j->tsum[k] > 0) printf(" %s %.1f s", TYPES[k].name, j->tsum[k]);
    /* where the idle time went: each pass's use, the time between passes (one thread builds the seed
       tables), threads waiting for memory, and segment time not on the CPU (reading the input from storage,
   or waiting for a core: other apps, or cores slowed by heat) */
    double pw = 0; printf("\n   ");
    for (int r = 0; r < j->nruns; r++) {
        printf(" pass %d: %.1f s, %.0f%% use;", r + 1, j->pwall[r], j->pwall[r] > 0 ? 100.0 * j->pbusy[r] / (j->pwall[r] * threads) : 0);
        pw += j->pwall[r];
    }
    printf(" between and around passes: %.1f s\n", wall - pw > 0 ? wall - pw : 0);
    printf("    waiting for memory: %.1f thread-seconds with the budget full, %.1f with other apps holding memory\n"
           "    segments off the CPU (reading the input, or no free core): %.1f thread-seconds",
           j->wait_mem, j->wait_dev, j->cpu > 0 && j->busy > j->cpu ? j->busy - j->cpu : 0);
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
    int nraw = 0, nlat = 0, n6 = 0, nkq = 0, nqk = 0, rawn[NTYPES] = {0}; uint64_t rawb[NTYPES] = {0}, rawmax[NTYPES] = {0};
    for (int i = 0; i < n; i++) {
        int k = type_index(s[i].type); ob[k] += s[i].size; cb[k] += clen[i]; vals[k] += s[i].rows * s[i].ne0;
        if (s[i].model == M_RAW) { nraw++; rawn[k]++; rawb[k] += s[i].size; if (s[i].size > rawmax[k]) rawmax[k] = s[i].size; }
        if (s[i].type == T_Q6K) { n6++; nlat += (s[i].model & 255) == M_LATTICE; }
        if (s[i].type == T_Q4K || s[i].type == T_Q5K) { nkq++; nqk += (s[i].model & 255) == M_QUICK; }
    }
    for (int k = 0; k < NTYPES; k++) if (ob[k]) {
        uint32_t id = TYPES[k].id;
        const char *unit = id == T_BYTES ? "byte (metadata etc.)" : id == T_STRS ? "byte (metadata strings)" :
                           id == T_M32 ? "value (metadata arrays)" : "weight incl. headers";
        printf("  %-5s %9.1f MiB -> %9.1f MiB  (%5.2f%% saved, %.3f bits/%s)\n", TYPES[k].name,
               ob[k] / 1048576.0, cb[k] / 1048576.0, 100.0 * (1 - (double)cb[k] / ob[k]), cb[k] * 8.0 / vals[k], unit);
    }

    if (n6) printf("  Q6_K segments using the AWQ lattice: %d of %d\n", nlat, n6);
    if (nqk) printf("  Q4_K/Q5_K segments in the quick model (-q): %d of %d\n", nqk, nkq);
    if (nraw) {
        printf("  segments stored raw (coding would not shrink them): %d\n", nraw);
        for (int k = 0; k < NTYPES; k++) if (rawn[k])
            printf("    %-5s %d, %llu bytes in all, largest %llu bytes\n", TYPES[k].name, rawn[k],
                   (unsigned long long)rawb[k], (unsigned long long)rawmax[k]);
    }
    for (int t = 0; t <= NGGML; t++) if (g_unmodelled[t])
        printf("  stored as they are (no model for the type): %.*f MiB of %s\n", MiB(g_unmodelled[t]) < 0.1 ? 3 : 1, MiB(g_unmodelled[t]), ggml_name(t));
    printf("  file  %9.1f MiB -> %9.1f MiB  (%5.2f%% saved)\n", total / 1048576.0, out_total / 1048576.0,
           100.0 * (1 - (double)out_total / total));
}
static void cmd_compress(const char *in, const char *out, int threads, uint64_t segb, int test, uint64_t limit, int hsym, int quick) {
    size_t n; const uint8_t *p = map_file(in, &n);
    /* -s not given: segments of n / 24, from 8 to 48 MiB (48 from about 1.1 GB on), so a small file has enough of them to keep
       every thread busy to the end; by file size only, so a file splits the same way everywhere */
    if (!segb) { segb = (uint64_t)n / 24 >> 20 << 20; if (segb < (8u << 20)) segb = 8u << 20; if (segb > (48u << 20)) segb = 48u << 20; }
    int ns; Seg *s = find_segments(p, n, segb, &ns);
    printf("%s: %.1f MiB, %d segments to code\n", in, n / 1048576.0, ns);
    for (int t = 0; t <= NGGML; t++) if (g_unmodelled[t])
        printf("note: %.*f MiB of %s tensors, a type tenz has no model for: stored as they are\n", MiB(g_unmodelled[t]) < 0.1 ? 3 : 1, MiB(g_unmodelled[t]), ggml_name(t));
    Job j; memset(&j, 0, sizeof j);
    j.mode = test ? 2 : 0; j.seg = s; j.n = ns; j.in = p; j.fd = -1; j.phase = 1; j.quick = quick;
    if (quick) printf("quick (-q): Q4_K and Q5_K in the faster model, at a small cost in size\n");
    if (hsym) printf("-H: no longer needed, every block header is coded as symbols (ignored)\n");
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
    for (int i = 0; i < ns; i++) j.skip[i] = kq_index(s[i].type) < 0;   /* metadata waits for pass 2 */
    double t0 = now(); run(&j, threads);                 /* pass 1: counts (Q6_K: lattice or not) */
    /* seeds: one table set per type, from the counts of all its static segments */
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
    /* pass 2: static segments and metadata */
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
    if (n < HDR) die("not a tenz file");
    if (memcmp(p, MAGIC, 8)) {
        if (!memcmp(p, "TENZ", 4) && p[6] == '\n')
            fprintf(stderr, "tenz: file format %d, but tenz " TENZ_VERSION " reads only format " TENZ_FORMAT ": "
                            "use the tenz version that made it\n", atoi((const char *)p + 4)), exit(1);
        if (!memcmp(p, "FOLDZ", 5) && p[7] == '\n') {         /* the predecessor: released 0.9 to 0.15 */
            int f = atoi((const char *)p + 5);
            if (f >= 9 && f <= 15)
                fprintf(stderr, "tenz: this is a foldz file (format %d): decompress it with foldz 0.%d, "
                                "the predecessor of tenz\n", f, f), exit(1);
            die("this file is from an unreleased foldz build: compress the original again with tenz");
        }
        die("not a tenz file");
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
        {   uint32_t md = s[i].model & 255;
            if (md != M_NORMAL && md != M_RAW && !(md == M_LATTICE && s[i].type == T_Q6K) &&
                !(md == M_QUICK && (s[i].type == T_Q4K || s[i].type == T_Q5K))) die("corrupt header");
            if (s[i].model >> 8) die("corrupt header"); }
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

/* output name: given, or derived from the input (in.gguf -> in.gguf.tenz, in.gguf.tenz -> in.gguf) */
static char *out_name(const char *in, const char *given, int decompress) {
    if (given) return strdup(given);
    size_t n = strlen(in), k = strlen(".tenz");
    char *o = xmalloc(n + 8);
    if (decompress) {
        if (n > k && !strcmp(in + n - k, ".tenz")) { memcpy(o, in, n - k); o[n - k] = 0; }
        else { memcpy(o, in, n); memcpy(o + n, ".out", 5); }
    } else { memcpy(o, in, n); memcpy(o + n, ".tenz", k + 1); }
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
        fprintf(stderr, "tenz: %s already exists (use -f to overwrite)\n", out);
        exit(1);
    }
}

int main(int argc, char **argv) {
    init_tables();
    init_stretch();
    if (argc == 2 && (argv[1][0] == 'v' ||
                      !strcmp(argv[1], "--version"))) {
        printf("tenz " TENZ_VERSION
               " (file format " TENZ_FORMAT ")\n");
        return 0;
    }
    if (argc < 3) {
        fprintf(stderr,
            "usage: tenz c in [out.tenz] [-t threads]"
            " [-s segment_MiB] [-m MiB] [-q] [-f]\n"
            "       tenz d in.tenz [out] [-t threads]"
            " [-m MiB] [-f]\n"
            "       tenz t in [-t threads] [-s segment_MiB]"
            " [-m MiB] [-q]   (round-trip test, no files)\n"
            "       tenz v   (version)\n"
            "  out defaults to in + .tenz (c) or in without"
            " .tenz (d); -f overwrites an existing file\n"
            "  -q: quick, Q4_K and Q5_K in a faster model"
            " (about 35%% faster for them, a little larger)\n"
            "  defaults: all cores, segments of 1/24 of the"
            " file (8 to 48 MiB); memory: what is free,\n"
            "  minus a reserve for the device (15%% of RAM,"
            " 512 MiB to 2 GiB); -m sets a limit instead\n");
        return 1;
    }
    int threads = (int)sysconf(_SC_NPROCESSORS_ONLN);
    int force = 0, hsym = 0, quick = 0, np = 0;
    uint64_t segb = 0, limit = 0;      /* segb 0: by file size */
    const char *pos[2] = {NULL, NULL};
    for (int i = 2; i < argc; i++) {   /* options anywhere */
        const char *a = argv[i];
        if (!strcmp(a, "-t") || !strcmp(a, "-s") ||
            !strcmp(a, "-m")) {
            if (i + 1 >= argc) die("option needs a value");
            long long v = atoll(argv[++i]);
            char o = a[1];
            if (o == 't') {
                threads = (int)v;
            } else if (o == 's') {
                segb = (uint64_t)(v < 1 ? 1 : v);
            } else {
                if (v < 64) die("-m needs at least 64 (MiB)");
                limit = (uint64_t)v << 20;
            }
        } else if (!strcmp(a, "-f")) {
            force = 1;
        } else if (!strcmp(a, "-H")) {           /* old scripts: accepted and ignored */
            hsym = 1;
        } else if (!strcmp(a, "-q")) {
            quick = 1;
        } else if (a[0] == '-' && a[1]) {
            fprintf(stderr, "tenz: unknown option %s\n", a);
            return 1;
        } else if (np < 2) {
            pos[np++] = a;
        } else {
            die("too many file names");
        }
    }
    if (!pos[0]) die("missing input file");
    segb <<= 20;
    char cmd = argv[1][0];
    int one = argv[1][1] == 0;
    if ((cmd == 'c' || cmd == 'd') && one) {
        char *out = out_name(pos[0], pos[1], cmd == 'd');
        check_output(pos[0], out, force);
        if (cmd == 'c')
            cmd_compress(pos[0], out, threads, segb, 0,
                         limit, hsym, quick);
        else
            { if (quick) printf("-q: only for c and t, the decoder reads it from the file (ignored)\n");
              cmd_decompress(pos[0], out, threads, limit); }
        free(out);
    } else if (cmd == 't' && one) {
        if (pos[1]) die("t takes no output name");
        cmd_compress(pos[0], NULL, threads, segb, 1,
                     limit, hsym, quick);
    } else {
        die("unknown command (use c, d, t or v)");
    }
    return 0;
}