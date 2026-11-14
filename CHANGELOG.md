# Changelog

The version and the file format share one number: tenz 0.23 writes and reads format 23 (magic `TENZ23`). Until 1.0, a `.tenz` file only decompresses with the version that made it. Versions up to 0.15 were released as foldz; a `.foldz` file needs the foldz version that made it.

Each section becomes the release notes when its tag (for example `v0.23`) is pushed; see `.github/workflows/release.yml`.

## [Unreleased]

## [0.23] - 2026-11-14
- New option `-q` (quick): faster models for Q4_K, Q5_K, Q4_0, Q8_0 and floats, a fraction of a point less saved (Qwen3.5-2B Q4_K_M 8.54% → 8.18%, Llama-3.2-1B Q8_0 8.27% → 7.82%, BF16 34.25% → 34.19%, Gemma-4-31B Q4_0 12.70% → 12.19%). Recorded per segment.
- Branch-free rANS refill in every decoder: Q4_K/Q5_K decoding about 20% faster, Q8_0 about 14%, output unchanged.

## [0.22] - 2026-11-14
- Q6_K column statistics in float instead of double, branch-free: Q6_K about twice as fast (x86), same size.
- Q5_K and Q6_K symbols found by vector search instead of a lookup table: decompression faster than compression again (Qwen3.5-2B: 6.8 s on the phone).

## [0.21]
- Q8_0 in rANS lanes from static tables, its scale given the previous block's scale.

## [0.20] - 2026-11-13
- Q6_K lattice: the step also given its column (sign share, average size); richer scale contexts. Gemma-4-31B Q4_0: 14694.9 MiB (12.70% saved).
- Segment size follows the file size (1/24 of the file, 8 to 48 MiB), so small files keep every thread busy.

## [0.19] - 2026-11-13
- The Q6_K lattice model in static tables and rANS lanes (3.7× faster decoding); pass 1 picks the lattice by counts. `-H` accepted and ignored.

## [0.18] - 2026-11-13
- Normal Q6_K in rANS lanes (Q6_K 2.2× faster on the phone); better scheduling and memory estimates; the report shows where idle time goes.

## [0.17] - 2026-11-13
- Renamed foldz to tenz. Q4_K/Q5_K whole blocks in rANS lanes; Q4_0 support; tensors of unsupported types stored as they are.

## [0.15] - 2026-11-12
- F32, F16 and BF16 in rANS lanes: about 1.6× faster decoding.

## [0.14.1] - 2026-11-12
- Q6_K context bases computed ahead and prefetched; output unchanged.

## [0.14] - 2026-11-12
- Floats from static tables (about 2× faster); metadata in parallel pieces; `bench.sh`.

## [0.13.2] - 2026-11-12
- The report lists raw segments.

## [0.13.1] - 2026-11-12
- Safetensors support (F32, F16, BF16).

## [0.13] - 2026-11-12
- Q8_0 as two 4-bit symbols from static tables (Q8_0 work halved); better scheduling.

## [0.12] - 2026-11-12
- K-quant values from static tables (a seed per file, differences per segment): smaller files, faster decompression.

## [0.11]
- `-H`: block headers as symbols; AWQ lattice values as symbols.

## [0.10]
- Quantized values coded as whole symbols: about 1.5–2× faster.

## [0.9.1]
- Memory-aware scheduling and `-m`; about 23% faster with identical output.

## [0.9]
- First public pre-release.
