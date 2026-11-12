# Changelog

Changes per foldz version. The version and the file format share one number: foldz 0.14 writes and reads format 14. Until 1.0, a `.foldz` file only decompresses with the version that made it.

Each section below becomes the release notes when its tag (for example `v0.12`) is pushed; see `.github/workflows/release.yml`.

## [Unreleased]

## [0.14] - 2026-11-12

File format 14.

### Changed
- F32, F16 and BF16 values are coded as 4-bit symbols from static seed tables, like the quantized types, instead of bit by bit with adaptive probabilities. Each 16 bits are four nibbles: sign, exponent and top mantissa bits each given the ones above, the low mantissa bits with little context. Test files: about 2× faster to decompress, 0.1–1.1% smaller (BF16 −0.4%, F32 converted from BF16 −1.1%).
- A segment can use the seed tables as they are, with no table section of its own, when that is cheaper: helps small segments.
- Model metadata (tokenizer data) is split into 2 MiB pieces coded in parallel, instead of one segment on one thread. About 3% larger metadata (about 0.06 MiB on Qwen3.5-2B); helps small models and machines with many cores, where that one segment was the longest part of the run.

### Added
- `bench.sh`: compare foldz with zstd and xz on any machine, as a markdown table.
- The `bench` workflow: runs `bench.sh` on a GitHub x86 runner with public models.

### Removed
- The per-segment list of raw segments added in 0.13.2; the report shows their count again. On real models they are tiny F32 tensors (16 values) that no coder can shrink.

### Results
- Safetensors, one model: BF16 169.7 → 169.2 MiB and 7.1 → 2.8 s round trip; F32 converted from BF16 167.6 → 165.8 MiB and 8.4 → 3.6 s.
- Qwen3.5-2B: 1109.8 → 1109.9 MiB (+0.1 MiB from the metadata pieces and tables for tiny F32 tensors), 25.3 → 23.5 s round trip.

## [0.13.2] - 2026-11-12

File format 13, unchanged.

### Added
- The report lists segments stored raw (type, shape, size, offset, and how large coding made them), to find what foldz fails to shrink.

### Investigated, not adopted
- Folding 1, 2 and 3 times: no effect on Q4_K/Q5_K values (0–1 byte), 4–10% larger on floats, and foldz's compressed output stays incompressible for xz however often it is folded.

## [0.13.1] - 2026-11-12

File format 13, unchanged: 0.13 decompresses these files.

### Added
- Safetensors support: the compressor reads the JSON header and codes F32, F16 and BF16 tensors with the float models instead of the byte model. Test files: 17–20× faster to compress, 1.5–8.5% smaller. A header that does not parse falls back to the byte model.

### Fixed
- A file with no supported tensors passed an empty list to `qsort` with a null pointer (undefined behaviour, harmless in practice).

### Investigated, not adopted
- The fold on whole float values (BF16, F16, F32): 4–19% larger, because it mixes the sign, exponent and mantissa fields the model keeps apart.

## [0.13] - 2026-11-12

File format 13.

### Changed
- Q8_0 values are coded as two 4-bit symbols (high nibble, then low nibble given the high one) from static seed tables, like the K-quants, instead of 8 bit-by-bit steps.
- Compression schedules better: pass 1 only counts the types with static tables, and everything else (metadata, floats) is coded in pass 2 alongside them. Same output, less idle time.

### Results
- Qwen3-0.6B-Q8_0: 565.4 → 563.9 MiB (Q8_0 6.56% → 6.84% saved), Q8_0 work halved (131 → 61 thread-seconds), round trip 17.8 → 11.2 s.
- Qwen3.5-2B (Q4_K_M, no Q8_0): same size (1109.8 MiB), round trip 33.4 → 25.3 s from the scheduling change.

### Investigated, not adopted
- The fold (Gray code) on every part of Q4_K_M and Q8_0 tensors, also per 4-bit group: no measurable effect, because both the bit trees and the symbol tables learn any relabelling of the values equally well.

## [0.12] - 2026-11-12

File format 12.

### Changed
- Q4_K, Q5_K and Q6_K values are coded from static tables instead of adaptive counts. The compressor counts the whole file first, stores one seed table per tensor type, and each segment stores, per context, either "use the seed" or its own table as differences from the seed.
- Decompression is now faster than compression: about 1.25× faster than 0.11 on the K-quants, measured on an Android phone.
- Compression makes two passes (count, then code), so it is slower than 0.11.

### Results
- Smaller files on every model measured: Qwen3.5-2B 1109.8 MiB (−1.4 MiB), Qwen3-4B 2070.0 MiB (−0.6 MiB), Qwen2.5-7B-Instruct 4111.0 MiB (−2.3 MiB).

### Investigated, not adopted
- Column statistics as context for Q4_K and Q5_K (as Q6_K uses): about 0.1% of the file, at a speed cost.
- Decoding inside llama.cpp at inference time: block-local contexts keep about 60–90% of the gain, but at most 4–5% fewer bytes to read.

## [0.11]

File format 11.

### Added
- `-H`: code block headers (scales, mins, d) as symbols. Faster to decompress, slightly larger; recorded per segment, so `foldz d` needs no option.

### Changed
- AWQ lattice values are coded as symbols: smaller and faster. Qwen3-4B: 2070.6 MiB (−2.6 MiB).

## [0.10]

File format 10.

### Changed
- Quantized values are coded as whole symbols (one coder step per value) instead of bit by bit, with counts halved at 2^14. About 1.5–2× faster round trips at the same size or slightly smaller.

### Removed
- The bit-wise value coder (Gray-code folds).

## [0.9.1]

File format 9, unchanged: files are byte-identical to 0.9.

### Added
- Memory-aware scheduling: segments start only while their estimated memory fits in what the device has free, keeping a reserve for the system (15% of RAM, 512 MiB to 2 GiB); checked again before every segment.
- `-m MiB`: a fixed memory limit instead of the automatic one.

### Changed
- Faster coding with identical output: separate encoder and decoder copies of each model, a branchless probability update, no `frexp` call per Q6_K weight. About 23% faster round trip on a phone.
- Raw segments are written straight from the input, without a copy.

## [0.9]

File format 9. First public pre-release.
