# Changelog

Changes per foldz version. The version and the file format share one number: foldz 0.12 writes and reads format 12. Until 1.0, a `.foldz` file only decompresses with the version that made it.

Each section below becomes the release notes when its tag (for example `v0.12`) is pushed; see `.github/workflows/release.yml`.

## [Unreleased]

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
