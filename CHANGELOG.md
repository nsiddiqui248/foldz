# Changelog

Changes per tenz version. The version and the file format share one number: tenz 0.17 writes and reads format 17 (magic `TENZ17`). Until 1.0, a `.tenz` file only decompresses with the version that made it.

tenz is the successor of foldz, and the numbers continue: versions up to 0.15 were released as foldz (0.16 was never released). A `.foldz` file needs the foldz version that made it.

Each section below becomes the release notes when its tag (for example `v0.17`) is pushed; see `.github/workflows/release.yml`.

## [Unreleased]

## [0.17] - 2026-11-13

File format 17. First version named tenz, pre-release.

### Renamed
- foldz is now **tenz**: the command, the `.tenz` extension, the source file `tenz.c`, the magic `TENZ17` and the build switches `TENZ_SCALAR` and `TENZ_NO_PREFETCH`. The name says what it compresses (tensors), not how, so it stays when the coding changes.
- Version and format numbers continue from foldz, so the release tags too. tenz recognises `.foldz` files and names the foldz version that reads them.

### Changed
Since foldz 0.15 (developed as foldz 0.16, never released):

- Q4_K and Q5_K blocks are coded with rANS in 4 interleaved lanes from static tables, like the floats in foldz 0.15; only the components differ. Nothing of these types goes through the range coder any more except the table section.
  - Values: same contexts and tables as before. Lane L takes sub-blocks L and 4 + L of a block, since the contexts restart at every sub-block, so the decoder works on 4 sub-blocks at once.
  - Headers: block b belongs to lane b % 4, with the lane's previous block as history. d and dmin are four nibbles each, with the float contexts; scales and mins are a high 2-bit and a low 4-bit symbol (scale given the previous scale; min given its scale and the previous min). They were adaptive before (bit by bit, or symbols with `-H`); now they have static tables in the same seed as the values.
- `-H` now only affects Q6_K: Q4_K and Q5_K headers are always symbols.
- **Q4_0 support.** Whole blocks from static tables in 4 rANS lanes, block b in lane b % 4. The scale d is four nibbles with the float contexts. Values are 4-bit symbols given the column's share of positive weights so far (as Q8_0, flipped when d < 0), whether q = 0 (the block's extreme) was seen, the average |q − 8| so far and the position. The column counts are those at the start of each group of 4 blocks, so the lanes stay independent. Before, Q4_0 went to the byte model: a 16.8 GB Gemma Q4_0 model ran for over 50 minutes and was a third done.
- **GGUF tensors of types without a model are stored as they are** instead of going to the byte model, which gains a few percent at most on quantized weights and takes hours on a large model. tenz names them and their size at the start and in the report. Each one runs from its offset to the next tensor's, so no table of block sizes is needed.
- The memory estimate for Q4_K and Q5_K segments includes the rANS buffers.

### Results
- Values in lanes alone (headers still in the range coder), Qwen3.5-2B, phone, one run each: Q4_K decoding 33.1 → 25.5 thread-seconds (−23%), Q5_K 8.4 → 6.6 (−21%), same size (1109.8 MiB). Wall time unchanged (12.8 vs 13.0 s): Q6_K is now the longest part.
- Whole blocks in lanes, Qwen3.5-2B, phone, `t` against foldz 0.15: Q4_K 489.6 → 488.6 MiB, Q5_K 138.1 → 137.9 MiB, file 1109.8 → 1108.7 MiB (8.42% → 8.52% saved); the static header tables code better than the adaptive model they replace. Q4_K work 51.4 → 40.5 thread-seconds (−21%), Q5_K 13.5 → 10.1 (−25%), total 189 → 176, round trip 23.7 → 22.1 s. Q6_K is now 104 of the 176 thread-seconds.
- BF16 unchanged (169.2 MiB on a 256.6 MiB safetensors model).
- Q4_0, synthetic test data (llama.cpp's quantizer on Student-t weights with per-column bias and scale), x86, one thread: 18.5% saved, decoding at 34 MiB/s (Q4_K: 47 MiB/s on similar data). The column context is about 15% of the decoding time and 0.85 points of the saving here.
- Gemma 4 31B Q4_0 (QAT release, 16833.3 MiB), phone, `tenz t`: 14704.2 MiB (12.65% saved), round trip 328 s. Q4_0 15710.6 → 13973.6 MiB (11.06% saved, 4.002 bits/weight); Q6_K 34.13% saved, with the AWQ lattice chosen for all 23 segments. With the byte model (before), the same file was a third done after 55 minutes.

## [0.15] - 2026-11-12

File format 15.

### Changed
- F32, F16 and BF16 values are coded with rANS in 4 interleaved lanes instead of the range coder: decoding a symbol is a mask, a table search, a multiply and an add, with no division, and the 4 lanes are independent chains the CPU runs at once. Same tables and seeds as 0.14; each lane keeps its own history, which made no measurable difference in size.
- Test files, one thread: BF16 decodes 1.56× faster than 0.14.1, F32 converted from BF16 1.73× faster; sizes within 0.01% (a few bytes per segment for the lane states).

## [0.14.1] - 2026-11-12

File format 14, unchanged: output is byte-identical to 0.14.

### Changed
- Q6_K: the context bases of a block (the column statistics, the expensive floating-point part) are computed before its values are coded, and their table rows prefetched. About 10% faster Q6_K decoding on x86. On an Android phone about the same speed as 0.14: there the prefetch makes Q6_K about 3% faster than computing the contexts first without it, which roughly cancels out. Build with `-DFOLDZ_NO_PREFETCH` to compare; the output is identical.
- `bench.sh`: xz only with `XZ=1`; usage message; on Android the machine line names the device and chip.

### Results
- Qwen3.5-2B with 0.14, phone: compress 11.9 s, decompress 12.6 s (about 100 MiB/s each way). BF16 safetensors (256.6 MiB): compress 0.96 s, decompress 2.1 s.

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
