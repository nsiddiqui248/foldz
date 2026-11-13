# Changelog

Changes per tenz version. The version and the file format share one number: tenz 0.17 writes and reads format 17 (magic `TENZ17`). Until 1.0, a `.tenz` file only decompresses with the version that made it.

tenz is the successor of foldz, and the numbers continue: versions up to 0.15 were released as foldz (0.16 was never released). A `.foldz` file needs the foldz version that made it.

Each section below becomes the release notes when its tag (for example `v0.17`) is pushed; see `.github/workflows/release.yml`.

## [Unreleased]

## [0.20] - 2026-11-13

File format 20.

### Changed
- **Q6_K scales: the low nibble is given the high nibble and the whole previous scale**, not just the previous scale's high nibble. Both Q6_K models. Scale bits −9% on AWQ-style test data.
- **Lattice: lanes over consecutive sub-blocks.** Lane L takes sub-blocks 4L … 4L+3, so a lane's previous lattice size is usually the previous sub-block's. The normal model keeps L, L+4, L+8, L+12; its only within-block context restarts every sub-block, so the layout doesn't matter there.
- **Segment size follows the file size** when `-s` isn't given: 1/24 of the file, from 8 to 48 MiB, so files above about 1.1 GB keep 48 MiB. A small file's largest segment no longer runs alone at the end (a 256.6 MiB BF16 model ran at 75% thread use, its 48 MiB segment taking 2.1 s of 2.8 s). It depends on the file size only, so a file splits the same way on every machine.
- **Lattice: the step k is also given its column**, over the rows above:
  - the share of positive k, as Q4_0's 8 buckets (k = 0 is a zero weight, so this is the weight's sign bias);
  - the average |k| / m, in 4 buckets.
  Per row, the buckets are a reciprocal and three thresholds (⌊sum/r⌋ > 10 ⇔ sum ≥ 11r), and |k|·32/m comes from a table, so decoding does no division. k has 66 × 32 context rows, up from 66.

### Results
- AWQ-style test data (4-bit weights with a scale and zero per 128, then llama.cpp's Q6_K quantizer; 19 MiB): 0.18's adaptive lattice 3.676 bits/weight, 0.19 3.701, 0.20 **3.612** (1.7% below 0.18, 2.4% below 0.19). The two column contexts give most of it, the scale context about 1%. Lattice decoding about 15% slower than 0.19 (0.47 → 0.57 s).
- Normal Q6_K test data: unchanged (its synthetic scales are random; real scales may gain).
- Q4_0, tried and not kept: a start table for the symbol search (10% slower), 8 rANS lanes (6% slower on x86, which runs out of registers; ARM has twice as many and might differ), division-free encoding by reciprocal (exact, but no faster). Q4_0 is unchanged from 0.19, which already took 22% less time per GiB round trip than 0.18 (63 → 49 s per GiB, x86, one thread).
- Phone, `tenz t`: Llama-3.2-1B-Instruct BF16 safetensors 2357.1 → 1549.8 MiB (34.25% saved), 21 s, 100% thread use. Qwen3.5-2B 1108.4 → 1108.5 MiB: the scale context gains nothing on its normal Q6_K, and its larger tables cost a little per segment. Synthetic 254.5 MiB BF16 safetensors with the new segment size: slowest segment 6.9 → 1.7 s, same size.
- Gemma 4 31B Q4_0, phone, `tenz t`: Q6_K (all lattice) 729.1 MiB (0.19) → **718.5 MiB**, also 1.1% below 0.18's adaptive lattice (726.3); file 14705.4 → **14694.9 MiB (12.70% saved)**, the smallest yet. Round trip 258 s (225 s for 0.19 on a cooler run): Q6_K work 222 → 262 thread-seconds, from the slower lattice decoding and the richer contexts in counting; Q4_0 1546 → 1761 thread-seconds with unchanged code, which is heat.
- The 256.6 MiB BF16 model with the new segment size: 10 MiB segments, slowest 2.1 → 0.5 s, thread use 75% → 99%, round trip 2.8 → 2.1 s.

## [0.19] - 2026-11-13

File format 19.

### Changed
- **The Q6_K lattice model in static tables and rANS lanes.** It was the last adaptive model on the range coder, and Gemma's slowest segments. Now it has its own static table set (table index 8), the normal model's headers, lanes and chunks, and these symbols:
  - per 16-value sub-block, in its lane: the lattice size m given the lane's previous m, then the span given m / 4;
  - per value: k + 32 given 2m + (k = −m seen), then the error e + 3 in 0..6 given m, or 7 and the value itself (escape).
  The number of symbols per block varies with the escapes, so the encoder records each chunk's symbols in the decoder's order (lane, row, symbol) and codes them backwards. Chunks are 512 blocks.
- **Pass 1 chooses the lattice by counts.** It counts both models and keeps the one whose counts have the lower entropy, both now being static. Before, the lattice was coded adaptively in pass 1 and finished there.
- **Q4_0 context in fewer operations.** The column's share of positive weights is one multiplication by a per-row reciprocal (in 1/32 steps, mapped to the same 8 buckets by their midpoints), instead of 7 threshold compares. The running average is one table load by (position, sum so far), instead of a multiply and a clamp.
- **`-H` is accepted and ignored**, since every block header is now coded as symbols. Files carry no header flag any more.
- **Faster lattice fit, same result.** Its cost estimate has a floor that rises with the lattice size (16·log₂(2m+1) bits), so the search stops once that floor reaches the best cost found. Output is identical; compressing noisy lattice data took 23% less time.
- **Removed:** the adaptive symbol coder and the adaptive fp16 coder, now unused.

### Results
- Synthetic lattice data (9.6 MiB of Q6_K on a 15-level grid), x86, one thread, lattice forced in both versions: decoding 0.75 → 0.2 s (3.7× faster), 38.33% → 38.42% saved. Compression 0.8 → 1.1 s: pass 1 counts both models, and pass 2 fits the lattice again.
- Synthetic Q4_0, decoding by CPU time, best of 7 runs: 37.6–38.5 → 40.6–45.0 MiB/s, same size (18.5% saved).
- Gemma 4 31B Q4_0, phone, `tenz t`, against 0.18: round trip 341 → 267 s (−22%), pass 1 84 → 49 s (the lattice is no longer coded adaptively there), slowest segment 12.6 → 8.3 s, thread use 100%. Q4_0 work 2421 → 1838 thread-seconds, though heat varies between runs (1723 with 0.17). Q6_K work barely changed (261 → 250): decoding is faster, but the encoder fitted the lattice in both passes (before the faster fit above). Size: Q6_K 726.3 → 729.1 MiB, as the static lattice codes 0.4% worse than the adaptive one; Q4_0 13973.6 → 13972.0 MiB; file 14704.2 → 14705.4 MiB.
- With the faster lattice fit: Gemma round trip 225 s (pass 1 40 s, Q6_K 221 thread-seconds, slowest segment 6.6 s); Qwen3.5-2B 17.0 s. Over the 0.18 → 0.19 work, Gemma went 341 → 225 s.
- Qwen3.5-2B: 17.4 s, 100% thread use, same size.

## [0.18] - 2026-11-13

File format 18.

### Changed
- **Q6_K in rANS lanes.** The normal Q6_K model (not the lattice) codes whole blocks from static tables in 4 rANS lanes, like Q4_K:
  - Headers: block b's header goes in lane b % 4, with the lane's previous block as history. d is four nibbles with the float contexts. Each scale (int8 + 128) is a high nibble given the previous scale, then a low nibble given the high one and the previous scale's high nibble. Before, they were adaptive, bit by bit.
  - Values: same contexts as before (column offset and width, position, values ≤ 1 seen). Lane L takes sub-blocks L, 4 + L, 8 + L and 12 + L, since the only context inside a block restarts at every 16-value sub-block.
  - Chunks: the column sums are doubles, and rANS encodes in reverse, so the encoder cannot run them backwards exactly (Q4_0's integer counts can be). Each segment is cut into streams of 2048 blocks (u32 word count, 4 lane states, words). The encoder computes a chunk's contexts forward, then codes it backwards. The cost is 20 bytes per chunk.
  - Context arithmetic: the divisions by the row count and by the step became multiplications by their reciprocals, computed once per block and sub-block, and the rounding no longer calls `rint`. All still exact IEEE operations, so files stay identical on every platform.
- **Pass 1 codes Q6_K at most once.** The normal model's size is estimated from its counts (their entropy) instead of coding it adaptively. Only the lattice is coded, and only when a sample suggests it.
- **`-H` now only affects lattice Q6_K segments**, the last ones with adaptive headers.
- **Scheduling weights brought up to date.** Segments start longest first, by size times a per-type weight relative to Q4_K. The old weights (byte model 12×, Q6_K 2×) made a 2 MiB metadata piece look shorter than a 48 MiB tensor segment, though it takes about twice as long; such a piece then ran alone at the end. New weights from the measurements below: byte model 25×, lattice Q6_K 3×, Q8_0 2× (estimated), Q4_0 1.6×, Q6_K 1.1× (2× in pass 1, which may try the lattice), Q5_K 0.85×.
- **The report lists raw segments by type**, with their count, total size and largest size.
- **Live memory check no longer stalls on small swings.** The budget already keeps the device's reserve free, but each new segment also required its need plus the full reserve to be free right then. Once tenz held most of its budget, free memory sat at that line, and every swing of a few MiB stalled threads. Now the live reading only lowers the budget when other apps have taken memory since the start: by what the device lost beyond what tenz holds, less a slack of 1/8 of the budget + 64 MiB.
- **rANS segments need half the memory, and the estimate says so.** The range coder that writes a rANS segment's table section had a buffer as large as the segment, though the tables need a small fraction of it; it is now sized to the tables. The estimate counted the output twice: a 48 MiB Q4_0 segment was estimated at 193 MiB in `tenz t` (it held 144 MiB), so an 872 MiB budget ran only 4 segments at once. Now 97 MiB, 8 at once. The startup message shows the larger need of the two passes; it showed only pass 1's.
- **The report shows where idle time goes**: each pass's wall time and thread use, the time between and around the passes (one thread builds the seed tables), thread-seconds spent waiting for memory, and segment time spent off the CPU (reading the input from storage, or waiting for a core).

### Results
- Synthetic Q6_K (llama.cpp's quantizer on Student-t weights with per-column bias and scale, 32 MiB), x86, one thread: decoding 1.9 → 1.05 s (1.8× faster), compression 1.8 → 1.2 s, size 11.50% → 11.53% saved. Without the column context, decoding would take 0.85 s.
- On grid-shaped test data (AWQ-like), the static normal model now beats the adaptive lattice (39.7% against 38.1% saved), so tenz picks it there.
- Qwen3.5-2B, phone, `tenz t` against 0.17: Q6_K work 104.3 → 47.5 thread-seconds (2.2× faster), total 176 → 123, round trip 22.1 → 17.2 s, file 1108.7 → 1108.4 MiB. Thread use was 89%: the slowest segment was a 2 MiB metadata piece (4.6 s) that started late, fixed by the new scheduling weights above.
- With the new weights: Qwen3.5-2B thread use 89% → 93%. The raw segments are tiny F32 tensors: Qwen3.5-2B 18 of 64 bytes, Gemma 60 of 4 bytes (one number per layer), too small to carry the coded form's fixed costs. Gemma's thread use went 94% → 89% → 78% over three runs as its budget fell (1247, 1220, 893 MiB). The new report showed why: pass 1 at 100%, pass 2 at 71%, with 659 thread-seconds waiting for memory. The live-check fix above did not change it (713 thread-seconds on the next run): the budget itself was full, because of the overestimated segment memory, fixed above. The report now splits waiting for memory into budget full and other apps holding memory. With the fix: Gemma 100% thread use in both passes and no waiting for memory, round trip 341 s; Qwen3.5-2B 99%, 18.9 s. Reading the 16.8 GB input from storage cost 163 thread-seconds (7% of the work).
- Gemma 4 31B Q4_0: the lattice was still chosen for all 23 Q6_K segments, same size. Q6_K work 320.8 → 247.8 thread-seconds, as pass 1 no longer codes Q6_K twice (82 → 74 s); round trip 328 → 304 s, thread use 79% → 94%.

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
