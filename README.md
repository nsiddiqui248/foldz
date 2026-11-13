# tenz

**Lossless compressor for GGUF models.** Successor of foldz: versions up to 0.15 were released as foldz, and the numbers continue. tenz shrinks quantized GGUF files (Q4_K_M and similar) by 7–13%, and decompression gives back the identical file, byte for byte. General-purpose compressors (zip, xz, zstd) gain very little on these files, because quantized weights look almost random to them.

tenz parses the GGUF file, unpacks each tensor's blocks into the real quantized values and their scales, and codes them with context models fitted to the data itself, using rANS and a range coder. Quantized values are coded as whole symbols (Q8_0 and floats as 4-bit symbols) from static tables: the compressor counts the whole file first and stores one shared seed table per type, and each segment stores only how its tables differ from the seed. Decompression therefore learns nothing. Metadata and padding are coded with a byte-level context-mixing model; tensors of types tenz has no model for are stored as they are. Nothing depends on the model's name or architecture. Safetensors files (F32, F16, BF16 tensors) are parsed the same way.

> Status: **0.20, pre-release** (file format 20). The file format may change before 1.0; a `.tenz` file is only guaranteed to decompress with the version that made it. Testing and review are very welcome (see [Help wanted](#help-wanted)). Changes per version: [CHANGELOG.md](CHANGELOG.md).

## Results

All measurements were made on a **Poco F5**, a mid-range Android phone, in Termux with 8 threads. Its chip is the Snapdragon 7+ Gen 2: 1 Cortex-X2 core at 2.91 GHz, 3 Cortex-A710 at 2.49 GHz and 4 Cortex-A510 at 1.8 GHz. Phones slow down as they heat up, so treat times as ±20%. Safetensors results are [below](#safetensors). Every round trip (`tenz t`: compress and decompress in memory, then compare) was byte-identical.

| Model | Type | Size MiB | tenz MiB | Saved | Round trip | Version |
|---|---|---|---|---|---|---|
| Qwen3.5-2B | Q4_K_M | 1211.9 | **1108.5** | 8.53% | 17 s | 0.20 |
| Qwen3-4B | Q4_K_M | 2381.6 | **2070.0** | 13.08% | 47 s | 0.12 |
| Qwen2.5-7B-Instruct | Q4_K_M | 4466.1 | **4111.0** | 7.95% | 81 s | 0.12 |
| Qwen3-0.6B | Q8_0 | 609.8 | **563.9** | 7.54% | 11 s | 0.13 |
| Gemma-4-31B-it (QAT) | Q4_0 | 16833.3 | **14694.9** | 12.70% | 258 s | 0.20 |

Later versions change the older rows' sizes by about 0.1 MiB and run faster: Qwen3.5-2B took 58 s with 0.9, 24 s with 0.14 and 17 s with 0.20. [CHANGELOG.md](CHANGELOG.md) has the results of each version.

Qwen3-4B and Gemma-4-31B save the most partly because most of their Q6_K values sit on a coarser lattice, the pattern left when a model is converted from a 4-bit AWQ/GPTQ checkpoint or trained for quantization. tenz detects this per segment and codes the lattice instead: Gemma's Q6_K tensors saved 34.83%, Qwen3-4B's 23.2%.

Nine public Q4_K_M models with the first release (0.9). Current versions make each file a few MiB smaller and run about three times faster:

| Model | Size MiB | 0.9 MiB | Saved | Round trip |
|---|---|---|---|---|
| Qwen1.5-0.5B-Chat | 388.3 | 353.6 | 8.93% | 17 s |
| Qwen3-0.6B | 378.3 | 348.7 | 7.82% | 17 s |
| Qwen2.5-1.5B-Instruct | 1065.6 | 984.2 | 7.64% | 40 s |
| Qwen3.5-2B | 1211.9 | 1111.3 | 8.30% | 58 s |
| Qwen2.5-Coder-3B-Instruct | 2007.4 | 1832.8 | 8.70% | 88 s |
| Llama-3.2-3B-Instruct | 1925.8 | 1780.4 | 7.55% | 93 s |
| Qwen3-4B | 2381.6 | 2073.3 | **12.94%** | 138 s |
| Qwen2.5-7B-Instruct | 4466.1 | 4113.8 | 7.89% | 220 s |
| Gemma-4-E2B (fine-tune) | 3269.1 | 3016.3 | 7.73% | 220 s |
| **All nine** | **17,094** | **15,614** | **8.66%** | |

By tensor type, across these models:

| Type | Saved | Notes |
|---|---|---|
| Q4_K | 6.9–8.5% | the bulk of every Q4_K_M file |
| Q5_K | 6.8–7.1% | |
| Q6_K | 6.3–8.8% | 23–35% on lattice models (above) |
| Q4_0 | 11.1% | Gemma-4-31B-it |
| Q8_0 | 6.8% | Qwen3-0.6B |
| BF16 | 34.1–34.3% | |
| F32 | 63–81% | norms and biases, small |
| metadata | about 80% | tokenizer, tensor index |

### Compared with general compressors

Qwen3.5-2B (Q4_K_M, 1211.9 MiB), on the same phone:

| Compressor | Size MiB | Saved | Time |
|---|---|---|---|
| `zstd -19 --long=27 -T0` | 1186.9 | 25.0 MiB (2.07%) | 3 min 58 s, compress only (24.6 CPU-minutes) |
| `xz -9 -T0` | 1190.4 | 21.6 MiB (1.78%) | 22 min 0 s, compress only (stayed on one thread) |
| tenz 0.20 | 1108.5 | **103.4 MiB (8.53%)** | 17 s to compress and decompress |

Quantized weights are close to incompressible for general-purpose compressors; most of what they save is probably the metadata. tenz saves about four times as much here. In fairness: tenz only works this well on GGUF, while zstd and xz work on any file, and zstd decompresses far faster. `xz -9 -T0` stayed on one thread, most likely because multithreaded level 9 would need more memory than xz allows itself by default. Baselines for other models are welcome.

## How it works

| Tensor type | Model |
|---|---|
| Q4_K, Q5_K | Whole blocks from static tables (seed per file, differences per segment), coded with rANS in 4 interleaved lanes. Values as whole symbols, one sub-block per lane; contexts: the sub-block's zero point, extremes seen (0 and max), position. Headers one block per lane: d and dmin as four 4-bit symbols each (like the floats), scales and mins as a 2-bit and a 4-bit symbol (scale given the previous scale, min given its scale and the previous min). |
| Q6_K | Whole blocks from static tables, coded with rANS in 4 interleaved lanes (one block's header per lane, 4 sub-blocks of each block per lane), in streams of 2048 blocks. Values as whole symbols; context: the column's running offset and width (mean and spread in units of the sub-block's step), position, extremes seen. Header: d as four 4-bit symbols, each scale as two (the low one given the high one and the whole previous scale). Or, if smaller, the AWQ lattice model for models converted from 4-bit AWQ/GPTQ checkpoints or trained for quantization: per 16-value sub-block the lattice size and span, then per value its lattice step (given the lattice size and its column's sign share and average step size over the rows above) and the error, with the same headers, its own static tables and lanes over consecutive sub-blocks. |
| Q4_0 | Whole blocks from static tables, coded with rANS in 4 interleaved lanes (one block per lane). Values as 4-bit symbols; contexts: the column's share of positive weights so far, whether the block's extreme (q = 0) was seen, the average distance from the centre so far, position. The scale d as four 4-bit symbols, like the floats. |
| Q8_0 | Values as two 4-bit symbols (high nibble, then low nibble given the high one) from static tables. Context: the column's sign bias, extremes seen, average magnitude so far in the block. |
| F32, F16, BF16 | Each 16 bits as four 4-bit symbols from static tables: sign, exponent and top mantissa bits each given the ones above (the first also given an earlier value); the low mantissa bits with little context, since they are close to random in trained weights. F32's low 16 bits the same way. Coded with rANS in 4 interleaved lanes: decoding needs no division, and the CPU works on 4 values at once. |
| Metadata, padding | Byte-level context mixing: orders 0–6, a match model, and a small logistic mixer. |

Safety and engineering:

- **Unmodelled types stored as is.** GGUF tensors of types tenz has no model for are copied as they are, so a model in such a type compresses in minutes, not hours.
- **Raw fallback.** Every segment whose coded size would not be smaller is stored raw, so the output is never larger than the input, apart from a small index.
- **Parallel segments.** Big tensors are split into segments (1/24 of the file, 8 to 48 MiB, by default), coded independently and in parallel.
- **Memory-aware.** A segment starts only while its estimated memory fits in what the device has free, keeping a reserve for the system and other apps. tenz checks free memory again before each new segment, so it backs off if another app needs memory while it runs.
- **Determinism.** Contexts use only exact or IEEE-754 arithmetic, so files decode identically on every platform. The build refuses settings that would break this: x87 excess precision and `-ffast-math`. Fused multiply-add is disabled in the source.
- **Untrusted input.** The decompressor checks every index field (sizes, overflow, offsets) before using it.

## Build

One C file, no dependencies.

```
gcc -O2 -pthread -ffp-contract=off -o tenz tenz.c -lm
```

clang works with the same flags, including on Termux. 32-bit x86 also needs `-msse2 -mfpmath=sse`.

Supported platforms: Linux, Android (Termux), macOS, BSD. Windows: use WSL, since tenz needs POSIX `mmap` and `pwrite`.

## Usage

```
tenz c model.gguf          # writes model.gguf.tenz
tenz d model.gguf.tenz     # writes model.gguf
tenz c in.gguf out.tenz    # explicit output name
tenz t model.gguf          # round-trip test in memory, writes nothing
tenz v                     # version
```

Options can go anywhere on the line:

| Option | Effect |
|---|---|
| `-t N` | threads (default: all cores) |
| `-s MiB` | segment size (default: 1/24 of the file, 8 to 48 MiB); lower uses less memory |
| `-m MiB` | memory limit for tenz, instead of the automatic one |
| `-H` | no longer needed: every block header is coded as symbols since 0.19. Accepted and ignored, so older scripts still work |
| `-f` | overwrite an existing output file |

tenz never overwrites a file unless `-f` is given, and never writes over its own input.

**Memory.** By default tenz uses what is free at start, minus a reserve for the device: 15% of RAM, at least 512 MiB and at most 2 GiB. Segments run in parallel only as far as that budget allows; one segment can always run, so tenz never stalls. With `-m` you set the limit yourself, and the free-memory checks are off. At start tenz prints the budget, and at the end the peak it used. Free memory is read from `/proc/meminfo` on Linux and Android, from the kernel on macOS and FreeBSD, and a cgroup v2 limit counts too (containers).

To check a decompressed file against the original:

```
cmp original.gguf restored.gguf && echo identical
```

## Benchmarks on other machines

All numbers above were measured on one phone, the Poco F5. To measure on your machine:

```
gcc -O2 -pthread -ffp-contract=off -o tenz tenz.c -lm
./bench.sh model.gguf [more files...]
```

`bench.sh` checks the round trip and prints a markdown table: tenz size, compress and decompress time, and `zstd -19 --long` for comparison (`XZ=1 ./bench.sh ...` adds `xz -9`, which takes many minutes per GB). The repository's **bench** workflow (Actions tab, Run workflow) does the same on a GitHub x86 runner with public models (by default Qwen2.5-0.5B-Instruct in Q4_K_M, Q8_0 and F16), so anyone can reproduce the numbers. Results from other machines are very welcome: open an issue with the table.

## Beyond GGUF

### Safetensors

tenz reads the safetensors header and codes F32, F16 and BF16 tensors with its float models; other tensor types and the header go to the byte model. Same phone:

| File | Size | tenz | Round trip | Version |
|---|---|---|---|---|
| Llama-3.2-1B-Instruct (BF16) | 2357.1 MiB | **1549.8 MiB (34.25%)** | 20 s | 0.20 |
| model-bf16.safetensors | 256.6 MiB | **169.2 MiB (34.07%)** | 2.1 s | 0.20 |
| model.safetensors (F32, converted from BF16) | 513.2 MiB | **165.8 MiB (67.70%)** | 4.1 s | 0.19 |

The byte model alone, without the float models, gave 171.9 MiB for both small files. The F32 file holds the same BF16 values with 16 zero bits added, and those cost almost nothing. As with GGUF, the gain on floats comes from the sign and exponent: the low mantissa bits of trained weights are close to random.

### Other files

Any other file is coded with the byte model alone. These file types were investigated during development; full details, including methods and every measurement, are in [docs/FINDINGS.md](docs/FINDINGS.md).

| File | xz -9 | Byte model | Note |
|---|---|---|---|
| C++ source, 1.4 MiB | 128 KiB | 113 KiB | about 12% smaller than xz |
| Python source corpus, 1.75 MB | 334,460 B | 298,333 B | 10.8% smaller than xz |
| APK, 73.4 MiB | about 69 MB | 70.2 MiB (4.4% saved) | xz wins: APK contents are already deflated |
| PDF, 1.4 MiB | | 2.4% saved | its streams are Flate-compressed |
| 16-bit numeric samples, 1 MB | 60,516 B | about 122,100 B | xz wins: long repeats favour LZ |
| JPEG | 0% | about 1% | already entropy-coded |

The byte model is strong on text and source code, about 6–12% better than xz. It cannot shrink data that is already compressed, and nothing can without parsing the format first.

### Format-aware experiments (`experiments/`)

These are measurement scripts, not compressors: they estimate what format-aware recompression could save, and verify that the original bytes can be rebuilt exactly.

| Script | What it found |
|---|---|
| `deflate_scan.py` | Finds deflate streams in ZIP/APK/JAR, gzip, PNG and PDF, and searches zlib settings that reproduce them bit-exactly. A 74.35 MB APK: every stream matched, estimated 74.35 → 49.10 MB (34% saved), against about 69 MB with xz. A 2 GB downloads folder: only 10.9 MB was deflate, saving about 0.9 MB. A PNG from an unknown encoder: 0% matched. |
| `jpeg_scan.py` | Decodes baseline JPEG to DCT coefficients, re-encodes byte-exact, and estimates coding with neighbour-context models. 16.6% saved on test images (11.6–19.2%), where xz saves 0%. |
| `recomp.py` | Blind byte- or bit-level modelling of gzip, bzip2 and xz output gains nothing (it costs 4–6% extra). Format-aware recompression of a gzip file gave 19.9%, rebuilt byte-identical. |

The conclusion from all experiments: gains come from parsing the format and modelling the real values with the right contexts, not from reading compressed bytes blindly. Mature tools already cover ZIP/PNG/PDF ([precomp](https://github.com/schnaader/precomp-cpp), [preflate-rs](https://github.com/microsoft/preflate-rs)) and JPEG ([Lepton](https://github.com/microsoft/lepton_jpeg_rust), about 22%). For quantized GGUF we found no comparable tool, which is the gap tenz fills. [ZipNN](https://arxiv.org/abs/2411.05239) targets floating-point (BF16) model weights.

## Limitations

- **Supported tensor types:** Q4_K, Q5_K, Q6_K, Q4_0, Q8_0, F32, F16 and BF16. GGUF tensors of other types (IQ quants, Q2_K, Q3_K, Q4_1, Q5_0...) are stored as they are, and tenz says so at the start: the byte model would gain a few percent at most on them and take hours on a large model. Safetensors tensors of other types (F8, integers) still go to the byte model.
- **Speed:** on the Poco F5, a round trip (compress and decompress) runs at about 65–70 MiB/s: Qwen3.5-2B (1.2 GB) in 17 s, Gemma-4-31B (16.8 GB) in 258 s. That is far slower than zstd decompresses, because each value is decoded from a context model; Q4_K, Q5_K, Q6_K, Q4_0 and floats in 4 interleaved rANS lanes without division, Q8_0 still with the range coder (a division per symbol). Metadata goes through the slower byte model, in 2 MiB pieces so that it runs in parallel.
- **Format:** pre-1.0, so it may change between versions.

## Help wanted

1. **Cross-platform round trip.** Compress on one machine and decompress on another, for example an x86-64 PC and an ARM64 phone. Q6_K models matter most here.
2. **Baselines.** Report `zstd -19 --long` and `xz -9` sizes (and times) on other models, or on any GGUF.
3. **More models and quant types,** especially Q5_K_M, Q6_K and Q8_0 files.
4. **Fuzzing.** Run AFL++ or libFuzzer with AddressSanitizer on `tenz d`.
5. **Ablations.** Measure how much each context contributes on real tensors. Measured so far: column statistics for Q4_K (as Q6_K uses) would gain only about 0.2 points on Q4_K values, about 0.1% of the file, and cost speed, so tenz does not use them.

When reporting results, include the full `tenz t` output, the model's Hugging Face name, and your CPU and OS.

**Security:** please report suspected vulnerabilities (for example, a crafted `.tenz` file that crashes the decompressor) privately via GitHub's "Report a vulnerability" button on the Security tab, not as a public issue.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
