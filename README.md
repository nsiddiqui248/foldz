# foldz

**Lossless compressor for GGUF models.** foldz shrinks quantized GGUF files (Q4_K_M and similar) by 7–13%, and decompression gives back the identical file, byte for byte. General-purpose compressors (zip, xz, zstd) gain very little on these files, because quantized weights look almost random to them.

foldz parses the GGUF file, unpacks each tensor's blocks into the real quantized values and their scales, and codes them with context models fitted to the data itself, using a range coder. Quantized values are coded as whole symbols from static tables: the compressor counts the whole file first and stores one shared seed table per type, and each segment stores only how its tables differ from the seed. Decompression therefore learns nothing and is faster than compression. Everything else (metadata, other tensor types, padding) is coded with a byte-level context-mixing model. Nothing depends on the model's name or architecture.

> Status: **0.12, pre-release** (file format 12; files from older versions need the version that made them). The file format may change before 1.0; a `.foldz` file is only guaranteed to decompress with the version that made it. Testing and review are very welcome (see [Help wanted](#help-wanted)). Changes per version: [CHANGELOG.md](CHANGELOG.md).

## Results

Nine public Q4_K_M models, tested with `foldz t` (compress and decompress in memory, then compare). Every round trip was byte-identical. Measured on an Android phone in Termux, with 8 threads, using 0.9. The current version gives smaller files:

| Model | 0.9 MiB | 0.12 MiB | Saved with 0.12 | 0.9 round trip | 0.12 round trip |
|---|---|---|---|---|---|
| Qwen3.5-2B | 1111.3 | **1109.8** | 8.43% | 58 s | 35 s |
| Qwen3-4B | 2073.3 | **2070.0** | 13.08% | 138 s | 47 s |
| Qwen2.5-7B-Instruct | 4113.8 | **4111.0** | 7.95% | 220 s | 81 s |

With `-H`, Qwen2.5-7B-Instruct came out 0.5 MiB larger (4111.5 MiB) with 19% less work (66 s round trip).

Speed on Qwen3.5-2B, same phone (it slows down as it heats up, so treat these as ±20%):

| Version | Compress | Decompress |
|---|---|---|
| 0.9 | 58 s for compress and decompress together | |
| 0.11 | 12.0 s | 17.9 s |
| 0.12 | 22.8 s | **15.1 s** |

0.12 trades slower compression (an extra counting pass) for faster decompression and smaller files: compress once, decompress many times.

Results with 0.9:

| Model | Size MiB | foldz MiB | Saved | Round trip |
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

By tensor type, across the same models:

| Type | Saved | Notes |
|---|---|---|
| Q4_K | 6.9–8.5% | the bulk of every Q4_K_M file |
| Q5_K | 6.8% | one model |
| Q6_K | 6.3–8.8% | **23.2%** on Qwen3-4B (23.6% with 0.12), where the AWQ lattice model was chosen for 36 of 43 segments |
| BF16 | 33.9% | one model |
| F32 | 63–81% | norms and biases, small |
| metadata | about 81% | tokenizer, tensor index |

In Qwen3-4B, most Q6_K values sit on a coarser lattice, the pattern left when a model is converted from a 4-bit AWQ/GPTQ checkpoint. foldz detects this per segment and codes the lattice instead.

### Compared with general compressors

Qwen3.5-2B (Q4_K_M, 1211.9 MiB), on the same phone (8 cores):

| Compressor | Size MiB | Saved | Time |
|---|---|---|---|
| `zstd -19 --long=27 -T0` | 1186.9 | 25.0 MiB (2.07%) | 3 min 58 s, compress only (24.6 CPU-minutes) |
| `xz -9 -T0` | 1190.4 | 21.6 MiB (1.78%) | 22 min 0 s, compress only (stayed on one thread) |
| foldz 0.12 | 1109.8 | **102.1 MiB (8.43%)** | 23 s compress, 15 s decompress |

Quantized weights are close to incompressible for general-purpose compressors; most of what they save is probably the metadata. foldz saves about four times as much here. In fairness: foldz only works this well on GGUF, while zstd and xz work on any file, and zstd decompresses far faster. `xz -9 -T0` stayed on one thread, most likely because multithreaded level 9 would need more memory than xz allows itself by default. Baselines for other models are welcome.

## How it works

| Tensor type | Model |
|---|---|
| Q4_K, Q5_K | Values as whole symbols from static tables (seed per file, differences per segment). Contexts: the block's zero point, extremes seen (0 and max), position. Headers (d, dmin, scales, mins) are coded adaptively with their own contexts. |
| Q6_K | Values as whole symbols from static tables. Context: the column's running offset and width (mean and spread in units of the sub-block's step), position, extremes seen. Or, if smaller, the AWQ lattice model for models converted from 4-bit AWQ/GPTQ checkpoints (lattice steps and errors coded as symbols). |
| Q8_0 | Context: the column's sign bias, extremes seen, average magnitude so far in the block. |
| F32, F16, BF16 | Sign, exponent (context: previous exponent), mantissa (top bits given the exponent). |
| Everything else | Byte-level context mixing: orders 0–6, a match model, and a small logistic mixer. |

Safety and engineering:

- **Raw fallback.** Every segment whose coded size would not be smaller is stored raw, so the output is never larger than the input, apart from a small index.
- **Parallel segments.** Big tensors are split into segments (48 MiB by default), coded independently and in parallel.
- **Memory-aware.** A segment starts only while its estimated memory fits in what the device has free, keeping a reserve for the system and other apps. foldz checks free memory again before each new segment, so it backs off if another app needs memory while it runs.
- **Determinism.** Contexts use only exact or IEEE-754 arithmetic, so files decode identically on every platform. The build refuses settings that would break this: x87 excess precision and `-ffast-math`. Fused multiply-add is disabled in the source.
- **Untrusted input.** The decompressor checks every index field (sizes, overflow, offsets) before using it.

## Build

One C file, no dependencies.

```
gcc -O2 -pthread -ffp-contract=off -o foldz foldz.c -lm
```

clang works with the same flags, including on Termux. 32-bit x86 also needs `-msse2 -mfpmath=sse`.

Supported platforms: Linux, Android (Termux), macOS, BSD. Windows: use WSL, since foldz needs POSIX `mmap` and `pwrite`.

## Usage

```
foldz c model.gguf                 # writes model.gguf.foldz
foldz d model.gguf.foldz           # writes model.gguf
foldz c in.gguf out.foldz          # explicit output name
foldz t model.gguf                 # round-trip test in memory, writes nothing
foldz v                            # version
```

Options can go anywhere on the line:

| Option | Effect |
|---|---|
| `-t N` | threads (default: all cores) |
| `-s MiB` | segment size (default 48); lower uses less memory |
| `-m MiB` | memory limit for foldz, instead of the automatic one |
| `-H` | code block headers (scales, mins, d) as symbols: about 20% less work on Qwen2.5-7B for 0.01% more size (up to 0.05% on other models). Recorded in the file, so `foldz d` needs no option |
| `-f` | overwrite an existing output file |

foldz never overwrites a file unless `-f` is given, and never writes over its own input.

**Memory.** By default foldz uses what is free at start, minus a reserve for the device: 15% of RAM, at least 512 MiB and at most 2 GiB. Segments run in parallel only as far as that budget allows; one segment can always run, so foldz never stalls. With `-m` you set the limit yourself, and the free-memory checks are off. At start foldz prints the budget, and at the end the peak it used. Free memory is read from `/proc/meminfo` on Linux and Android, from the kernel on macOS and FreeBSD, and a cgroup v2 limit counts too (containers).

To check a decompressed file against the original:

```
cmp original.gguf restored.gguf && echo identical
```

## Beyond GGUF

Any other file is coded with the byte model alone. The other file types below were investigated while developing foldz. Full details, including methods and every measurement, are in [docs/FINDINGS.md](docs/FINDINGS.md).

### foldz on other files

| File | xz -9 | foldz | Note |
|---|---|---|---|
| C++ source, 1.4 MiB | 128 KiB | 113 KiB | about 12% smaller than xz |
| Python source corpus, 1.75 MB | 334,460 B | 298,333 B | 10.8% smaller than xz |
| APK, 73.4 MiB | about 69 MB | 70.2 MiB (4.4% saved) | xz wins: APK contents are already deflated |
| PDF, 1.4 MiB | | 2.4% saved | its streams are Flate-compressed |
| 16-bit numeric samples, 1 MB | 60,516 B | about 122,100 B | xz wins: long repeats favour LZ |
| JPEG | 0% | about 1% | already entropy-coded |

foldz's byte model is strong on text and source code, about 6–12% better than xz. It cannot shrink data that is already compressed, and nothing can without parsing the format first.

### Format-aware experiments (`experiments/`)

These are measurement scripts, not compressors: they estimate what format-aware recompression could save, and verify that the original bytes can be rebuilt exactly.

| Script | What it found |
|---|---|
| `deflate_scan.py` | Finds deflate streams in ZIP/APK/JAR, gzip, PNG and PDF, and searches zlib settings that reproduce them bit-exactly. A 74.35 MB APK: every stream matched, estimated 74.35 → 49.10 MB (34% saved), against about 69 MB with xz. A 2 GB downloads folder: only 10.9 MB was deflate, saving about 0.9 MB. A PNG from an unknown encoder: 0% matched. |
| `jpeg_scan.py` | Decodes baseline JPEG to DCT coefficients, re-encodes byte-exact, and estimates coding with neighbour-context models. 16.6% saved on test images (11.6–19.2%), where xz saves 0%. |
| `recomp.py` | Blind byte- or bit-level modelling of gzip, bzip2 and xz output gains nothing (it costs 4–6% extra). Format-aware recompression of a gzip file gave 19.9%, rebuilt byte-identical. |

The conclusion from all experiments: gains come from parsing the format and modelling the real values with the right contexts, not from reading compressed bytes blindly. Mature tools already cover ZIP/PNG/PDF ([precomp](https://github.com/schnaader/precomp-cpp), [preflate-rs](https://github.com/microsoft/preflate-rs)) and JPEG ([Lepton](https://github.com/microsoft/lepton_jpeg_rust), about 22%). For quantized GGUF we found no comparable tool, which is the gap foldz fills. [ZipNN](https://arxiv.org/abs/2411.05239) targets floating-point (BF16) model weights.

## Limitations

- **Supported tensor types:** Q4_K, Q5_K, Q6_K, Q8_0, F32, F16 and BF16. Other types (IQ quants, Q2_K, Q3_K, Q4_0...) are coded with the byte model, which gains little on them.
- **Speed:** on a phone with 8 threads, about 50 MiB/s to compress and 80 MiB/s to decompress. Within a segment each value still waits for the one before it, so foldz is far slower than zstd to decompress. `-H` cuts the work by up to about 20% (measured on ARM; less on x86). The metadata segment is coded on one thread: on Qwen3.5-2B it now takes most of the decompression time, and on machines with many cores it can be the longest part of any run.
- **Format:** pre-1.0, so it may change between versions.

## Help wanted

1. **Cross-platform round trip.** Compress on one machine and decompress on another, for example an x86-64 PC and an ARM64 phone. Q6_K models matter most here.
2. **Baselines.** Report `zstd -19 --long` and `xz -9` sizes (and times) on other models, or on any GGUF.
3. **More models and quant types,** especially Q5_K_M, Q6_K and Q8_0 files.
4. **Fuzzing.** Run AFL++ or libFuzzer with AddressSanitizer on `foldz d`.
5. **Ablations.** Measure how much each context contributes on real tensors. Measured so far: column statistics for Q4_K (as Q6_K uses) would gain only about 0.2 points on Q4_K values, about 0.1% of the file, and cost speed, so foldz does not use them.

When reporting results, include the full `foldz t` output, the model's Hugging Face name, and your CPU and OS.

**Security:** please report suspected vulnerabilities (for example, a crafted `.foldz` file that crashes the decompressor) privately via GitHub's "Report a vulnerability" button on the Security tab, not as a public issue.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
