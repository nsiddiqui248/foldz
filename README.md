# foldz

**Lossless compressor for GGUF models.** foldz shrinks quantized GGUF files (Q4_K_M and similar) by 7–13%, and decompression gives back the identical file, byte for byte. General-purpose compressors (zip, xz, zstd) gain very little on these files, because quantized weights look almost random to them.

foldz parses the GGUF file, unpacks each tensor's blocks into the real quantized values and their scales, and codes them with context models learned from the data itself, using a binary adaptive range coder. Everything else (metadata, other tensor types, padding) is coded with a byte-level context-mixing model. Nothing depends on the model's name or architecture.

> Status: **0.9, pre-release.** The file format may change before 1.0; a `.foldz` file is only guaranteed to decompress with the version that made it. Testing and review are very welcome (see [Help wanted](#help-wanted)).

## Results

Nine public Q4_K_M models, tested with `foldz t` (compress and decompress in memory, then compare). Every round trip was byte-identical. Measured on an Android phone in Termux, with 8 threads.

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
| Q6_K | 6.3–8.8% | **23.2%** on Qwen3-4B, where the AWQ lattice model was chosen for 36 of 43 segments |
| BF16 | 33.9% | one model |
| F32 | 63–81% | norms and biases, small |
| metadata | about 81% | tokenizer, tensor index |

In Qwen3-4B, most Q6_K values sit on a coarser lattice, the pattern left when a model is converted from a 4-bit AWQ/GPTQ checkpoint. foldz detects this per segment and codes the lattice instead.

Comparisons with `zstd -19 --long` and `xz -9` on the same models are not in this table yet. Contributions of those numbers are welcome.

## How it works

| Tensor type | Model |
|---|---|
| Q4_K, Q5_K | Values are folded (Gray code `g = n ^ (n >> 1)`), and every bit is coded given the lower ones. Contexts: the block's zero point, extremes seen (0 and max), position. Headers (d, dmin, scales, mins) are coded with their own contexts. |
| Q6_K | Context: the column's running offset and width (mean and spread in units of the sub-block's step), position, extremes seen. Or, if smaller, the AWQ lattice model for models converted from 4-bit AWQ/GPTQ checkpoints. |
| Q8_0 | Context: the column's sign bias, extremes seen, average magnitude so far in the block. |
| F32, F16, BF16 | Sign, exponent (context: previous exponent), mantissa (top bits given the exponent). |
| Everything else | Byte-level context mixing: orders 0–6, a match model, and a small logistic mixer. |

Safety and engineering:

- **Raw fallback.** Every segment whose coded size would not be smaller is stored raw, so the output is never larger than the input, apart from a small index.
- **Parallel segments.** Big tensors are split into segments (48 MiB by default), coded independently and in parallel.
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
| `-f` | overwrite an existing output file |

foldz never overwrites a file unless `-f` is given, and never writes over its own input. Peak memory is printed at start; reduce `-t` or `-s` if it's too high for your machine.

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
- **Speed:** about 19 MiB/s round trip (compress plus decompress) on a phone with 8 threads. Context modelling is symmetric, so decompression is as slow as compression.
- **Format:** pre-1.0, so it may change between versions.

## Help wanted

1. **Cross-platform round trip.** Compress on one machine and decompress on another, for example an x86-64 PC and an ARM64 phone. Q6_K models matter most here.
2. **Baselines.** Report `zstd -19 --long` and `xz -9` sizes on the models above, or on any GGUF.
3. **More models and quant types,** especially Q5_K_M, Q6_K and Q8_0 files.
4. **Fuzzing.** Run AFL++ or libFuzzer with AddressSanitizer on `foldz d`.
5. **Ablations.** Measure how much each context, and the Gray fold, contributes on real tensors.

When reporting results, include the full `foldz t` output, the model's Hugging Face name, and your CPU and OS.

**Security:** please report suspected vulnerabilities (for example, a crafted `.foldz` file that crashes the decompressor) privately via GitHub's "Report a vulnerability" button on the Security tab, not as a public issue.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
