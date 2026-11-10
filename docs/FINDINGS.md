# foldz and format-aware recompression: findings and experiments

November 2026 · Follow-up to the recompression handover (9 November 2026)

This document collects everything measured while extending the foldz work beyond GGUF: the deflate scanner, the JPEG scanner, tests of foldz itself on other files, a security review of the decompressor, and the release fixes. Every number below was measured, unless it is marked as a published figure from another project.

## 1. Summary

- Format-aware recompression works on real files. A real APK (74.35 MB) was estimated at 49.10 MB, 34% smaller, with every deflate stream reproducible bit-exactly by zlib. `xz` on the same APK gave about 69 MB and zip about 70 MB.
- The gain depends entirely on how much of a file is zlib-made deflate. A 2 GB downloads folder gave about 0.9 MB, because almost nothing in it was deflate.
- A first JPEG model (pure Python) saved 16.6% on baseline JPEGs, rebuilt byte-exact. `xz -9` saved 0%.
- For ZIP/PNG/PDF and JPEG, existing tools already go further: precomp with preflate, and Lepton (22%). foldz's niche, lossless compression of quantized GGUF, has no comparable tool that we found.
- foldz's byte model (context mixing) beats `xz -9e` by 6–11% on text and source code.
- A heap-buffer overflow in the foldz decompressor was found and fixed. A crafted 80-byte file triggered it.
- Release fixes were applied: version numbering, cross-platform determinism guards, platform checks.

## 2. Deflate scanner (`deflate_scan.py`)

The scanner finds deflate streams in ZIP (APK, JAR, docx...), gzip, PNG, PDF and bare zlib files, including nested containers up to 3 levels deep. For each stream it searches zlib settings (level, memLevel, window, strategy) for one that reproduces the original bytes exactly. Matched streams are costed as LZMA2 of the content plus 3 bytes of settings; unmatched streams stay raw.

### 2.1 Memory problem and fix

Scanning a 2 GB folder caused memory pressure, and Android killed the app. There were three causes:

| Cause | Effect |
|---|---|
| Whole files read with `f.read()` | Videos of several GB loaded into RAM, although they contain no deflate |
| LZMA2 preset 9 always used a 64 MiB dictionary | About 11× the dictionary in memory per worker |
| `-j` defaulted to the CPU count (8 on phones) | All of the above multiplied by 8 |

LZMA2 preset 9 memory, measured on a 40 MB input:

| Dictionary | Extra memory | Output size |
|---|---|---|
| 64 MiB (preset default) | +430 MB | 11,310,883 B |
| 16 MiB | +207 MB | 11,374,419 B |
| 8 MiB | +115 MB | 11,430,680 B |

Fixes:

- **Memory-mapped files.** Files are mapped instead of read, so non-deflate files cost almost nothing.
- **Capped LZMA dictionary.** The dictionary is never larger than the content, and is capped per worker. On a 60 MB stream this gave 23.9% instead of 24.4% at a 32 MiB cap, and 20.0% at a 4 MiB cap, so the estimate errs on the conservative side.
- **Bounded inflate.** Streams are inflated in pieces, and over-limit streams are measured without being kept.
- **Memory budget.** By default it is 60% of `MemAvailable`; the worker count, dictionary cap and skip size are derived from it.
- **Per-worker cap.** `RLIMIT_DATA` on each worker means an outlier fails one file instead of the whole app.

Peak memory on the same test set dropped from 855 MB to 205 MB, with identical results.

### 2.2 Results on real files

**PNG** (`wallhaven-j83zom_2560x1920.png`, 8.41 MB): 0% matched. It has 2055 IDAT chunks of 4096 B. The ceiling if the stream could be rebuilt by a diff-based method is 46.2%. The encoder is not the local zlib.

**APK** (`nightmare-1.6.0.apk.zip`, 74.35 MB):

| Level | Streams | Matched | Size MB | After MB | Saved |
|---|---|---|---|---|---|
| Outer zip | 1 | 100% | 74.35 | 73.13 | 1.6% |
| Nested zip (APK members) | 100 | 100% | 73.38 | 49.37 | 32.7% |
| Nested PNG | 13 | 100% | 0.18 | 0.16 | 13.9% |
| Total with nested streams | | | 74.35 | 49.10 | 34.0% |

Matching settings: 99 × L6 mem8 w15 default, 13 × L9 mem8 w15 filtered, 2 × L9 mem8 w15 default. For comparison, `xz` on the APK gave about 69 MB and zip about 70 MB. They cannot see inside deflated data.

**Downloads folder** (`1DM/`, about 2 GB, 119 files):

- **No deflate stream:** 91 files (videos and other media).
- **ZIP files:** 28, all ad-block filter lists, 10.93 MB in total.
- **Matched:** 11 of 28 streams (39.3%; 47.3% by bytes), all L6 mem8 w15 default.
- **Saved:** 8.4%, about 0.9 MB. The ceiling if all matched is 18.5%.
- **Unmatched streams:** 17 failed all 3,150 settings combinations, so another encoder or zlib version made them. Not yet diagnosed with `--diagnose`.

**Plain text and xz** (`.cpp`, `.txt`, `.cpp.xz`): no deflate stream, so nothing to recompress. This is expected; xz uses LZMA, not deflate.

### 2.3 Conclusion for deflate

The approach pays off on software packages (APK, JAR and similar), where most of the bytes are zlib-deflated code and text. It does nothing for media-heavy folders. Unzipping and using xz gives a similar size, but cannot rebuild the original archive, so an APK's signature breaks. Reproducing the exact bytes is the point.

## 3. Whole-file fallback with the Gray fold

Files with no deflate stream are now processed with a warning instead of skipped. Each file is sampled first (three 1 MB samples; if zlib -1 can't shrink any of them, the file is kept raw). Otherwise it is compressed with LZMA2 twice, plain and after the Gray fold `g = x ^ (x >> 1)` per byte, and the smaller result is kept.

| File | Saved | Winner |
|---|---|---|
| `.cpp` source | 66.9% | fold, by 1 byte |
| 16-bit numeric data | 93.9% | plain |
| `.csv` | 68.3% | plain |
| `.xz` | 0% | (already compressed) |
| random `.mp4` | 0% | detected as already compressed, kept raw |

With an LZ-style compressor, the fold almost never changes the result: a repeated string is still repeated after a one-to-one relabeling of byte values.

## 4. Bit-level models on compressed data (from the handover)

Adaptive bit-level models on the output of good compressors, measured in bits per stored bit (1.0 = no gain):

| Compressed with | Byte entropy | After fold | Bit given previous 16 bits | Bit given previous byte | xz again |
|---|---|---|---|---|---|
| gzip -6 | 0.9999 | 0.9999 | 1.047 | 1.036 | 1.0002 |
| bzip2 -9 | 0.9989 | 0.9989 | 1.048 | 1.038 | 1.0002 |
| xz -6 | 0.9999 | 0.9999 | 1.056 | 1.042 | 1.0002 |

Blind byte- or bit-level modelling does not shrink good compressor output; it costs 4–6% extra. Gains appear only after parsing the format and modelling the real symbols.

## 5. JPEG scanner (`jpeg_scan.py`)

The scanner, in pure Python, works in three steps:

1. **Decode.** It parses baseline JPEG and Huffman-decodes the scan into quantized DCT coefficients.
2. **Verify.** It re-encodes the coefficients with the file's own Huffman tables and checks the result is byte-identical.
3. **Estimate.** It estimates the coded size with adaptive binary context models:
   - **DC:** MED prediction from the left, above and above-left blocks; the residual is centre-folded (signed to unsigned).
   - **Block nonzero count:** a 6-bit binary tree, with the neighbours' counts as context.
   - **AC coefficients:** zero flag, magnitude, mantissa and sign, each with the same coefficient in the left and above blocks as context.

Results on test images (made with libjpeg via Pillow from scikit-image sample photos):

| File | Saved |
|---|---|
| big_q90.jpg (2400×1800) | 19.15% |
| gravel_q85.jpg | 16.80% |
| astro_q95_444.jpg (4:4:4) | 15.35% |
| camera_q85.jpg (grayscale) | 12.93% |
| astro_rst.jpg (restart markers) | 12.63% |
| motorcycle_left_q85.jpg | 12.61% |
| astronaut_q85.jpg | 12.52% |
| coffee_q85.jpg | 11.93% |
| astro_opt.jpg (optimized Huffman tables) | 11.58% |
| astro_prog.jpg | unsupported (progressive) |
| **All 9 rebuilt files** | **16.6%** (xz -9: 0.0%) |

Coefficient data went from 1.203 MB Huffman-coded to 1.004 MB modelled (16.5% smaller). Where the modelled bits go:

| Part | Share |
|---|---|
| dc | 11.1% |
| nonzero count | 6.6% |
| zero flags | 18.4% |
| magnitude | 28.4% |
| mantissa | 19.1% |
| sign | 16.4% |

Limitations: baseline only (no progressive or arithmetic coding), estimate only (no range coder output yet), and slow in pure Python (about 4 s for a 4-megapixel image on a PC).

## 6. Existing tools (prior art)

| Tool | What it does | Published result |
|---|---|---|
| Lepton (Microsoft Rust port of Dropbox's tool) | Lossless JPEG recompression, baseline and progressive, bit-exact | up to 22% |
| packJPG, JPEG XL lossless transcoding | Lossless JPEG recompression | about 20% |
| precomp (+ preflate) | Recompresses deflate in ZIP/PNG/PDF, plus JPEG, MP3, GIF, bzip2; writes real files | depends on content |
| preflate-rs (Microsoft) | Deflate reconstruction by recording differences from a model, so it also handles non-zlib encoders | depends on content |
| ZipNN (IBM Research) | Lossless compression of floating-point model weights | often 33% on BF16 models |

None of these target quantized GGUF (Q4_K, Q5_K, Q6_K, Q8_0). That remains foldz's niche.

Links:
- https://github.com/microsoft/lepton_jpeg_rust
- https://github.com/schnaader/precomp-cpp
- https://github.com/microsoft/preflate-rs
- https://arxiv.org/abs/2411.05239 (ZipNN)

## 7. foldz review

### 7.1 Byte model on non-GGUF files

foldz codes metadata and other bytes with a context-mixing model: orders 0–6, a match model, and a small mixer in the logistic domain. With a small patch, any file can go through it. A real compress and decompress round trip was byte-identical.

| File | xz -9e | foldz | vs xz |
|---|---|---|---|
| Python source, 1,746,492 B (handover corpus) | 334,460 B | 298,333 B | 10.8% smaller |
| Source file, 29,367 B | 9,776 B | 9,147 B | 6.4% smaller |
| 16-bit numeric samples, 1,000,000 B | 60,516 B | about 122,100 B | 2× larger |
| JPEG, 635,497 B | 0% | about 1% | |

For text and code, foldz's byte model is a better back end than LZMA2. On numeric data, LZ matching wins.

### 7.2 Gray fold with full-tree bit coding

foldz's own bit model (same update rule) was run on 200,000 synthetic 4-bit values, coded plain and folded. In both cases every bit was coded given all the lower bits:

| Values centred at | Plain | Folded |
|---|---|---|
| 3 | 3.0924 bits/value | 3.0931 bits/value |
| 8 | 3.3723 bits/value | 3.3723 bits/value |

When every bit is coded given all lower bits of the same value, the coding covers the full 16-symbol alphabet, and a one-to-one relabeling gives essentially the same cost. In this test the gains came from the contexts (zero point, extremes seen, position, column statistics, AWQ lattice). Reviewers are invited to test the fold's contribution on real Q4_K tensors, for example by running foldz with the Gray step removed on the same models.

### 7.3 Security: decompressor heap-buffer overflow (fixed)

The index of a `.foldz` file was trusted. `rows × blocks × block size` could wrap around in 64-bit arithmetic, so a crafted file passed the size checks:

- **Crafted file:** 80 bytes, one Q8_0 segment with `ne0 = 32` and `rows = 2^63`, so the size wraps to 0.
- **AddressSanitizer result:** heap-buffer-overflow, WRITE in `q80_code`. Q6_K and the float types had the same path.

Fix:

- **Decompressor:** rejects zero rows, an `ne0` that isn't a multiple of the block size (and of 256 for Q6_K), and any overflowing size product.
- **Compressor:** applies the same overflow checks to GGUF tensor dimensions and offsets, since GGUF downloads are untrusted too.

After the fix the crafted file is rejected with "corrupt index". Normal output is byte-identical to before.

### 7.4 Release fixes

- **Version.** Version and file format share one number: foldz 0.9, file format 9 (`FOLDZ09`). Until 1.0, a file is only guaranteed to decompress with the version that made it. `foldz v` prints the version, and files from other format versions get a clear message.
- **Determinism.**
  - x87 excess precision (`FLT_EVAL_METHOD != 0`) and `-ffast-math` now stop the build with an error.
  - A pragma disables fused multiply-add even if `-ffp-contract=off` is forgotten (GCC verified: no FMA instruction with it, one without). This matters on Termux, where clang fuses multiply-adds by default on ARM64.
- **Platforms.** Linux, Android (Termux), macOS and BSD are supported. A native Windows build stops with a message pointing to WSL or Cygwin.
- **Any-file mode.** Non-GGUF input goes through the byte model instead of being rejected.

## 8. Open items for testers

1. **Cross-platform round trip.** Compress on one machine and decompress on another (x86-64 PC and ARM64 phone), especially with Q6_K models. This is the real test of determinism.
2. **Fuzzing.** Run AFL++ or libFuzzer with AddressSanitizer on `foldz d` for several hours.
3. **Benchmarks.** Test 3–4 public models (e.g. Llama, Qwen, Mistral at Q4_K_M and Q6_K) against `zstd -19 --long` and `xz -9`, reporting ratio, speed both ways and peak memory.
4. **Fold contribution.** Compare foldz with and without the Gray step on real Q4_K tensors.
5. **Unmatched deflate streams.** Run `deflate_scan.py --diagnose` on the 17 unmatched filter-list ZIPs.

## 9. Reproducing

```
# deflate scan (Python 3.8+, no dependencies)
python deflate_scan.py PATH [--csv out.csv] [--diagnose] [--mem-mb 1200] [-j 2] [--no-raw]

# JPEG scan (Python 3.8+, no dependencies)
python jpeg_scan.py PATH [--csv out.csv]

# foldz
gcc -O2 -pthread -ffp-contract=off -o foldz foldz.c -lm
./foldz c model.gguf model.foldz      # compress
./foldz d model.foldz model.gguf      # decompress
./foldz t model.gguf                  # round trip in memory
./foldz v                             # version
```

Files: `deflate_scan.py`, `jpeg_scan.py`, `foldz.c` (0.9, with all fixes), `recomp.py` (handover experiment).
