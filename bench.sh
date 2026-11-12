#!/bin/sh
# Benchmark foldz against zstd and xz on the given files, on this machine.
# Prints a markdown table; sizes are exact, times are wall-clock seconds.
#   ./bench.sh model.gguf [more files...]
# Needs a built ./foldz (or FOLDZ=/path/to/foldz); zstd and xz are used if installed.
set -e
FOLDZ=${FOLDZ:-./foldz}
[ -x "$FOLDZ" ] || { echo "build foldz first: gcc -O2 -pthread -ffp-contract=off -o foldz foldz.c -lm" >&2; exit 1; }
now() { date +%s.%N; }
since() { awk -v a="$1" -v b="$(now)" 'BEGIN { printf "%.1f", b - a }'; }
pct() { awk -v a="$1" -v b="$2" 'BEGIN { printf "%.2f%%", 100 * (1 - b / a) }'; }
mib() { awk -v a="$1" 'BEGIN { printf "%.1f", a / 1048576 }'; }
cpu=$( (grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | sed 's/^ //') || uname -m)
echo "Machine: ${cpu:-unknown}, $(nproc 2>/dev/null || echo '?') threads, $("$FOLDZ" v)"
echo
echo "| File | Size MiB | foldz MiB | foldz saved | foldz compress | foldz decompress | zstd -19 --long saved | zstd time | xz -9 saved | xz time |"
echo "|---|---|---|---|---|---|---|---|---|---|"
tmp=${TMPDIR:-/tmp}/foldz-bench.$$
for f in "$@"; do
    n=$(wc -c < "$f")
    t=$(now); "$FOLDZ" c "$f" "$tmp.foldz" -f > /dev/null 2>&1; tc=$(since "$t")
    t=$(now); "$FOLDZ" d "$tmp.foldz" "$tmp.out" -f > /dev/null 2>&1; td=$(since "$t")
    cmp -s "$f" "$tmp.out" || { echo "ROUND TRIP FAILED: $f" >&2; exit 1; }
    z=$(wc -c < "$tmp.foldz"); rm -f "$tmp.out"
    zs="-"; zt="-"; xs="-"; xt="-"
    if command -v zstd > /dev/null; then t=$(now); zb=$(zstd -19 --long=27 -T0 -q -c "$f" | wc -c); zt="$(since "$t") s"; zs=$(pct "$n" "$zb"); fi
    if command -v xz > /dev/null; then t=$(now); xb=$(xz -9 -T0 -q -c "$f" | wc -c); xt="$(since "$t") s"; xs=$(pct "$n" "$xb"); fi
    echo "| $(basename "$f") | $(mib "$n") | $(mib "$z") | **$(pct "$n" "$z")** | $tc s | $td s | $zs | $zt | $xs | $xt |"
done
rm -f "$tmp.foldz"
