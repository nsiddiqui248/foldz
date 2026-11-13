#!/bin/sh
# Benchmark tenz against zstd and xz on the given files, on this machine.
# Prints a markdown table; sizes are exact, times are wall-clock seconds.
#   ./bench.sh model.gguf [more files...]
# Needs a built ./tenz (or TENZ=/path/to/tenz). zstd is compared if installed; xz only with XZ=1
# (xz -9 takes many minutes per GB, especially on a phone).
set -e
TENZ=${TENZ:-./tenz}
[ -x "$TENZ" ] || { echo "build tenz first: gcc -O2 -pthread -ffp-contract=off -o tenz tenz.c -lm" >&2; exit 1; }
[ $# -gt 0 ] || { echo "usage: ./bench.sh model.gguf [more files...]   (TENZ=/path/to/tenz to use another build)" >&2; exit 1; }
now() { date +%s.%N; }
since() { awk -v a="$1" -v b="$(now)" 'BEGIN { printf "%.1f", b - a }'; }
pct() { awk -v a="$1" -v b="$2" 'BEGIN { printf "%.2f%%", 100 * (1 - b / a) }'; }
mib() { awk -v a="$1" 'BEGIN { printf "%.1f", a / 1048576 }'; }
# machine: Android (getprop), then lscpu, then /proc/cpuinfo, then the architecture
cpu=""
if command -v getprop > /dev/null; then
    dev=$(getprop ro.product.manufacturer 2>/dev/null) ; mod=$(getprop ro.product.model 2>/dev/null)
    soc=$(getprop ro.soc.model 2>/dev/null); [ -n "$soc" ] || soc=$(getprop ro.board.platform 2>/dev/null)
    cpu=$(echo "$dev $mod${soc:+, $soc}" | sed 's/^ *//')
fi
[ -n "$cpu" ] || cpu=$(lscpu 2>/dev/null | sed -n 's/^Model name: *//p' | head -1)
[ -n "$cpu" ] || cpu=$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | sed 's/^ //')
[ -n "$cpu" ] || cpu=$(uname -m)
echo "Machine: $cpu, $(nproc 2>/dev/null || echo '?') cores online, $(uname -s) $(uname -m), $("$TENZ" v)"
echo
echo "| File | Size MiB | tenz MiB | tenz saved | tenz compress | tenz decompress | zstd -19 --long saved | zstd time | xz -9 saved | xz time |"
echo "|---|---|---|---|---|---|---|---|---|---|"
tmp=${TMPDIR:-/tmp}/tenz-bench.$$
for f in "$@"; do
    n=$(wc -c < "$f")
    t=$(now); "$TENZ" c "$f" "$tmp.tenz" -f > /dev/null 2>&1; tc=$(since "$t")
    t=$(now); "$TENZ" d "$tmp.tenz" "$tmp.out" -f > /dev/null 2>&1; td=$(since "$t")
    cmp -s "$f" "$tmp.out" || { echo "ROUND TRIP FAILED: $f" >&2; exit 1; }
    z=$(wc -c < "$tmp.tenz"); rm -f "$tmp.out"
    zs="-"; zt="-"; xs="-"; xt="-"
    if command -v zstd > /dev/null; then t=$(now); zb=$(zstd -19 --long=27 -T0 -q -c "$f" | wc -c); zt="$(since "$t") s"; zs=$(pct "$n" "$zb"); fi
    if [ "${XZ:-0}" = 1 ] && command -v xz > /dev/null; then t=$(now); xb=$(xz -9 -T0 -q -c "$f" | wc -c); xt="$(since "$t") s"; xs=$(pct "$n" "$xb"); fi
    echo "| $(basename "$f") | $(mib "$n") | $(mib "$z") | **$(pct "$n" "$z")** | $tc s | $td s | $zs | $zt | $xs | $xt |"
done
rm -f "$tmp.tenz"
