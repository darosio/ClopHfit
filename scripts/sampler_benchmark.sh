#!/usr/bin/env bash
# Benchmark NUTS samplers (default, nutpie, numpyro, blackjax) across MCMC modes.
#
# Uses a reduced sample count (--mcmc-samples 400) to give a quick timing comparison.
# Run from the plate data directory (must contain list.pH.csv, scheme.0.txt, additions.pH).
#
# Usage (from L2 or L4 dir):
#   bash /path/to/sampler_benchmark.sh [plate_dir]
#
# Example:
#   cd /home/dati/arslanbaeva/data/raw
#   bash /home/dan/workspace/ClopHfit/scripts/sampler_benchmark.sh L2
#   bash /home/dan/workspace/ClopHfit/scripts/sampler_benchmark.sh L4

set -uo pipefail

PLATE_DIR="${1:-L2}"
BASE="/home/dati/arslanbaeva/data/raw/${PLATE_DIR}"
PPR="${PPR_CMD:-ppr}"
SAMPLES="${BENCH_SAMPLES:-400}"

SAMPLERS=(default nutpie numpyro blackjax)
MODES=(single multi multi-noise multi-noise-xrw)

OUTFILE="${BASE}/sampler_benchmark_results.tsv"
printf "plate\tmode\tsampler\tstatus\ttime_s\n" > "$OUTFILE"

echo "=== Sampler benchmark: ${PLATE_DIR} (${SAMPLES} samples) ==="
echo "Results → ${OUTFILE}"
echo ""

for mode in "${MODES[@]}"; do
  for sampler in "${SAMPLERS[@]}"; do
    outdir="${BASE}/compare/bench_${mode//-/_}_${sampler}"
    printf "  %-22s %-10s ... " "${mode}" "${sampler}"
    t_start=$(date +%s)
    set +e
    "$PPR" -o "$outdir" tecan "${BASE}/list.pH.csv" \
      --bg-adj --nrm \
      --sch "${BASE}/scheme.0.txt" \
      --add "${BASE}/additions.pH" \
      --fit-method huber \
      --mcmc "$mode" \
      --nuts-sampler "$sampler" \
      --mcmc-samples "$SAMPLES" \
      --no-png \
      > /tmp/ppr_bench_out.txt 2>&1
    status=$?
    set -e
    grep -iv "DEBUG\|PIL\|STREAM\|IDAT\|sBIT\|pHYs\|IHDR\|loop.fusion" /tmp/ppr_bench_out.txt |
      grep -i "error\|warning\|diverge\|cuda" | head -5 || true
    t_end=$(date +%s)
    elapsed=$((t_end - t_start))
    result_status="ok"
    if [[ $status -ne 0 ]]; then
      result_status="FAILED(${status})"
    elif ! find "$outdir" -name "ffit4.csv" 2> /dev/null | grep -q .; then
      result_status="no_ffit4"
    fi
    printf "  %s  (%ds)\n" "$result_status" "$elapsed"
    printf "%s\t%s\t%s\t%s\t%d\n" \
      "$PLATE_DIR" "$mode" "$sampler" "$result_status" "$elapsed" >> "$OUTFILE"
  done
  echo ""
done

echo ""
echo "=== Summary ==="
column -t "$OUTFILE"
