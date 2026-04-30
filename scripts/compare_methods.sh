#!/usr/bin/env bash
# Compare fit methods and MCMC modes on L1–L4 real plate data.
#
# Usage:
#   ./scripts/compare_methods.sh [L1|L2|L3|L4|all]  [fit|mcmc|noise|all]
#
# Results land in:
#   /home/dati/arslanbaeva/data/raw/{L1,L2,L3,L4}/compare/{method}/
#
# Requires: ppr in PATH (activate venv first or use uv run ppr).
set -euo pipefail

PLATE="${1:-all}" # L1 | L2 | L3 | L4 | all
MODE="${2:-all}"  # fit | mcmc | noise | all

L1_DIR="/home/dati/arslanbaeva/data/raw/L1"
L2_DIR="/home/dati/arslanbaeva/data/raw/L2"
L3_DIR="/home/dati/arslanbaeva/data/raw/L3"
L4_DIR="/home/dati/arslanbaeva/data/raw/L4"

PPR="${PPR_CMD:-ppr}"
NUTS_SAMPLER="${NUTS_SAMPLER:-default}"
MCMC_SAMPLES="${MCMC_SAMPLES:-2000}"

# Per-plate MCMC multi-noise parameters (from shared_noise_params.csv, scheme.0.txt runs).
# L1: MCMC unreliable (4433 divergences) → fall back to L3 values as proxy.
declare -A ALPHA_Y1=([L1]="0.042" [L2]="0.057" [L3]="0.042" [L4]="0.101")
declare -A ALPHA_Y2=([L1]="0.019" [L2]="0.006" [L3]="0.019" [L4]="0.050")
declare -A GAIN_Y1=([L1]="0.064" [L2]="1.528" [L3]="0.064" [L4]="0.306")
declare -A GAIN_Y2=([L1]="0.138" [L2]="0.028" [L3]="0.138" [L4]="0.056")

# Per-plate scheme filename.
scheme_for() {
  local dir="$1"
  if [[ -f "${dir}/scheme.0.txt" ]]; then
    echo "scheme.0.txt"
  else
    echo "scheme.txt"
  fi
}

run_fit_methods() {
  local dir="$1"
  local sch
  sch=$(scheme_for "$dir")
  echo "=== Fit-method comparison: $dir (scheme=${sch}) ==="
  for method in lm huber irls; do
    outdir="compare/${method}"
    echo "  → fit-method=${method}  out=${outdir}"
    "$PPR" -o "${outdir}" tecan list.pH.csv \
      --bg-adj --nrm --sch "${sch}" --add additions.pH \
      --fit-method "${method}" --no-png 2>&1 | grep -E "ERROR|WARNING|CRITICAL" || true
    echo "    done"
  done
  # Huber + outlier removal at multiple z-score thresholds
  for zscore in 3.0 2.5 2.0; do
    outdir="compare/outlier_${zscore}"
    echo "  → fit-method=huber --outlier zscore:${zscore}:4  out=${outdir}"
    "$PPR" -o "${outdir}" tecan list.pH.csv \
      --bg-adj --nrm --sch "${sch}" --add additions.pH \
      --fit-method huber --outlier "zscore:${zscore}:4" --no-png 2>&1 | grep -E "ERROR|WARNING|CRITICAL" || true
    echo "    done"
  done
}

run_noise_fit() {
  local dir="$1"
  local plate
  plate=$(basename "$dir")
  local sch
  sch=$(scheme_for "$dir")
  local ay1="${ALPHA_Y1[$plate]}" ay2="${ALPHA_Y2[$plate]}"
  local gy1="${GAIN_Y1[$plate]}" gy2="${GAIN_Y2[$plate]}"
  echo "=== Noise-corrected fit: $plate  α=($ay1,$ay2)  gain=($gy1,$gy2) ==="

  # Alpha-only correction (gain still 1)
  outdir="compare/huber_alpha"
  echo "  → huber + --noise-alpha  out=${outdir}"
  "$PPR" -o "${outdir}" tecan list.pH.csv \
    --bg-adj --nrm --sch "${sch}" --add additions.pH \
    --fit-method huber \
    --noise-alpha "${ay1}" --noise-alpha "${ay2}" \
    --no-png 2>&1 | grep -E "ERROR|WARNING|CRITICAL" || true
  echo "    done"

  # Full correction: alpha + gain
  outdir="compare/huber_noise"
  echo "  → huber + --noise-alpha + --noise-gain  out=${outdir}"
  "$PPR" -o "${outdir}" tecan list.pH.csv \
    --bg-adj --nrm --sch "${sch}" --add additions.pH \
    --fit-method huber \
    --noise-alpha "${ay1}" --noise-alpha "${ay2}" \
    --noise-gain "${gy1}" --noise-gain "${gy2}" \
    --no-png 2>&1 | grep -E "ERROR|WARNING|CRITICAL" || true
  echo "    done"
}

run_mcmc_modes() {
  local dir="$1"
  local sch
  sch=$(scheme_for "$dir")
  echo "=== MCMC-mode comparison: $dir (scheme=${sch}) ==="
  for mcmc in single multi multi-noise multi-noise-xrw; do
    outdir="compare/mcmc_${mcmc//-/_}"
    echo "  → mcmc=${mcmc}  sampler=${NUTS_SAMPLER}  samples=${MCMC_SAMPLES}  out=${outdir}"
    "$PPR" -o "${outdir}" tecan list.pH.csv \
      --bg-adj --nrm --sch "${sch}" --add additions.pH \
      --fit-method huber --mcmc "${mcmc}" \
      --nuts-sampler "${NUTS_SAMPLER}" \
      --mcmc-samples "${MCMC_SAMPLES}" \
      2>&1 | grep -E "ERROR|WARNING|CRITICAL" || true
    echo "    done"
  done
  # multi-noise-xrw with per-well independent K (hierarchical partial pooling per CTR)
  outdir="compare/mcmc_multi_noise_xrw_free_ctr"
  echo "  → mcmc=multi-noise-xrw --ctr-free-k  sampler=${NUTS_SAMPLER}  samples=${MCMC_SAMPLES}  out=${outdir}"
  "$PPR" -o "${outdir}" tecan list.pH.csv \
    --bg-adj --nrm --sch "${sch}" --add additions.pH \
    --fit-method huber --mcmc multi-noise-xrw --ctr-free-k \
    --nuts-sampler "${NUTS_SAMPLER}" \
    --mcmc-samples "${MCMC_SAMPLES}" \
    2>&1 | grep -E "ERROR|WARNING|CRITICAL" || true
  echo "    done"
}

PLATES=("$L1_DIR" "$L2_DIR" "$L3_DIR" "$L4_DIR")

for dir in "${PLATES[@]}"; do
  plate=$(basename "$dir")
  if [[ "$PLATE" != "all" && "$PLATE" != "$plate" ]]; then
    continue
  fi
  pushd "$dir" > /dev/null
  mkdir -p compare
  if [[ "$MODE" == "fit" || "$MODE" == "all" ]]; then
    run_fit_methods "$dir"
  fi
  if [[ "$MODE" == "noise" || "$MODE" == "all" ]]; then
    run_noise_fit "$dir"
  fi
  if [[ "$MODE" == "mcmc" || "$MODE" == "all" ]]; then
    run_mcmc_modes "$dir"
  fi
  popd > /dev/null
done

echo ""
echo "✅ All runs complete. Run scripts/compare_methods.py to compare results."
