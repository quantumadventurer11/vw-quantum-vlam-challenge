#!/usr/bin/env bash
# scripts/kaggle_run_quickeval.sh
# Push Phase 3 quick-eval notebook (1 seed × 20 episodes, χ∈{16,128}) to Kaggle,
# poll until done, pull output.
#
# Phase 2 checkpoints (cores.pt) are mounted via kernel_sources from
# benjaminbrumm/vw-phase2-compression (its /kaggle/working/ output).
# Path inside kernel: /kaggle/input/vw-phase2-compression/checkpoints/
#
# Prerequisites:
#   1. kaggle CLI authenticated (~/.kaggle/kaggle.json)
#   2. Phase 2 kernel (benjaminbrumm/vw-phase2-compression) must have a
#      successful completed run for kernel_sources to be mountable.
#
# Expected runtime: ~30-40 min (FP16 model load ~20 min + 2 chi × 20 eps)

set -euo pipefail

KERNEL_ID="benjaminbrumm/vw-phase3-quickeval"
OUTPUT_DIR="kaggle_output_quickeval"
MAX_WAIT_S=7200      # 2-hour hard timeout
POLL_INTERVAL_S=60

# ── Helpers ───────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }

get_status() {
    local raw
    raw=$(PYTHONUTF8=1 kaggle kernels status "$KERNEL_ID" 2>/dev/null)
    echo "$raw" | grep -oP 'KernelWorkerStatus\.\K[A-Z_]+' | tr '[:upper:]' '[:lower:]'
}

extract_errors() {
    local nb_file="$1"
    python - "$nb_file" <<'PYEOF'
import json, sys, re
try:
    nb = json.load(open(sys.argv[1], encoding='utf-8'))
except Exception as e:
    print(f"Could not parse notebook: {e}")
    sys.exit(0)
ansi = re.compile(r'\x1b\[[0-9;]*[mK]')
printed = 0
for cell in nb.get('cells', []):
    cid = cell.get('id', '?')
    for out in cell.get('outputs', []):
        otype = out.get('output_type', '')
        if otype == 'error':
            printed += 1
            print(f"\n{'─'*60}")
            print(f"CELL {cid!r}  ERROR: {out.get('ename')} — {out.get('evalue')}")
            for line in out.get('traceback', [])[-15:]:
                print(ansi.sub('', line))
        elif otype == 'stream' and 'text' in out:
            text = ''.join(out['text'])
            if 'error' in text.lower() or 'traceback' in text.lower() or 'exception' in text.lower():
                printed += 1
                print(f"\n{'─'*60}")
                print(f"CELL {cid!r}  STDERR excerpt:")
                print(text[-1200:])
if printed == 0:
    print("(no error outputs found in notebook)")
PYEOF
}

# ── Stage push directory ───────────────────────────────────────────────────────
STAGE_DIR=$(mktemp -d)
trap 'rm -rf "$STAGE_DIR"' EXIT

cp notebooks/phase3_quickeval.ipynb "$STAGE_DIR/"

cat > "$STAGE_DIR/kernel-metadata.json" <<'METADATA'
{
  "id": "benjaminbrumm/vw-phase3-quickeval",
  "code_file": "phase3_quickeval.ipynb",
  "enable_gpu": true,
  "is_private": true,
  "kernel_sources": ["benjaminbrumm/vw-phase2-compression"]
}
METADATA

# ── Push ──────────────────────────────────────────────────────────────────────
log "Pushing quick-eval kernel to Kaggle (kernel: $KERNEL_ID)..."
log "  Config: 1 seed × 20 episodes × chi∈{16,128}"
log "  Phase 2 cores.pt mounted from: benjaminbrumm/vw-phase2-compression"
PYTHONUTF8=1 kaggle kernels push -p "$STAGE_DIR"
log "Pushed. Waiting 90s for Kaggle to register the new run..."
sleep 90
log "Starting poll every ${POLL_INTERVAL_S}s (max ${MAX_WAIT_S}s)..."
echo ""

# ── Poll ──────────────────────────────────────────────────────────────────────
elapsed=90
while [[ $elapsed -lt $MAX_WAIT_S ]]; do
    status=$(get_status)
    log "Status: ${status:-<empty>}  (${elapsed}s elapsed)"

    case "$status" in
        "complete")
            log "Quick-eval kernel completed successfully!"
            echo ""

            mkdir -p "$OUTPUT_DIR"
            log "Pulling output to $OUTPUT_DIR/ ..."
            PYTHONUTF8=1 kaggle kernels output "$KERNEL_ID" -p "$OUTPUT_DIR"
            echo ""

            # ── Copy sanity result ────────────────────────────────────────────
            mkdir -p results
            found=$(find "$OUTPUT_DIR" -name "quickeval_sanity.json" | head -1)
            if [[ -n "$found" ]]; then
                cp "$found" "results/quickeval_sanity.json"
                log "Copied → results/quickeval_sanity.json"
                echo ""
                log "=== QUICK-EVAL RESULTS ==="
                python - "results/quickeval_sanity.json" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
bl1 = d["baseline_l1"]
bstd = d["baseline_l1_std"]
print(f"Baseline (Phase 1 FP16): L1 = {bl1:.4f} ± {bstd:.4f}")
print()
for chi_str, r in sorted(d["results"].items(), key=lambda x: int(x[0])):
    chi = int(chi_str)
    l1 = r["l1_error_mean"]
    std = r.get("l1_error_std", 0)
    delta = l1 - bl1
    pct = 100 * delta / max(bl1, 1e-9)
    flag = "OK" if abs(pct) < 50 else ("DEGRADED" if abs(pct) < 200 else "CATASTROPHIC")
    print(f"  chi={chi:3d}: L1 = {l1:.4f} ± {std:.4f}  delta = {delta:+.4f} ({pct:+.1f}%)  [{flag}]")
print()
worst_pct = max(abs(100*(r["l1_error_mean"]-bl1)/max(bl1,1e-9)) for r in d["results"].values())
if worst_pct < 50:
    print("GO: proceed to full Phase 3 sweep (bash scripts/kaggle_run_phase3.sh)")
elif worst_pct < 200:
    print("CAUTION: flag results to Vidhi before committing to full sweep")
else:
    print("NO-GO: catastrophic degradation — healing/fine-tune needed before full sweep")
PYEOF
            else
                log "WARNING: quickeval_sanity.json not found in output."
                log "Check $OUTPUT_DIR for notebook output."
            fi

            echo ""
            log "=== Commit quick-eval results: ==="
            echo "  git add results/quickeval_sanity.json"
            echo "  git commit -m '[phase3] add quick-eval sanity check (1 seed x 20 eps, chi=16,128)'"
            exit 0
            ;;

        "error"|"cancelAcknowledged"|"cancelled")
            log "KERNEL FAILED with status: $status"
            echo ""

            mkdir -p "$OUTPUT_DIR"
            log "Pulling output for post-mortem..."
            PYTHONUTF8=1 kaggle kernels output "$KERNEL_ID" -p "$OUTPUT_DIR" || true

            NB_OUT=$(find "$OUTPUT_DIR" -name "*.ipynb" | head -1)
            if [[ -n "$NB_OUT" ]]; then
                log "Extracting errors from executed notebook: $NB_OUT"
                extract_errors "$NB_OUT"
            else
                log "No executed notebook in output — check Kaggle UI for logs."
            fi
            exit 1
            ;;

        "queued"|"running"|"")
            sleep "$POLL_INTERVAL_S"
            elapsed=$((elapsed + POLL_INTERVAL_S))
            ;;

        *)
            log "Unexpected status: '$status' — continuing to poll..."
            sleep "$POLL_INTERVAL_S"
            elapsed=$((elapsed + POLL_INTERVAL_S))
            ;;
    esac
done

log "Timed out after ${MAX_WAIT_S}s — check Kaggle UI manually."
exit 2
