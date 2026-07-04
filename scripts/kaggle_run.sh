#!/usr/bin/env bash
# scripts/kaggle_run.sh
# Push phase1_baseline.ipynb to Kaggle, poll until done, pull output,
# copy artifacts to results/ and configs/, then print a commit command.
#
# Prerequisites:
#   1. kaggle CLI installed and authenticated (kaggle.json or ~/.kaggle/access_token)
#   2. On your Kaggle kernel page, add two secrets:
#        GH_TOKEN   = GitHub PAT with repo-read scope
#        HF_TOKEN   = HuggingFace token with openvla/openvla-7b read access
#   3. GPU must be enabled in kernel settings (done via kernel-metadata.json)
#
# Usage:
#   bash scripts/kaggle_run.sh
#
# The script commits artifacts automatically when the run succeeds.

set -euo pipefail

KERNEL_ID="benjaminbrumm/vw-phase1-baseline"
META_DIR="notebooks"           # directory containing kernel-metadata.json + notebook
OUTPUT_DIR="kaggle_output"     # where pulled output lands locally
MAX_WAIT_S=7200                # 2-hour hard timeout
POLL_INTERVAL_S=30

# ── Helpers ───────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }

get_status() {
    # kaggle kernels status output (API v2):
    #   benjaminbrumm/vw-phase1-baseline has status "KernelWorkerStatus.RUNNING"
    # Extract the bare status token after the last dot.
    local raw
    raw=$(PYTHONUTF8=1 kaggle kernels status "$KERNEL_ID" 2>/dev/null)
    # e.g. "KernelWorkerStatus.COMPLETE" → "COMPLETE"
    echo "$raw" | grep -oP 'KernelWorkerStatus\.\K[A-Z_]+' | tr '[:upper:]' '[:lower:]'
}

extract_errors() {
    local nb_file="$1"
    python3 - "$nb_file" <<'PYEOF'
import json, sys, re

try:
    nb = json.load(open(sys.argv[1]))
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
            if 'error' in text.lower() or 'traceback' in text.lower():
                printed += 1
                print(f"\n{'─'*60}")
                print(f"CELL {cid!r}  STDERR excerpt:")
                print(text[-800:])

if printed == 0:
    print("(no error outputs found in notebook)")
PYEOF
}

# ── Push ──────────────────────────────────────────────────────────────────────
log "Pushing kernel from '$META_DIR/' to Kaggle..."
PYTHONUTF8=1 kaggle kernels push -p "$META_DIR"
log "Pushed. Polling every ${POLL_INTERVAL_S}s (max ${MAX_WAIT_S}s)..."
echo ""

# ── Poll ──────────────────────────────────────────────────────────────────────
elapsed=0
while [[ $elapsed -lt $MAX_WAIT_S ]]; do
    status=$(get_status)
    log "Status: ${status:-<empty>}  (${elapsed}s elapsed)"

    case "$status" in
        # ── SUCCESS ───────────────────────────────────────────────────────────
        "complete")
            log "Kernel completed successfully!"
            echo ""

            mkdir -p "$OUTPUT_DIR"
            log "Pulling output to $OUTPUT_DIR/ ..."
            PYTHONUTF8=1 kaggle kernels output "$KERNEL_ID" -p "$OUTPUT_DIR"
            echo ""
            log "Output files:"
            find "$OUTPUT_DIR" -type f | sort | while read -r f; do
                printf "  %s  (%s)\n" "$f" "$(du -sh "$f" | cut -f1)"
            done

            # ── Copy artifacts ────────────────────────────────────────────────
            echo ""
            METRICS=$(find "$OUTPUT_DIR" -name "baseline_metrics.json" | head -1)
            HWPROFILE=$(find "$OUTPUT_DIR" -name "hardware_profile.yaml" | head -1)

            if [[ -n "$METRICS" ]]; then
                mkdir -p results
                cp "$METRICS" results/baseline_metrics.json
                log "Copied → results/baseline_metrics.json"
            else
                log "WARNING: baseline_metrics.json not found in output."
            fi

            if [[ -n "$HWPROFILE" ]]; then
                mkdir -p configs
                cp "$HWPROFILE" configs/hardware_profile.yaml
                log "Copied → configs/hardware_profile.yaml"
            else
                log "WARNING: hardware_profile.yaml not found in output."
            fi

            # ── Print summary from metrics ────────────────────────────────────
            if [[ -n "$METRICS" ]]; then
                echo ""
                log "=== Baseline Metrics Summary ==="
                python3 - "$METRICS" <<'PYEOF'
import json, sys
m = json.load(open(sys.argv[1]))
agg = m.get('aggregate', {})
hw  = m.get('hardware', {})
print(f"  GPU              : {hw.get('gpu_name', '?')}")
print(f"  Params           : {hw.get('n_params_b', '?')}B")
print(f"  Peak mem (load)  : {hw.get('peak_mem_load_mib', '?')} MiB")
print()
for key in ['l1_error', 'inference_time_ms', 'peak_mem_mib', 'avg_power_w', 'total_kwh']:
    v = agg.get(key, {})
    if v:
        print(f"  {key:<24} mean={v.get('mean',0):.4g}  std={v.get('std',0):.4g}  (n={v.get('n_runs','?')})")
PYEOF
            fi

            # ── Commit prompt ─────────────────────────────────────────────────
            echo ""
            log "Run this to commit the artifacts:"
            echo "  git add results/baseline_metrics.json configs/hardware_profile.yaml"
            echo "  git commit -m '[phase1] commit baseline artifacts from Kaggle run'"
            exit 0
            ;;

        # ── FAILURE ───────────────────────────────────────────────────────────
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

        # ── STILL RUNNING ─────────────────────────────────────────────────────
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
