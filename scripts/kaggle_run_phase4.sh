#!/usr/bin/env bash
# scripts/kaggle_run_phase4.sh
# Push Phase 4 MuJoCo visualization notebook to Kaggle, poll until done, pull output.
#
# Phase 2 checkpoints (cores.pt) are mounted via kernel_sources from
# benjaminbrumm/vw-phase2-compression (its /kaggle/working/ output).
# Path in Phase 4: /kaggle/input/vw-phase2-compression/checkpoints/
#
# Prerequisites:
#   1. kaggle CLI authenticated (~/.kaggle/kaggle.json)
#   2. Phase 2 kernel (benjaminbrumm/vw-phase2-compression) must have a
#      successful completed run for kernel_sources to be mountable.

set -euo pipefail

KERNEL_ID="benjaminbrumm/vw-phase4-visualization"
OUTPUT_DIR="kaggle_output_phase4"
MAX_WAIT_S=14400     # 4-hour hard timeout (2× 200-step episodes + model load)
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

cp notebooks/phase4_mujoco_viz.ipynb "$STAGE_DIR/"

cat > "$STAGE_DIR/kernel-metadata.json" <<'METADATA'
{
  "id": "benjaminbrumm/vw-phase4-visualization",
  "code_file": "phase4_mujoco_viz.ipynb",
  "enable_gpu": true,
  "is_private": true,
  "kernel_sources": ["benjaminbrumm/vw-phase2-compression"]
}
METADATA

# ── Push ──────────────────────────────────────────────────────────────────────
log "Pushing Phase 4 kernel to Kaggle (kernel: $KERNEL_ID)..."
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
            log "Phase 4 kernel completed successfully!"
            echo ""

            mkdir -p "$OUTPUT_DIR"
            log "Pulling output to $OUTPUT_DIR/ ..."
            PYTHONUTF8=1 kaggle kernels output "$KERNEL_ID" -p "$OUTPUT_DIR"
            echo ""

            # ── Copy result artifacts ─────────────────────────────────────────
            mkdir -p results
            for artifact in \
                "demo_compressed_chi64.mp4" \
                "demo_compressed_chi64.gif" \
                "demo_baseline_fp16.mp4" \
                "demo_baseline_fp16.gif" \
                "demo_side_by_side_chi64.mp4" \
                "demo_side_by_side_chi64.gif" \
                "arm_trajectory_chi64.png"
            do
                found=$(find "$OUTPUT_DIR" -name "$artifact" | head -1)
                if [[ -n "$found" ]]; then
                    cp "$found" "results/$artifact"
                    log "Copied → results/$artifact"
                else
                    log "WARNING: $artifact not found in output."
                fi
            done

            echo ""
            log "=== Phase 4 complete. Commit the results: ==="
            echo "  git add -f results/demo_compressed_chi64.mp4 results/demo_compressed_chi64.gif"
            echo "  git add -f results/demo_baseline_fp16.mp4 results/demo_baseline_fp16.gif"
            echo "  git add -f results/demo_side_by_side_chi64.mp4 results/demo_side_by_side_chi64.gif"
            echo "  git add results/arm_trajectory_chi64.png"
            echo "  git commit -m '[phase4] add MuJoCo 3D visualization chi=64'"
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
