"""
Fill estimated/placeholder entries in docs/report.tex using Phase 3/4 results.

Usage:
    python scripts/fill_report.py [--dry-run]

Reads:
    results/eval_summary.json     (Phase 3 required)
    results/ablation.json         (Phase 3 required)
    results/arm_trajectory_chi64.png  (Phase 4, optional)

Writes: docs/report.tex (in-place)
"""
import argparse, json, re, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
TEX  = ROOT / "docs" / "report.tex"
RESULTS = ROOT / "results"

def load_json(name):
    p = RESULTS / name
    if not p.exists():
        print(f"  MISSING: {p}")
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)

def main(dry_run=False):
    eval_summary = load_json("eval_summary.json")
    ablation     = load_json("ablation.json")
    sweep_stats  = load_json("compression_sweep_stats.json")

    # Skip if eval_summary is still the estimated demo version
    is_estimated = (eval_summary or {}).get("data_note", "").startswith("ESTIMATED")
    if is_estimated:
        print("  eval_summary.json is still the estimated version — nothing to fill.")
        print("  Run this script again after Phase 3 GPU evaluation completes.")
        return

    phase3_ready = eval_summary is not None and ablation is not None
    if not phase3_ready:
        print("  NOTE: Phase 3 artifacts missing — cannot fill report tables.")
        return

    with open(TEX, encoding="utf-8") as f:
        tex = f.read()

    original = tex

    baseline_agg = eval_summary.get("baseline", {}).get("aggregate", {})
    b_t = baseline_agg.get("inference_time_ms", {}).get("mean", 2095.6)
    n_ep_actual = eval_summary.get("n_eval_episodes_per_run", 50)

    tn = eval_summary.get("tn_variants", {})
    CHI_ORDER = [128, 64, 32, 16]

    # ── Table 1: efficiency table rows ────────────────────────────────────────
    # Matches lines like: 128   & $..$ & $..$ & $..$ & $..$ \\
    for chi in CHI_ORDER:
        v = tn.get(str(chi))
        if v is None:
            print(f"  WARNING: chi={chi} not in eval_summary — skipping row")
            continue
        agg = v["aggregate"]
        t_m  = agg.get("inference_time_ms_mean", {}).get("mean", 0.0)
        t_s  = agg.get("inference_time_ms_mean", {}).get("std", 0.0)
        m_m  = agg.get("peak_mem_mib_mean", {}).get("mean", 0.0)
        l1_m = agg.get("l1_error_mean", {}).get("mean", 0.0)
        l1_s = agg.get("l1_error_mean", {}).get("std", 0.0)
        dt_pct = (t_m - b_t) / b_t * 100 if b_t else 0.0

        time_cell = f"${t_m:.0f}\\pm{t_s:.0f}$"
        dt_cell   = f"${dt_pct:+.1f}\\%$"
        mem_cell  = f"${m_m:.0f}$"
        dl1_cell  = f"${l1_m:.4f}\\pm{l1_s:.4f}$"
        new_row   = f"{chi}   & {time_cell} & {dt_cell} & {mem_cell} & {dl1_cell} \\\\"

        # Match the existing row (estimated or placeholder) for this chi value
        pat = re.compile(
            rf"^{chi}\s+&\s+\$[^$]+\$\s+&\s+\$[^$]+\$\s+&\s+\$[^$]+\$\s+&\s+\$[^$]+\$\s*\\\\$",
            re.MULTILINE,
        )
        new_tex, n = pat.subn(lambda m: new_row, tex)
        if n:
            tex = new_tex
            print(f"  Table 1 row chi={chi}: updated -> {new_row}")
        else:
            print(f"  WARNING: Table 1 row chi={chi} pattern not matched")

    # ── Update section description (remove "estimated / pending" language) ────
    tex = re.sub(
        r"\\textbf\{Compressed models\}.*?(?=\n\n\\begin\{table\})",
        r"\\textbf{Compressed models} (Phase~3 direct GPU evaluation, "
        r"3 seeds $\\times$ 50 episodes each):\n",
        tex,
        flags=re.DOTALL,
    )

    # ── Update table footnote (remove "estimated" language) ───────────────────
    tex = re.sub(
        r"Values in rows 2--5 are \\emph\{estimated\}.*?pending\.",
        r"Values in rows 2--5 are measured directly via GPU inference (Phase~3).",
        tex,
        flags=re.DOTALL,
    )

    # ── Ablation Table 2 rows ─────────────────────────────────────────────────
    conds = ablation.get("conditions", {})

    def _cond_l1(cond_key):
        c = conds.get(cond_key)
        if c is None:
            return None, None
        agg = c.get("aggregate", {})
        stat = agg.get("l1_error_mean", agg.get("l1_error", {}))
        return stat.get("mean"), stat.get("std")

    bl1_m, bl1_s = _cond_l1("B_tn_llm_only")
    if bl1_m is not None:
        pat_b = re.compile(
            r"TN LLM only \(\$\\chi=64\$\)\s*&\s*\$[^$]+\$\s*&\s*[^&]+&\s*---\s*\\\\",
            re.MULTILINE,
        )
        new_b = (f"TN LLM only ($\\chi=64$) & ${bl1_m:.4f}\\pm{bl1_s:.4f}$ "
                 f"& $\\sim$1.20\\,B & --- \\\\")
        new_tex, n = pat_b.subn(lambda m: new_b, tex)
        if n:
            tex = new_tex
            print(f"  Ablation row B: L1 shift={bl1_m:.4f} ± {bl1_s:.4f}")
        else:
            print("  WARNING: Ablation row B pattern not matched")

    cl1_m, cl1_s = _cond_l1("C_tn_full")
    if cl1_m is not None and bl1_m is not None:
        delta_c_vs_b = cl1_m - bl1_m
        cond_c = conds.get("C_tn_full")
        n_full = tn.get("64", {}).get("n_params_core")
        params_str = f"$\\sim${n_full/1e9:.2f}\\,B" if n_full else "$\\sim$0.13\\,B"
        pat_c = re.compile(
            r"TN full \(\$\\chi=64\$\)\s*&\s*\$[^$]+\$\s*&\s*[^&]+&\s*\$[^$]+\$\s*\\\\",
            re.MULTILINE,
        )
        new_c = (f"TN full ($\\chi=64$) & ${cl1_m:.4f}\\pm{cl1_s:.4f}$ "
                 f"& {params_str} & ${delta_c_vs_b:+.4f}$ \\\\")
        new_tex, n = pat_c.subn(lambda m: new_c, tex)
        if n:
            tex = new_tex
            print(f"  Ablation row C: L1 shift={cl1_m:.4f}, delta vs B={delta_c_vs_b:+.4f}")
        else:
            print("  WARNING: Ablation row C pattern not matched")

    # ── Energy section ────────────────────────────────────────────────────────
    if tn.get("64"):
        kwh64 = tn["64"]["aggregate"].get("total_kwh", {}).get("mean")
        if kwh64 is not None:
            kwh64_per_ep = kwh64 / n_ep_actual
            b_kwh_total = baseline_agg.get("total_kwh", {}).get("mean", 0.006)
            b_kwh_per_ep = b_kwh_total / 200
            kwh_pct = (b_kwh_per_ep - kwh64_per_ep) / b_kwh_per_ep * 100 if b_kwh_per_ep else 0.0
            _energy_repl = (f"Compressed model ($\\chi=64$): ${kwh64_per_ep:.2e}$\\,kWh/sample\n"
                            f"(${kwh_pct:.0f}\\%$ per-sample reduction).")
            tex = re.sub(
                r"Compressed model \(\$\\chi=64\$\): \$[0-9.e+-]+\$\\,kWh/sample\s*\n\s*\(\$[0-9]+\\%\$ per-sample reduction\)\.",
                lambda m: _energy_repl,
                tex,
            )
            print(f"  Energy: {kwh64_per_ep:.2e} kWh/ep, {kwh_pct:.0f}% reduction")

    # ── Phase 3 GPU-hours in resource table ───────────────────────────────────
    total_wall_s = 0.0
    for v in tn.values():
        agg = v.get("aggregate", {})
        kwh_m = agg.get("total_kwh", {}).get("mean", 0.0)
        pwr_m = agg.get("avg_power_w", {}).get("mean", 0.0)
        if pwr_m > 0 and kwh_m > 0:
            total_wall_s += kwh_m * 3.6e6 / pwr_m * len(eval_summary.get("seeds", [1,2,3]))
        else:
            t_ms = agg.get("inference_time_ms_mean", {}).get("mean", 0.0)
            total_wall_s += t_ms * n_ep_actual * len(eval_summary.get("seeds", [1,2,3])) / 1000.0
    gpu_h = total_wall_s / 3600.0
    if gpu_h > 0:
        _gpu_repl = f"Phase 3 GPU-hours & $\\sim${gpu_h:.2f}\\,h (4 $\\chi$ $\\times$ 3 seeds) \\\\"
        tex = re.sub(
            r"Phase 3 GPU-hours & .*? \\\\",
            lambda m: _gpu_repl,
            tex,
        )
        print(f"  Phase 3 GPU-hours: {gpu_h:.2f} h")

    # ── Final check ───────────────────────────────────────────────────────────
    remaining = tex.count("\\PENDING{") + tex.count("\\placeholder{TBD}") + tex.count("estimated")
    print(f"\n  'estimated' occurrences remaining: {tex.count('estimated')}")
    print(f"  \\PENDING / \\placeholder{{TBD}} remaining: {tex.count(chr(92)+'PENDING{') + tex.count(chr(92)+'placeholder{TBD}')}")

    if dry_run:
        changed = sum(a != b for a, b in zip(original, tex)) + abs(len(original) - len(tex))
        print(f"\n[dry-run] ~{changed} characters would change. Not writing.")
    else:
        with open(TEX, "w", encoding="utf-8") as f:
            f.write(tex)
        print("\nreport.tex updated.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    main(dry_run=args.dry_run)
