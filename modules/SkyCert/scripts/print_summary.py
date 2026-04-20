"""Pretty-print the multi-seed aggregated metrics for paper tables."""
import json
import sys
from pathlib import Path

out_dir = Path("outputs")
d = json.load(open(out_dir / "multi_seed.json"))
ext = json.load(open(out_dir / "extensions.json"))


def fmt(agg, key):
    v = agg.get(key, {})
    return f"{v.get('mean', float('nan')):.3f}+-{v.get('std', float('nan')):.3f}"


print("=== Table 1: Main experiment (5 seeds) ===")
print(f"{'scenario':22s}  cov          |C|         crit_base    crit_after   abst         CRIT_cov     FN           FP")
for sc, agg in d["main"].items():
    print(f"{sc:22s}  "
          f"{fmt(agg,'coverage')}  "
          f"{fmt(agg,'average_set_size')}  "
          f"{fmt(agg,'critical_error_rate_base')}  "
          f"{fmt(agg,'critical_error_rate_after_abstain')}  "
          f"{fmt(agg,'abstain_rate')}  "
          f"{fmt(agg,'per_class_coverage.class_3')}  "
          f"{fmt(agg,'fp_fn.critical_fn')}  "
          f"{fmt(agg,'fp_fn.critical_fp')}")

print()
print("=== Table 2: Ablation under distribution_shift (5 seeds) ===")
print(f"{'variant':15s}  cov          |C|         crit_after   abst")
for v, agg in d["ablation"].items():
    print(f"{v:15s}  {fmt(agg,'coverage')}  {fmt(agg,'avg_set_size')}  "
          f"{fmt(agg,'critical_error_rate_after_abstain')}  "
          f"{fmt(agg,'abstain_rate')}")

print()
print("=== Table 3: Baseline comparison (5 seeds, matched abstention) ===")
print(f"{'method':22s}  crit_after   abst")
for m, agg in d["baselines"].items():
    print(f"{m:22s}  {fmt(agg,'critical_error_rate_after_abstain')}  "
          f"{fmt(agg,'abstain_rate')}")

print()
print("=== Appendix: Detection delay + martingale peak (5 seeds) ===")
print(f"{'scenario':22s}  detect_delay  mart_max_max   false_alarm")
for sc, agg in d["main"].items():
    dd = agg.get("detection.detection_delay", {})
    mm = agg.get("martingale_max", {})
    fa = agg.get("detection.false_alarm_rate", {})
    print(f"{sc:22s}  "
          f"mean={dd.get('mean', float('nan')):.1f}+-{dd.get('std', float('nan')):.1f}  "
          f"max={mm.get('max', float('nan')):.2e}  "
          f"mean={fa.get('mean', float('nan')):.4f}")

print()
print("=== Lambda sweep (distribution_shift) ===")
print(f"{'lambda_drift':12s}  cov          |C|         crit_after   abst")
for r in ext["lambda_sweep"]:
    print(f"{r['lambda_drift']:<12.1f}  "
          f"{r['coverage']:.3f}        "
          f"{r['avg_set_size']:.2f}          "
          f"{r['critical_error_rate_after_abstain']:.3f}        "
          f"{r['abstain_rate']:.3f}")

print()
print("=== Attack-strength sweep beta4 (covariate_shift) ===")
print(f"{'strength':10s}  cov          crit_base   crit_after   abst")
for r in ext["attack_strength_sweep"]["beta4"]:
    print(f"{r['strength']:<10.2f}  "
          f"{r['coverage']:.3f}        "
          f"{r['critical_error_rate_base']:.3f}        "
          f"{r['critical_error_rate_after_abstain']:.3f}        "
          f"{r['abstain_rate']:.3f}")

print()
print("=== Attack-strength sweep beta3 (feature_attack) ===")
print(f"{'strength':10s}  cov          crit_base   crit_after   abst")
for r in ext["attack_strength_sweep"]["beta3"]:
    print(f"{r['strength']:<10.2f}  "
          f"{r['coverage']:.3f}        "
          f"{r['critical_error_rate_base']:.3f}        "
          f"{r['critical_error_rate_after_abstain']:.3f}        "
          f"{r['abstain_rate']:.3f}")

print()
print("=== MLP backbone replication ===")
print(f"{'scenario':22s}  cov         |C|         crit_base   crit_after   abst")
for r in ext["mlp_backbone"]:
    print(f"{r['scenario']:22s}  "
          f"{r['coverage']:.3f}       "
          f"{r['avg_set_size']:.2f}         "
          f"{r['critical_error_rate_base']:.3f}        "
          f"{r['critical_error_rate_after_abstain']:.3f}        "
          f"{r['abstain_rate']:.3f}")

print()
print("=== Failure cases ===")
for rec in ext["failure_cases"]:
    print(f"idx={rec['sample_id']:4d} kind={rec['decision_kind']:<9s} "
          f"true=3 pred={rec['predicted_label']} set={rec['prediction_set']} "
          f"mart={rec['martingale_value']:.2e}")
