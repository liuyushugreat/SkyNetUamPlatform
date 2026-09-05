#!/usr/bin/env python3
"""Reproduce the offline evidence added for the JSS submission revision."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


MODULE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = MODULE_ROOT / "release" / "jss-submission-v1.0" / "artifact"
DEFAULT_OUTPUT_DIR = MODULE_ROOT / "outputs" / "jss-submission-v1.0"
DIRECT_DEPENDENCIES = (
    "matplotlib",
    "ortools",
    "pytest",
    "langgraph",
    "langgraph-checkpoint-sqlite",
)


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required frozen artifact is missing: {path}")
    return path


def run(command: Sequence[str], *, env: dict[str, str]) -> list[str]:
    rendered = [str(item) for item in command]
    print("+", " ".join(rendered), flush=True)
    subprocess.run(rendered, cwd=MODULE_ROOT, env=env, check=True)
    return rendered


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_archive(root: Path) -> list[dict[str, object]]:
    """Verify the archive's own SHA256SUMS and return portable input records."""

    sums = require(root / "SHA256SUMS.txt")
    records: list[dict[str, object]] = []
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        path = require(root / relative)
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(f"Frozen artifact hash mismatch: {relative}")
        records.append(
            {"path": f"outputs/SkyRescue-Bench-v1.0.0/{relative}", "sha256": observed, "bytes": path.stat().st_size}
        )
    return records


def portable_commands(
    commands: list[list[str]], *, artifact_root: Path, output_dir: Path, python: str
) -> list[list[str]]:
    replacements = (
        (str(artifact_root), "${ARTIFACT_ROOT}"),
        (str(output_dir), "${OUTPUT_DIR}"),
        (str(MODULE_ROOT), "${MODULE_ROOT}"),
    )
    portable: list[list[str]] = []
    for command in commands:
        row = []
        for item in command:
            rendered = "python" if item == python else item
            for prefix, token in replacements:
                rendered = rendered.replace(prefix, token)
            row.append(rendered)
        portable.append(row)
    return portable


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({field for row in rows for field in row})
    if "method" in fields:
        fields.remove("method")
        fields.insert(0, "method")
    if "domain" in fields:
        fields.remove("domain")
        fields.insert(0, "domain")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_manuscript_ready_outputs(output_dir: Path) -> None:
    target = output_dir / "manuscript_tables"
    figures = output_dir / "manuscript_figures"
    target.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    shutil.copy2(output_dir / "workflow" / "compiler_results.csv", target / "table_compilation.csv")
    runtime_rows = list(
        csv.DictReader((output_dir / "workflow" / "runtime_results.csv").open(encoding="utf-8", newline=""))
    )
    langgraph = json.loads(
        (output_dir / "langgraph" / "langgraph_workflow_baseline.json").read_text(encoding="utf-8")
    )["metrics"]
    runtime_rows.append({"method": "langgraph_workflow", **langgraph})
    write_csv(target / "table_runtime.csv", runtime_rows)
    shutil.copy2(
        output_dir / "runtime_latency" / "SkyRescue_runtime_benchmark_summary.csv",
        target / "table_latency.csv",
    )
    shutil.copy2(output_dir / "workflow_scale" / "workflow_scale_summary.csv", target / "table_scale.csv")
    shutil.copy2(output_dir / "heldout100" / "bootstrap_ci.csv", target / "table_heldout_bootstrap.csv")
    shutil.copy2(output_dir / "heldout100" / "risk_coverage.csv", target / "risk_coverage.csv")

    cross_domain = json.loads(
        (output_dir / "devops" / "devops_portability.json").read_text(encoding="utf-8")
    )
    devops = cross_domain
    uav = cross_domain["uav_source_domain"]
    domain_rows = []
    for domain, payload in (("devops", devops), ("uav_emergency", uav)):
        runtime = payload["runtime"]
        calls = (
            payload["observed_core_calls"]
            if "observed_core_calls" in payload
            else cross_domain["portability_contract"]["core_identity"]["domain_executions"][domain]["observed_core_calls"]
        )
        domain_rows.append(
            {
                "domain": domain,
                "admitted": payload["admission"]["admitted"],
                "structured_failures": payload["admission"]["observed_structured_failures"],
                "repaired": runtime["repaired_events"],
                "escalated": runtime["correct_escalations"],
                "duplicates_ignored": runtime["duplicates_ignored"],
                "core_admit_calls": calls["admit"],
                "core_event_calls": calls["process_event"],
                "core_closure_calls": calls["impact_closure"],
                "core_commit_calls": calls["commit_external_effect"],
                "external_invocations": runtime["external_invocations"],
                "external_effects": runtime["external_effects"],
                "stored_receipts": runtime["stored_execution_receipts"],
                "preserved_commitments": runtime["preserved_commitments"],
                "protected_commitments": runtime["protected_commitments"],
                "change_ratio": runtime["workflow_change_ratio"],
            }
        )
    write_csv(target / "table_devops.csv", domain_rows)

    for source in (
        output_dir / "heldout100" / "heldout_risk_coverage.pdf",
        output_dir / "workflow_scale" / "workflow_scale_latency.pdf",
        output_dir / "workflow_scale" / "workflow_scale_memory.pdf",
    ):
        shutil.copy2(source, figures / source.name)


def git_value(*arguments: str) -> str | None:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=MODULE_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help="directory containing SkyRescue-Bench/ and outputs/SkyRescue-Bench-v1.0.0/",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--require-clean-tag",
        action="store_true",
        help="fail unless the source tree is clean and HEAD carries jss-submission-v1.0",
    )
    args = parser.parse_args()

    artifact_root = args.artifact_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    intent_dataset = require(artifact_root / "SkyRescue-Bench" / "data" / "intent_synth_v1")
    security_dataset = require(artifact_root / "SkyRescue-Bench" / "data" / "security_challenge_v1")
    frozen_root = require(artifact_root / "outputs" / "SkyRescue-Bench-v1.0.0")
    heldout_gold = require(
        frozen_root
        / "data"
        / "entity_grounding_heldout100"
        / "SkyRescue_EntityGrounding_HeldOut100_GoldStandard_v1.0.0.jsonl"
    )
    heldout_responses = require(frozen_root / "results" / "llm_confirmatory")
    input_records = verify_frozen_archive(frozen_root)
    for directory in (intent_dataset, security_dataset):
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                input_records.append(
                    {
                        "path": str(path.relative_to(artifact_root)),
                        "sha256": sha256(path),
                        "bytes": path.stat().st_size,
                    }
                )

    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(MODULE_ROOT) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    commands: list[list[str]] = []

    source_dirty = bool(git_value("status", "--porcelain", "--untracked-files=no"))
    tags_at_head = (git_value("tag", "--points-at", "HEAD") or "").splitlines()
    if args.require_clean_tag and (source_dirty or "jss-submission-v1.0" not in tags_at_head):
        raise RuntimeError("--require-clean-tag needs a clean tree at tag jss-submission-v1.0")

    commands.append(run([args.python, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"], env=env))
    commands.append(run([args.python, "scripts/verify_entity_grounding_freeze.py"], env=env))
    commands.append(
        run(
            [
                args.python,
                "scripts/run_workflow_benchmark.py",
                "--dataset",
                str(intent_dataset),
                "--output-dir",
                str(output_dir / "workflow"),
            ],
            env=env,
        )
    )
    commands.append(
        run(
            [
                args.python,
                "scripts/run_langgraph_baseline.py",
                "--dataset",
                str(intent_dataset),
                "--output-dir",
                str(output_dir / "langgraph"),
            ],
            env=env,
        )
    )
    heldout_output = output_dir / "heldout100"
    commands.append(
        run(
            [
                args.python,
                "scripts/score_heldout_llm_blind.py",
                "--gold",
                str(heldout_gold),
                "--response-dir",
                str(heldout_responses),
                "--output-dir",
                str(heldout_output),
                "--bootstrap-iterations",
                "10000",
                "--bootstrap-seed",
                "20260905",
            ],
            env=env,
        )
    )
    commands.append(
        run(
            [
                args.python,
                "scripts/plot_risk_coverage.py",
                "--input",
                str(heldout_output / "risk_coverage.csv"),
                "--output-dir",
                str(heldout_output),
            ],
            env=env,
        )
    )
    commands.append(
        run(
            [
                args.python,
                "scripts/run_crash_recovery_experiment.py",
                "--trials",
                "30",
                "--output-dir",
                str(output_dir / "crash_recovery"),
            ],
            env=env,
        )
    )
    commands.append(
        run(
            [
                args.python,
                "scripts/run_runtime_latency_benchmark.py",
                "--intent-dataset",
                str(intent_dataset),
                "--security-dataset",
                str(security_dataset),
                "--output-dir",
                str(output_dir / "runtime_latency"),
                "--warmup-rounds",
                "5",
                "--repeats",
                "30",
            ],
            env=env,
        )
    )
    commands.append(
        run(
            [
                args.python,
                "scripts/run_workflow_scale.py",
                "--intent-dataset",
                str(intent_dataset),
                "--output-dir",
                str(output_dir / "workflow_scale"),
            ],
            env=env,
        )
    )
    commands.append(
        run(
            [
                args.python,
                "scripts/run_devops_portability.py",
                "--seed",
                "20260905",
                "--instructions",
                "60",
                "--events",
                "60",
                "--output",
                str(output_dir / "devops" / "devops_portability.json"),
            ],
            env=env,
        )
    )

    build_manuscript_ready_outputs(output_dir)

    generated = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "reproduction_manifest.json"
    )
    manifest = {
        "release": "jss-submission-v1.0",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": git_value("rev-parse", "HEAD"),
        "repository_tags_at_head": tags_at_head,
        "source_tree_dirty": source_dirty,
        "python": sys.version,
        "platform": platform.platform(),
        "direct_dependencies": {
            package: importlib.metadata.version(package) for package in DIRECT_DEPENDENCIES
        },
        "artifact_root": "${ARTIFACT_ROOT}",
        "input_artifacts": input_records,
        "commands": portable_commands(
            commands, artifact_root=artifact_root, output_dir=output_dir, python=args.python
        ),
        "outputs": [
            {
                "path": str(path.relative_to(output_dir)),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in generated
        ],
        "network_calls": 0,
        "seeds": {
            "intent_synth": 20260802,
            "security_challenge": 20261001,
            "bootstrap": 20260905,
            "devops": 20260905,
            "uav_adapter": 20260905,
            "scale": [20260811, 20260812, 20260813, 20260814, 20260815],
        },
    }
    manifest_path = output_dir / "reproduction_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Reproduction complete: {manifest_path}")


if __name__ == "__main__":
    main()
