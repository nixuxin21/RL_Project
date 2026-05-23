"""Run reproducible paper-grade experiment presets for execution-mismatch AirComp."""

import argparse
from datetime import UTC, datetime
import hashlib
import itertools
import json
from pathlib import Path
import shlex
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "experiments" / "paper_experiment_presets.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "paper_experiment_runs"
RUNNER_ONLY_KEYS = {
    "axes",
    "description",
    "method_group",
    "methods",
}


def stable_json(data):
    """Serialize config-like data in a deterministic form for hashing/snapshots."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


def short_hash(data, length=10):
    """Return a short content hash for run and job directories."""
    return hashlib.sha1(stable_json(data).encode("utf-8")).hexdigest()[:length]


def load_config(path):
    """Load the preset JSON file."""
    with open(path, encoding="utf-8") as jsonfile:
        return json.load(jsonfile)


def parse_csv_items(value):
    """Parse comma-separated command-line values."""
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_override(value):
    """Parse one key=value override for evaluator arguments."""
    if "=" not in value:
        raise ValueError(f"override must be key=value, got {value!r}")
    key, raw_value = value.split("=", 1)
    key = key.strip().replace("-", "_")
    raw_value = raw_value.strip()
    if raw_value.lower() == "true":
        return key, True
    if raw_value.lower() == "false":
        return key, False
    return key, raw_value


def as_list(value):
    """Normalize scalar or list config values to a list."""
    if isinstance(value, list):
        return value
    return [value]


def cli_value(value):
    """Render a config value as one CLI argument value."""
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)


def cli_flag(key):
    """Convert snake_case config keys to argparse long options."""
    return "--" + key.replace("_", "-")


def resolve_methods(config, preset):
    """Resolve user-facing suite method names to evaluator policy aliases."""
    if "methods" in preset:
        method_names = list(preset["methods"])
    else:
        group = preset.get("method_group", "paper_core")
        method_names = list(config["method_groups"][group])
    aliases = config.get("method_aliases", {})
    return [aliases.get(name, name) for name in method_names]


def expanded_axis_records(config, preset):
    """Expand one preset's scenario axes into concrete evaluator argument records."""
    axes = preset.get("axes", {})
    axis_names = list(axes)
    axis_values = [as_list(axes[name]) for name in axis_names]
    if not axis_names:
        yield {}, {}
        return
    feedback_map = config.get("feedback_noise_std", {})
    for values in itertools.product(*axis_values):
        evaluator_args = {}
        labels = {}
        for name, value in zip(axis_names, values):
            if name == "feedback_noise":
                if value not in feedback_map:
                    raise ValueError(f"unknown feedback noise label {value!r}")
                evaluator_args["confirmation_feedback_noise_std"] = feedback_map[value]
                labels["feedback_noise"] = value
            else:
                evaluator_args[name] = value
                labels[name] = value
        yield evaluator_args, labels


def parse_budget_list(value):
    """Parse comma-separated probe budgets from a preset value."""
    return [int(item) for item in parse_csv_items(value)]


def filter_probe_budgets(evaluator_args):
    """Drop invalid B > C entries explicitly instead of letting the evaluator fail late."""
    if "probe_budgets" not in evaluator_args or "num_codebook_states" not in evaluator_args:
        return None
    codebook_size = int(evaluator_args["num_codebook_states"])
    raw_budgets = parse_budget_list(evaluator_args["probe_budgets"])
    valid_budgets = [budget for budget in raw_budgets if budget <= codebook_size]
    if not valid_budgets:
        raise ValueError(
            f"all probe budgets {raw_budgets} exceed codebook size C={codebook_size}"
        )
    evaluator_args["probe_budgets"] = ",".join(str(budget) for budget in valid_budgets)
    if valid_budgets != raw_budgets:
        return f"filtered probe_budgets {raw_budgets} to {valid_budgets} for C={codebook_size}"
    return None


def safe_label(value):
    """Render a compact filesystem-safe label fragment."""
    text = str(value).replace(",", "-").replace(".", "p")
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)


def job_label(labels):
    """Build a short human-readable job label from the main scenario axes."""
    parts = []
    aliases = {
        "num_nodes": "K",
        "num_slots": "S",
        "num_codebook_states": "C",
        "num_irs_elements": "M",
        "feedback_noise": "fb",
    }
    for key in ("num_nodes", "num_slots", "num_codebook_states", "num_irs_elements", "feedback_noise"):
        if key in labels:
            parts.append(f"{aliases[key]}{safe_label(labels[key])}")
    return "_".join(parts) if parts else "base"


def merged_preset_args(config, preset, overrides):
    """Merge default, preset-level, and command-line evaluator args."""
    args = dict(config.get("defaults", {}))
    for key, value in preset.items():
        if key not in RUNNER_ONLY_KEYS:
            args[key] = value
    args.update(overrides)
    return args


def build_jobs(config, preset_names, run_dir, python_executable, overrides, with_plots):
    """Build concrete job commands without executing them."""
    jobs = []
    for preset_name in preset_names:
        if preset_name not in config["presets"]:
            raise ValueError(f"unknown preset {preset_name!r}")
        preset = config["presets"][preset_name]
        base_args = merged_preset_args(config, preset, overrides)
        policies = ",".join(resolve_methods(config, preset))
        for index, (axis_args, labels) in enumerate(expanded_axis_records(config, preset), start=1):
            evaluator_args = {**base_args, **axis_args, "policies": policies}
            budget_note = filter_probe_budgets(evaluator_args)
            descriptor = {
                "preset": preset_name,
                "index": index,
                "args": evaluator_args,
                "labels": labels,
            }
            job_id = f"{preset_name}_{index:04d}_{job_label(labels)}_{short_hash(descriptor, 8)}"
            job_dir = run_dir / preset_name / job_id
            output_prefix = job_dir / "summary"
            structured_log_dir = job_dir / "structured_logs"
            evaluator_args["output_prefix"] = output_prefix
            evaluator_args["structured_log_dir"] = structured_log_dir
            evaluator_args["config_name"] = f"{preset_name}:{job_id}"
            command = [python_executable, "evaluate_execution_channel_mismatch.py"]
            for key, value in evaluator_args.items():
                if key in RUNNER_ONLY_KEYS or value is None:
                    continue
                if isinstance(value, bool):
                    if value:
                        command.append(cli_flag(key))
                    continue
                command.extend([cli_flag(key), cli_value(value)])
            if not with_plots:
                command.append("--no-plots")
            jobs.append(
                {
                    "preset": preset_name,
                    "job_id": job_id,
                    "job_dir": str(job_dir),
                    "output_prefix": str(output_prefix),
                    "structured_log_dir": str(structured_log_dir),
                    "command": command,
                    "labels": labels,
                    "evaluator_args": {key: str(value) for key, value in evaluator_args.items()},
                    "note": budget_note,
                }
            )
    return jobs


def expected_outputs(job):
    """Return files that indicate a completed main evaluator run."""
    prefix = Path(job["output_prefix"])
    structured_dir = Path(job["structured_log_dir"])
    return [
        prefix.with_suffix(".csv"),
        Path(f"{prefix}_slots.csv"),
        structured_dir / "run_metadata.jsonl",
        structured_dir / "scenario_summary.csv",
        structured_dir / "slot_records.csv",
        structured_dir / "diagnostic_records.jsonl",
    ]


def job_completed(job):
    """Check whether a job was already completed by this runner."""
    manifest = Path(job["job_dir"]) / "job_manifest.json"
    if not manifest.exists():
        return False
    try:
        with open(manifest, encoding="utf-8") as jsonfile:
            data = json.load(jsonfile)
    except json.JSONDecodeError:
        return False
    return data.get("status") == "success" and all(path.exists() for path in expected_outputs(job))


def write_json(path, data):
    """Write JSON with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as jsonfile:
        json.dump(data, jsonfile, indent=2, sort_keys=True)
        jsonfile.write("\n")


def print_dry_run(jobs):
    """Print planned jobs and commands without touching outputs."""
    print(f"DRY RUN: planned jobs = {len(jobs)}")
    for job in jobs:
        note = f" [{job['note']}]" if job.get("note") else ""
        print(f"- {job['job_id']}{note}")
        print("  " + shlex.join(str(part) for part in job["command"]))


def run_jobs(jobs, keep_going=False, overwrite=False):
    """Execute planned jobs with resume/skip semantics."""
    manifest = []
    for offset, job in enumerate(jobs, start=1):
        job_dir = Path(job["job_dir"])
        if not overwrite and job_completed(job):
            print(f"[{offset}/{len(jobs)}] skip completed {job['job_id']}")
            manifest.append({**job, "status": "skipped"})
            continue
        print(f"[{offset}/{len(jobs)}] running {job['job_id']}")
        if job.get("note"):
            print(f"  note: {job['note']}")
        job_dir.mkdir(parents=True, exist_ok=True)
        write_json(job_dir / "command.json", job)
        started = datetime.now(UTC).isoformat(timespec="seconds")
        completed = subprocess.run(
            [str(part) for part in job["command"]],
            cwd=PROJECT_ROOT,
            check=False,
        )
        finished = datetime.now(UTC).isoformat(timespec="seconds")
        status = "success" if completed.returncode == 0 else "failed"
        record = {
            **job,
            "status": status,
            "returncode": int(completed.returncode),
            "started_utc": started,
            "finished_utc": finished,
        }
        write_json(job_dir / "job_manifest.json", record)
        manifest.append(record)
        if completed.returncode != 0 and not keep_going:
            return completed.returncode, manifest
    return 0, manifest


def parse_args():
    """Parse runner arguments."""
    parser = argparse.ArgumentParser(description="Run paper experiment presets.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--presets", default="smoke_test")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--with-plots", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--override", action="append", default=[], help="Evaluator arg override: key=value")
    return parser.parse_args()


def main():
    """Runner entrypoint."""
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    preset_names = parse_csv_items(args.presets)
    overrides = dict(parse_override(item) for item in args.override)
    run_descriptor = {
        "config": str(config_path),
        "presets": preset_names,
        "overrides": overrides,
        "with_plots": bool(args.with_plots),
    }
    run_stamp = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.output_root) / f"{safe_label(run_stamp)}_{short_hash(run_descriptor)}"
    jobs = build_jobs(
        config,
        preset_names,
        run_dir,
        args.python,
        overrides,
        with_plots=args.with_plots,
    )
    if args.max_jobs > 0:
        jobs = jobs[: args.max_jobs]
    if args.dry_run:
        print_dry_run(jobs)
        return 0
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "suite_config_snapshot.json", config)
    write_json(run_dir / "run_descriptor.json", run_descriptor)
    write_json(run_dir / "planned_jobs.json", jobs)
    status, manifest = run_jobs(jobs, keep_going=args.keep_going, overwrite=args.overwrite)
    write_json(run_dir / "run_manifest.json", manifest)
    print(f"Saved suite run: {run_dir}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
