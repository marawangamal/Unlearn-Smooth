#!/usr/bin/env python3
"""Plot NPO and NPO+SAM relearning results from full lm-eval evaluations."""

import argparse
import glob
import json
import os
import os.path as osp

import pandas as pd
import seaborn as sns


def get_lm_eval_results(path):
    """Load full task results, preferring later evaluation timestamps."""
    result_paths = sorted(
        glob.glob(osp.join(path, "**", "results_*.json"), recursive=True),
        key=osp.basename,
    )
    if not result_paths:
        result_paths = [
            osp.join(path, name)
            for name in ("mmlu.json", "wmdp.json")
            if osp.isfile(osp.join(path, name))
        ]
    if not result_paths:
        raise FileNotFoundError(f"No lm-eval results found in {path}")
    task_results = {}
    for result_path in result_paths:
        with open(result_path) as handle:
            result = json.load(handle)
        if result["config"].get("limit") is None:
            task_results.update(result["results"])
    return task_results


task_to_task = {
    "wmdp_bio": "WMDP-Bio unlearning effectiveness ↑",
    "mmlu": "MMLU accuracy ↑",
}
task_to_metrics = {
    "wmdp_bio": lambda metrics: 1 - metrics["acc,none"],
    "mmlu": lambda metrics: metrics["acc,none"],
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--outpath", default="artifacts/figures/zephyr-7b-relearning-trajectory.pdf"
    )
    args = parser.parse_args()

    rootdir = osp.join("artifacts/tamia/evals", args.run_name)
    configs = [
        {"path": osp.join(rootdir, "npo"), "name": "NPO", "samples": 0},
        {
            "path": osp.join(rootdir, "npo-n60-m1-seed1"),
            "name": "NPO",
            "samples": 60,
        },
        {"path": osp.join(rootdir, "npo-sam"), "name": "NPO+SAM", "samples": 0},
        {
            "path": osp.join(rootdir, "npo-sam-n60-m1-seed1"),
            "name": "NPO+SAM",
            "samples": 60,
        },
    ]

    rows = []
    for cfg in configs:
        task_results = get_lm_eval_results(cfg["path"])
        for task, get_metric in task_to_metrics.items():
            rows.append(
                {
                    "Method": cfg["name"],
                    "Samples": cfg["samples"],
                    "Score (%)": 100 * get_metric(task_results[task]),
                    "Eval": task_to_task[task],
                }
            )

    df = pd.DataFrame(rows)
    sns.set_theme(context="paper", style="ticks", font_scale=1.2)
    grid = sns.relplot(
        data=df,
        x="Samples",
        y="Score (%)",
        hue="Method",
        col="Eval",
        kind="line",
        marker="o",
        height=3,
        aspect=1.2,
    )
    grid.set_titles("{col_name}")
    grid.set_axis_labels("Relearning samples", "Score (%)")
    grid.set(xticks=sorted(df["Samples"].unique()))
    os.makedirs(osp.dirname(args.outpath) or ".", exist_ok=True)
    grid.figure.savefig(args.outpath, bbox_inches="tight")
