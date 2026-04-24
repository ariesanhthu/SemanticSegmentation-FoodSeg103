import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from analysis.ccnet_report import generate_ccnet_report
from configs.ccnet_foodseg103 import CFG, get_paths
from datasets.foodseg103_ccnet import resolve_dataset_meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an evaluation report with plots for a saved CCNet checkpoint."
    )
    parser.add_argument("--data-root", type=str, default=None, help="Override dataset root.")
    parser.add_argument("--work-dir", type=str, default=None, help="Override work directory.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path. Defaults to best_miou.pth inside work_dir.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test"],
        default="test",
        help="Dataset split used for the report.",
    )
    parser.add_argument("--max-items", type=int, default=None, help="Evaluate at most N samples.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override eval batch size.")
    parser.add_argument("--num-worst-cases", type=int, default=8, help="Number of worst cases to render.")
    parser.add_argument("--num-best-cases", type=int, default=8, help="Number of best cases to render.")
    parser.add_argument("--top-k-classes", type=int, default=20, help="Number of low-IoU classes to plot.")
    parser.add_argument("--top-k-confusions", type=int, default=20, help="Number of confusion pairs to plot.")
    parser.add_argument(
        "--report-name",
        type=str,
        default=None,
        help="Optional output folder name under work_dir/reports.",
    )
    return parser.parse_args()


def get_runtime_cfg(args: argparse.Namespace) -> dict:
    cfg = CFG.copy()
    if args.data_root:
        cfg["data_root"] = args.data_root
    if args.work_dir:
        cfg["work_dir"] = args.work_dir
    if args.batch_size is not None:
        cfg["eval_batch_size"] = args.batch_size
    return resolve_dataset_meta(cfg)


def main() -> None:
    args = parse_args()
    cfg = get_runtime_cfg(args)
    paths = get_paths(cfg)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else paths["work_dir"] / cfg["save_best_name"]
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    report = generate_ccnet_report(
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        split=args.split,
        report_name=args.report_name,
        max_items=args.max_items,
        batch_size=args.batch_size,
        num_worst_cases=args.num_worst_cases,
        num_best_cases=args.num_best_cases,
        top_k_classes=args.top_k_classes,
        top_k_confusions=args.top_k_confusions,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
