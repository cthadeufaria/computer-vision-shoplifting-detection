#!/usr/bin/env python3
"""Run STG-NF with simple workflow flags.

Workflows:
- eval-pretrained: evaluate official checkpoints
- train-scratch: full training from scratch (includes evaluation at the end)
"""

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STG-NF workflow runner")
    parser.add_argument(
        "--mode",
        choices=["eval-pretrained", "train-scratch"],
        required=True,
        help="Workflow to run.",
    )
    parser.add_argument(
        "--dataset",
        choices=["shanghaitech", "ubnormal", "all"],
        required=True,
        help="Dataset selection for the workflow.",
    )
    parser.add_argument(
        "--ubnormal-variant",
        choices=["supervised", "unsupervised", "both"],
        default="both",
        help="UBnormal checkpoint variant(s) for eval-pretrained.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint override (single run only).",
    )

    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--ubnormal-seg-len", type=int, default=16)

    parser.add_argument("--logs-dir", type=str, default="logs")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def print_cmd(cmd: list[str]) -> None:
    print("$ " + " ".join(shlex.quote(c) for c in cmd))


def run_cmd(cmd: list[str], cwd: Path, log_file: Optional[Path], dry_run: bool) -> None:
    print_cmd(cmd)
    if dry_run:
        return

    if log_file is None:
        subprocess.run(cmd, cwd=str(cwd), check=True)
        return

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w") as f:
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            f.write(line)
        ret = process.wait()
        if ret != 0:
            raise subprocess.CalledProcessError(ret, cmd)


def eval_targets(args: argparse.Namespace) -> list[tuple[str, str, str]]:
    # (dataset_name, checkpoint_path, tb_name)
    targets: list[tuple[str, str, str]] = []

    if args.checkpoint:
        if args.dataset == "all":
            raise ValueError("--checkpoint override supports only a single dataset run.")
        if args.dataset == "shanghaitech":
            targets.append(("ShanghaiTech", args.checkpoint, f"eval_{Path(args.checkpoint).stem}"))
        else:
            targets.append(("UBnormal", args.checkpoint, f"eval_{Path(args.checkpoint).stem}"))
        return targets

    if args.dataset in ("shanghaitech", "all"):
        ckpt = "checkpoints/ShanghaiTech_85_9.tar"
        targets.append(("ShanghaiTech", ckpt, "eval_shanghaitech_85_9"))

    if args.dataset in ("ubnormal", "all"):
        if args.ubnormal_variant in ("supervised", "both"):
            ckpt = "checkpoints/UBnormal_supervised_79_2.tar"
            targets.append(("UBnormal", ckpt, "eval_ubnormal_supervised_79_2"))
        if args.ubnormal_variant in ("unsupervised", "both"):
            ckpt = "checkpoints/UBnormal_unsupervised_71_8.tar"
            targets.append(("UBnormal", ckpt, "eval_ubnormal_unsupervised_71_8"))

    return targets


def train_targets(args: argparse.Namespace) -> list[str]:
    if args.dataset == "all":
        return ["ShanghaiTech", "UBnormal"]
    if args.dataset == "shanghaitech":
        return ["ShanghaiTech"]
    return ["UBnormal"]


def main() -> None:
    args = parse_args()
    # Run train_eval.py from within the stg_nf_official submodule directory.
    repo_root = Path(__file__).resolve().parent.parent / "stg_nf_official"
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")

    if args.mode == "eval-pretrained":
        targets = eval_targets(args)
        if not targets:
            raise ValueError("No eval targets resolved. Check --dataset/--ubnormal-variant flags.")

        for dataset_name, checkpoint, _tb_name in targets:
            cmd = [
                sys.executable,
                "train_eval.py",
                "--dataset",
                dataset_name,
                "--checkpoint",
                checkpoint,
                "--device",
                args.device,
                "--num_workers",
                str(args.num_workers),
            ]
            if dataset_name == "UBnormal":
                cmd.extend(["--seg_len", str(args.ubnormal_seg_len)])
            run_cmd(cmd, repo_root, log_file=None, dry_run=args.dry_run)

    elif args.mode == "train-scratch":
        targets = train_targets(args)
        for dataset_name in targets:
            slug = "shanghaitech" if dataset_name == "ShanghaiTech" else "ubnormal"
            log_file = Path(args.logs_dir) / f"{slug}_train_{run_id}.log"

            cmd = [
                sys.executable,
                "train_eval.py",
                "--dataset",
                dataset_name,
                "--device",
                args.device,
                "--num_workers",
                str(args.num_workers),
            ]
            if dataset_name == "UBnormal":
                cmd.extend(["--seg_len", str(args.ubnormal_seg_len)])

            run_cmd(cmd, repo_root, log_file=log_file, dry_run=args.dry_run)

    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
