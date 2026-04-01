#!/usr/bin/env python3
"""
STG-NF inference helper.

This script loads an exported weights-only checkpoint (see --export_dir usage in train_eval.py),
runs inference on a dataset split, and writes a small bundle of artifacts you can reuse later.

Notes:
- The official STG-NF pipeline operates on pose JSONs (not raw RGB frames).
- The official downloadable "data/" bundle contains pose + GT, but typically not frames.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

# Allow importing STG-NF internals from the submodule.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stg_nf_official"))

import numpy as np
import torch

from models.STG_NF.model_pose import STG_NF
from dataset import get_dataset_and_loader
from utils.data_utils import trans_list
from utils.scoring_utils import score_dataset
from utils.train_utils import init_model_params
from args import init_parser, init_sub_args


def load_export(export_dir: Path) -> Tuple[Dict, Dict, Dict]:
    model_args = json.loads((export_dir / "model_args.json").read_text())
    run_args = json.loads((export_dir / "run_args.json").read_text())
    weights = torch.load(export_dir / "weights.pt", map_location="cpu", weights_only=False)
    return model_args, run_args, weights


def compute_normality_scores(model: STG_NF, test_loader, device: str, use_confidence: bool) -> np.ndarray:
    model.eval().to(device)
    probs = torch.empty(0, device=device)
    for data_arr in test_loader:
        data = []
        for t in data_arr:
            if not torch.is_tensor(t):
                data.append(t)
                continue
            # MPS doesn't support float64.
            if str(device).startswith("mps") and t.is_floating_point() and t.dtype == torch.float64:
                t = t.float()
            data.append(t.to(device, non_blocking=False))

        score = data[-2].amin(dim=-1)
        if use_confidence:
            samp = data[0]
        else:
            samp = data[0][:, :2]

        with torch.no_grad():
            _, nll = model(samp.float(), label=torch.ones(data[0].shape[0], device=device), score=score)
        if use_confidence:
            nll = nll * score
        probs = torch.cat((probs, -1 * nll), dim=0)
    return probs.detach().cpu().numpy().squeeze().copy(order="C")


class STGNFInferencer:
    """
    Class-based STG-NF inferencer.

    The `model` parameter points to an exported model directory containing:
    - weights.pt
    - model_args.json
    - run_args.json
    """

    def __init__(self, model: str, exports_root: str = "exports"):
        self.exports_root = Path(exports_root)
        self.export_dir = self.resolve_model_path(model, self.exports_root)
        self.model_args, self.run_args, self.weights = load_export(self.export_dir)

    @staticmethod
    def list_saved_models(exports_root: str = "exports") -> List[str]:
        root = Path(exports_root)
        if not root.exists():
            return []
        models = []
        for weights_path in root.glob("**/weights.pt"):
            models.append(str(weights_path.parent))
        return sorted(models)

    @staticmethod
    def resolve_model_path(model: str, exports_root: Path) -> Path:
        candidate = Path(model)
        if candidate.is_dir() and (candidate / "weights.pt").exists():
            return candidate

        if candidate.exists() and candidate.is_file():
            if candidate.name == "weights.pt":
                return candidate.parent
            raise ValueError(f"Model file '{model}' is not a weights.pt file.")

        if exports_root.exists():
            matches = []
            for weights_path in exports_root.glob("**/weights.pt"):
                parent = weights_path.parent
                if parent.name == model:
                    matches.append(parent)
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise ValueError(
                    "Model name is ambiguous. Use a full path. Matches: "
                    + ", ".join(str(m) for m in matches)
                )

        raise ValueError(
            f"Could not resolve model '{model}'. "
            "Pass an export directory path or a unique export folder name under exports/."
        )

    def infer(self,
              remaining_base_args: List[str],
              output: str = str(Path(__file__).parent / "inference_outputs"),
              device: Optional[str] = None,
              num_workers: int = 0,
              no_gt: bool = False) -> Dict:
        # Reuse upstream parser so data semantics remain identical.
        base_parser = init_parser()
        base_args = base_parser.parse_args(remaining_base_args)

        out_root = Path(output)
        out_root.mkdir(parents=True, exist_ok=True)

        # Setup MPS fallback for unsupported ops.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

        # Runtime overrides.
        if device is not None:
            base_args.device = device
        base_args.num_workers = num_workers

        # Init derived args + model args mapping (upstream behavior).
        base_args, _ = init_sub_args(base_args)

        # Build dataset + loader exactly like training/eval.
        dataset, loader = get_dataset_and_loader(base_args, trans_list=trans_list, only_test=True)

        # Build model_args for current environment from dataset shape, then keep exported hyperparams.
        inferred_model_args = init_model_params(base_args, dataset)
        for k in ("hidden_channels", "K", "L", "R", "actnorm_scale", "flow_permutation", "flow_coupling",
                  "LU_decomposed", "learn_top", "edge_importance", "temporal_kernel_size",
                  "strategy", "max_hops"):
            if k in self.model_args:
                inferred_model_args[k] = self.model_args[k]
        inferred_model_args["device"] = base_args.device

        model = STG_NF(**inferred_model_args)
        model.load_state_dict(self.weights["state_dict"], strict=False)
        model.set_actnorm_init()

        seg_scores = compute_normality_scores(
            model, loader["test"], device=base_args.device, use_confidence=base_args.model_confidence
        )
        metadata = np.array(dataset["test"].metadata, dtype=object)

        eval_metrics = None
        frame_scores = None
        if not no_gt:
            auc_roc, auc_pr, eer, frame_scores = score_dataset(seg_scores, dataset["test"].metadata, args=base_args)
            eval_metrics = {"auc_roc": float(auc_roc), "auc_pr": float(auc_pr), "eer": float(eer)}

        # Save artifacts.
        run_dir = out_root / base_args.dataset / self.export_dir.name
        run_dir.mkdir(parents=True, exist_ok=True)
        np.save(run_dir / "segment_scores.npy", seg_scores)
        np.save(run_dir / "segment_metadata.npy", metadata)
        if frame_scores is not None:
            np.save(run_dir / "frame_scores.npy", frame_scores)
        (run_dir / "export_path.txt").write_text(str(self.export_dir.resolve()) + "\n")

        # Always provide a simple per-clip aggregation for inference use (no GT required).
        clip_scores: Dict[str, float] = {}
        if metadata.shape[0] == seg_scores.shape[0]:
            for (scene_id, clip_id, _person_id, _start_frame), s in zip(metadata, seg_scores):
                key = f"{scene_id}_{clip_id}"
                prev = clip_scores.get(key)
                if prev is None or float(s) > prev:
                    clip_scores[key] = float(s)
        (run_dir / "clip_scores_max.json").write_text(json.dumps(clip_scores, indent=2) + "\n")

        metrics_payload = {
            "dataset": base_args.dataset,
            "export": str(self.export_dir),
            "device": base_args.device,
            "auc_roc": (None if eval_metrics is None else eval_metrics["auc_roc"]),
            "num_segment_scores": int(seg_scores.shape[0]),
            "num_frame_scores": (None if frame_scores is None else int(frame_scores.shape[0])),
            "num_clips": len(clip_scores),
            "torch_version": torch.__version__,
        }
        (run_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2) + "\n")

        return {
            "output_dir": str(run_dir),
            "auc_roc": metrics_payload["auc_roc"],
            "num_frame_scores": metrics_payload["num_frame_scores"],
            "num_segment_scores": metrics_payload["num_segment_scores"],
            "num_clips": metrics_payload["num_clips"],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="STG-NF inference", add_help=True)

    parser.add_argument(
        "--model",
        "--export",
        dest="model",
        type=str,
        default=None,
        help="Model selector: export directory path, export folder name, or weights.pt path.",
    )
    parser.add_argument("--exports_root", type=str, default="exports",
                        help="Root directory to search when --model is a folder name.")
    parser.add_argument("--list_models", action="store_true",
                        help="List available exported models under --exports_root and exit.")
    _default_output = str(Path(__file__).parent / "inference_outputs")
    parser.add_argument("--output", type=str, default=_default_output, help="Output root dir.")
    parser.add_argument("--device", type=str, default=None, help="Override device (e.g. mps/cpu/cuda:0).")
    parser.add_argument("--num_workers", type=int, default=0, help="Dataloader workers.")
    parser.add_argument("--no_gt", action="store_true", help="Skip GT-based scoring (AUC/frame-level scores).")

    # Parse known args and forward remaining args to the upstream dataset parser.
    args, remaining = parser.parse_known_args()
    if args.list_models:
        print(json.dumps({"models": STGNFInferencer.list_saved_models(args.exports_root)}, indent=2))
        return

    if not args.model:
        parser.error("--model is required unless --list_models is set.")

    inferencer = STGNFInferencer(model=args.model, exports_root=args.exports_root)
    result = inferencer.infer(
        remaining_base_args=remaining,
        output=args.output,
        device=args.device,
        num_workers=args.num_workers,
        no_gt=args.no_gt,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
