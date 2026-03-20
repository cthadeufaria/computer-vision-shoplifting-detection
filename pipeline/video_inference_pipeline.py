#!/usr/bin/env python3
"""
Online video inference pipeline for STG-NF exported models.

Pipeline order per frame:
1) Real pose estimation with MMPose
   - Primary: RTMO (single-stage, no detector dependency)
   - Fallback: RTMPose-s (top-down)
2) IoU-based pose tracking for persistent person IDs
3) STG-NF anomaly scoring on rolling windows

Outputs:
- per-frame inference JSON
- generated tracked-pose JSON (STG-NF compatible structure)
- optional annotated output video
"""

import argparse
import json
import os
import sys
import warnings
from collections import defaultdict
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict
from typing import Deque
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch

# Allow importing STG-NF internals from the submodule.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stg_nf_official"))

# Reduce noisy third-party warnings during per-frame inference loops.
warnings.filterwarnings("ignore", message="`torch.cuda.amp.autocast")
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")
warnings.filterwarnings("ignore", message="Fail to import ``MultiScaleDeformableAttention``")
try:
    from urllib3.exceptions import NotOpenSSLWarning

    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except Exception:
    pass

from dataset import keypoints17_to_coco18
from inference import STGNFInferencer
from models.STG_NF.model_pose import STG_NF
from utils.data_utils import normalize_pose

SUPPORTED_DATASETS = ("ShanghaiTech", "UBnormal")
SUPPORTED_POSE_BACKENDS = ("rtmo", "rtmpose_s")


def _require_cv2():
    try:
        import cv2  # type: ignore

        return cv2
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "OpenCV is required for video inference pipeline. Install with: pip install opencv-python"
        ) from exc


def _iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


@dataclass
class PoseDetection:
    bbox: Tuple[float, float, float, float]
    conf: float
    keypoints17: np.ndarray  # [17, 3] as (x, y, conf)
    pose_score: float


class MMPosePoseEstimator:
    """Real pose estimator powered by MMPose inferencer."""

    RTMO_MODEL = "rtmo"
    RTMPOSE_S_MODEL = "rtmpose-s_8xb256-420e_body8-256x192"

    def __init__(
        self,
        backend: str = "rtmo",
        fallback_backend: Optional[str] = "rtmpose_s",
        allow_fallback: bool = True,
        device: str = "cpu",
        scope: str = "mmpose",
        det_model: Optional[str] = None,
        det_weights: Optional[str] = None,
        kpt_thr: float = 0.05,
    ):
        self.backend = self._normalize_backend_name(backend)
        self.fallback_backend = (
            None if fallback_backend is None else self._normalize_backend_name(fallback_backend)
        )
        self.allow_fallback = allow_fallback
        self.device = device
        self.scope = scope
        self.det_model = det_model
        self.det_weights = det_weights
        self.kpt_thr = kpt_thr

        self.active_backend = self.backend
        self.inferencer = self._build_with_optional_fallback()

    @staticmethod
    def _normalize_backend_name(name: str) -> str:
        norm = name.lower().replace("-", "_")
        if norm not in SUPPORTED_POSE_BACKENDS:
            raise ValueError(f"Unsupported pose backend '{name}'. Supported: {SUPPORTED_POSE_BACKENDS}")
        return norm

    @staticmethod
    def _parse_bbox_field(bbox_field: object, keypoints_xy: np.ndarray) -> Tuple[float, float, float, float]:
        if bbox_field is not None:
            bbox_arr = np.asarray(bbox_field, dtype=np.float32).reshape(-1)
            if bbox_arr.shape[0] >= 4:
                x1, y1, x2, y2 = [float(v) for v in bbox_arr[:4]]
                return (x1, y1, x2, y2)

        x_vals = keypoints_xy[:, 0]
        y_vals = keypoints_xy[:, 1]
        return (float(np.min(x_vals)), float(np.min(y_vals)), float(np.max(x_vals)), float(np.max(y_vals)))

    @staticmethod
    def _as_17_keypoints(keypoints: np.ndarray, kpt_scores: np.ndarray) -> Optional[np.ndarray]:
        if keypoints.ndim != 2 or keypoints.shape[1] < 2:
            return None
        if keypoints.shape[0] < 17:
            return None

        # RTMO/RTMPose body models typically output 17 keypoints already.
        keypoints_xy = keypoints[:17, :2].astype(np.float32)
        if kpt_scores.size >= 17:
            kp_conf = kpt_scores[:17].astype(np.float32)
        else:
            kp_conf = np.ones(17, dtype=np.float32) * 0.5

        return np.concatenate([keypoints_xy, kp_conf[:, None]], axis=1)

    def _import_inferencer_class(self):
        try:
            from mmpose.apis import MMPoseInferencer  # type: ignore

            return MMPoseInferencer
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Failed to import MMPoseInferencer. "
                "Install MMPose runtime deps including mmengine, mmdet, and full mmcv (not only mmcv-lite)."
            ) from exc

    def _inferencer_kwargs(self, backend: str) -> Dict:
        if backend == "rtmo":
            # RTMO is a one-stage pose model; use whole-image mode to skip detector.
            return dict(
                pose2d=self.RTMO_MODEL,
                device=self.device,
                scope=self.scope,
                det_model="whole_image",
            )

        if backend == "rtmpose_s":
            kwargs = dict(
                pose2d=self.RTMPOSE_S_MODEL,
                device=self.device,
                scope=self.scope,
            )
            # Top-down RTMPose-s can use MMPose's default human detector when det_model is None.
            if self.det_model is not None:
                kwargs["det_model"] = self.det_model
            if self.det_weights is not None:
                kwargs["det_weights"] = self.det_weights
            return kwargs

        raise ValueError(f"Unsupported backend: {backend}")

    def _build_inferencer(self, backend: str):
        InferencerCls = self._import_inferencer_class()
        kwargs = self._inferencer_kwargs(backend)
        return InferencerCls(**kwargs)

    def _build_with_optional_fallback(self):
        try:
            self.active_backend = self.backend
            return self._build_inferencer(self.backend)
        except Exception as primary_exc:
            if (
                not self.allow_fallback
                or self.fallback_backend is None
                or self.fallback_backend == self.backend
            ):
                raise RuntimeError(
                    f"Failed to initialize MMPose backend '{self.backend}'. "
                    "A common cause is missing compiled MMCV ops (mmcv._ext). "
                    "Install a compatible full mmcv build for your torch/platform."
                ) from primary_exc

            try:
                self.active_backend = self.fallback_backend
                return self._build_inferencer(self.fallback_backend)
            except Exception as fallback_exc:
                raise RuntimeError(
                    f"Failed to initialize both pose backends: '{self.backend}' and '{self.fallback_backend}'. "
                    "A common cause is missing compiled MMCV ops (mmcv._ext). "
                    "Install a compatible full mmcv build for your torch/platform."
                ) from fallback_exc

    def infer_pose(self, frame_bgr: np.ndarray) -> List[PoseDetection]:
        # MMPoseInferencer returns a generator; each step yields one result dict.
        result_iter = self.inferencer(
            frame_bgr,
            return_datasamples=False,
            return_vis=False,
            show=False,
            draw_bbox=False,
        )
        result = next(result_iter)

        predictions = result.get("predictions", [])
        # For image inputs, some inferencer versions return nested predictions.
        if predictions and isinstance(predictions[0], list):
            predictions = predictions[0]

        out: List[PoseDetection] = []
        for pred in predictions:
            keypoints = np.asarray(pred.get("keypoints", []), dtype=np.float32)
            keypoint_scores = np.asarray(pred.get("keypoint_scores", []), dtype=np.float32).reshape(-1)

            kps17 = self._as_17_keypoints(keypoints, keypoint_scores)
            if kps17 is None:
                continue

            mean_conf = float(np.mean(kps17[:, 2]))
            if mean_conf < self.kpt_thr:
                continue

            bbox = self._parse_bbox_field(pred.get("bbox"), kps17[:, :2])
            bbox_score = pred.get("bbox_score", mean_conf)
            try:
                conf = float(np.asarray(bbox_score).reshape(-1)[0])
            except Exception:
                conf = mean_conf

            # Keep score scale similar to existing STG-NF tracked JSONs.
            pose_score = float(mean_conf * 3.0)
            out.append(PoseDetection(bbox=bbox, conf=conf, keypoints17=kps17, pose_score=pose_score))

        return out


class SimpleIoUTracker:
    def __init__(self, iou_th: float = 0.3, max_missing: int = 6):
        self.iou_th = iou_th
        self.max_missing = max_missing
        self.next_track_id = 1
        self.tracks: Dict[int, Dict] = {}

    def _prune_stale(self, frame_idx: int) -> None:
        stale_ids = [
            tid for tid, st in self.tracks.items()
            if frame_idx - int(st["last_frame"]) > self.max_missing
        ]
        for tid in stale_ids:
            del self.tracks[tid]

    def update(self, frame_idx: int, detections: List[PoseDetection]) -> List[Tuple[int, PoseDetection]]:
        self._prune_stale(frame_idx)

        detections_sorted = sorted(detections, key=lambda d: d.conf, reverse=True)
        used_track_ids = set()
        assignments: List[Tuple[int, PoseDetection]] = []

        for det in detections_sorted:
            best_tid = None
            best_iou = 0.0
            for tid, st in self.tracks.items():
                if tid in used_track_ids:
                    continue
                iou = _iou_xyxy(det.bbox, st["bbox"])
                if iou > self.iou_th and iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_tid is None:
                best_tid = self.next_track_id
                self.next_track_id += 1

            self.tracks[best_tid] = {
                "bbox": det.bbox,
                "last_frame": frame_idx,
            }
            used_track_ids.add(best_tid)
            assignments.append((best_tid, det))

        return assignments


class OnlineVideoInferencePipeline:
    def __init__(
        self,
        model: str,
        dataset: str,
        exports_root: str = "exports",
        device: str = "cpu",
        seg_len: Optional[int] = None,
        batch_size: int = 1,
        pose_backend: str = "rtmo",
        pose_fallback_backend: Optional[str] = "rtmpose_s",
        pose_allow_fallback: bool = True,
        pose_device: str = "cpu",
        pose_scope: str = "mmpose",
        pose_det_model: Optional[str] = None,
        pose_det_weights: Optional[str] = None,
        pose_kpt_thr: float = 0.05,
        track_iou_th: float = 0.3,
        track_max_missing: int = 6,
        anomaly_threshold: Optional[float] = None,
    ):
        if dataset not in SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported dataset '{dataset}'. Supported: {SUPPORTED_DATASETS}")

        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        self.inferencer = STGNFInferencer(model=model, exports_root=exports_root)
        self.model_confidence = bool(self.inferencer.run_args.get("model_confidence", False))
        self.seg_len = int(seg_len if seg_len is not None else self.inferencer.run_args.get("seg_len", 24))

        self.pose_estimator = MMPosePoseEstimator(
            backend=pose_backend,
            fallback_backend=pose_fallback_backend,
            allow_fallback=pose_allow_fallback,
            device=pose_device,
            scope=pose_scope,
            det_model=pose_det_model,
            det_weights=pose_det_weights,
            kpt_thr=pose_kpt_thr,
        )
        self.tracker = SimpleIoUTracker(iou_th=track_iou_th, max_missing=track_max_missing)
        self.anomaly_threshold = anomaly_threshold

        self.model = None

    def _build_model(self, num_vertices: int) -> None:
        pose_shape = (3 if self.model_confidence else 2, self.seg_len, num_vertices)
        model_args = dict(self.inferencer.model_args)
        model_args["pose_shape"] = pose_shape
        model_args["device"] = self.device

        self.model = STG_NF(**model_args)
        self.model.load_state_dict(self.inferencer.weights["state_dict"], strict=False)
        self.model.set_actnorm_init()
        self.model.eval().to(self.device)

    @staticmethod
    def _video_meta(video_path: Path) -> Dict:
        cv2 = _require_cv2()
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        meta = {
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        }
        cap.release()
        return meta

    def _infer_single_segment(
        self,
        seg_pose17: np.ndarray,
        seg_scores: np.ndarray,
        vid_res: Tuple[int, int],
    ) -> float:
        # [T, 17, 3] -> [T, 18, 3]
        seg_pose18 = keypoints17_to_coco18(seg_pose17)
        if self.model is None:
            self._build_model(num_vertices=int(seg_pose18.shape[1]))

        # Normalize exactly as training/eval pipeline expects.
        seg_norm = normalize_pose(seg_pose18[None, ...], vid_res=[int(vid_res[0]), int(vid_res[1])]).astype(np.float32)
        seg_ctv = np.transpose(seg_norm, (0, 3, 1, 2))  # [1, C, T, V]

        x = torch.from_numpy(seg_ctv).to(self.device)
        score = torch.tensor([float(np.min(seg_scores))], dtype=torch.float32, device=self.device)

        if self.model_confidence:
            sample = x
        else:
            sample = x[:, :2]

        with torch.no_grad():
            _, nll = self.model(sample.float(), label=torch.ones(sample.shape[0], device=self.device), score=score)
            if self.model_confidence:
                nll = nll * score

        return float((-1.0 * nll).detach().cpu().numpy().reshape(-1)[0])

    @staticmethod
    def _frame_key(frame_idx: int) -> str:
        return f"{frame_idx:04d}"

    def _pick_first_video_with_person(self, videos: Iterable[Path]) -> Path:
        cv2 = _require_cv2()
        videos = list(videos)
        if not videos:
            raise ValueError("No videos to select from")
        fallback = videos[0]
        for video_path in videos:
            cap = cv2.VideoCapture(str(video_path))
            ok, frame = cap.read()
            cap.release()
            if not ok:
                continue
            dets = self.pose_estimator.infer_pose(frame)
            if dets:
                return video_path
        return fallback

    def _select_sinth_samples(self, sinth_root: Path) -> Dict[str, Path]:
        normal_dir = sinth_root / "Normal"
        shoplifting_dir = sinth_root / "Shoplifting"
        if not normal_dir.exists() or not shoplifting_dir.exists():
            raise FileNotFoundError(
                f"Expected subfolders 'Normal' and 'Shoplifting' under: {sinth_root}"
            )

        normal_videos = sorted(normal_dir.glob("*.mp4"))
        shoplifting_videos = sorted(shoplifting_dir.glob("*.mp4"))
        if not normal_videos:
            raise FileNotFoundError(f"No .mp4 videos found in: {normal_dir}")
        if not shoplifting_videos:
            raise FileNotFoundError(f"No .mp4 videos found in: {shoplifting_dir}")

        return {
            "Normal": self._pick_first_video_with_person(normal_videos),
            "Shoplifting": self._pick_first_video_with_person(shoplifting_videos),
        }

    def infer_video(
        self,
        video_path: str,
        output_dir: str = str(Path(__file__).parent / "inference_outputs" / "video_online"),
        output_name: Optional[str] = None,
        display: bool = True,
        save_annotated_video: bool = True,
        max_frames: Optional[int] = None,
    ) -> Dict:
        cv2 = _require_cv2()
        video_p = Path(video_path)
        if not video_p.exists():
            raise FileNotFoundError(f"Video not found: {video_p}")

        meta = self._video_meta(video_p)
        frame_w, frame_h = int(meta["width"]), int(meta["height"])
        fps = float(meta["fps"] if meta["fps"] > 0 else 30.0)

        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)

        stem = output_name if output_name else video_p.stem
        score_json_path = out_root / f"{stem}_{self.inferencer.export_dir.name}_scores.json"
        pose_json_path = out_root / f"{stem}_{self.inferencer.export_dir.name}_tracked_pose.json"
        output_video_path = out_root / f"{stem}_{self.inferencer.export_dir.name}_annotated.mp4"

        cap = cv2.VideoCapture(str(video_p))
        if not cap.isOpened():
            raise ValueError(f"Could not open video stream: {video_p}")

        writer = None
        if save_annotated_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (frame_w, frame_h))

        # track_id -> frame_key -> {keypoints, scores}
        clip_dict: DefaultDict[str, Dict[str, Dict]] = defaultdict(dict)

        # track_id -> deque[(frame_idx, keypoints17, pose_score)]
        track_histories: Dict[int, Deque[Tuple[int, np.ndarray, float]]] = {}

        frame_results = []
        frame_idx = 0
        window_name = f"Online Inference: {video_p.name}"

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and frame_idx >= max_frames:
                break

            detections = self.pose_estimator.infer_pose(frame)
            assignments = self.tracker.update(frame_idx, detections)

            # Build/update pose tracks.
            for track_id, det in assignments:
                frame_key = self._frame_key(frame_idx)
                clip_dict[str(track_id)][frame_key] = {
                    "keypoints": det.keypoints17.reshape(-1).astype(float).tolist(),
                    "scores": float(det.pose_score),
                }

                if track_id not in track_histories:
                    track_histories[track_id] = deque(maxlen=self.seg_len)
                track_histories[track_id].append((frame_idx, det.keypoints17, det.pose_score))

            # Online frame score from current rolling windows.
            per_track_scores: Dict[str, float] = {}
            for track_id, hist in track_histories.items():
                if len(hist) < self.seg_len:
                    continue
                frame_ids = [item[0] for item in hist]
                if any(frame_ids[i] != frame_ids[0] + i for i in range(len(frame_ids))):
                    continue

                seg_pose17 = np.stack([item[1] for item in hist], axis=0)
                seg_scores = np.array([item[2] for item in hist], dtype=np.float32)
                score = self._infer_single_segment(seg_pose17, seg_scores, (frame_w, frame_h))
                per_track_scores[str(track_id)] = float(score)

            frame_score = float(min(per_track_scores.values())) if per_track_scores else float("nan")
            serializable_score = None if np.isnan(frame_score) else float(frame_score)
            alert = None
            if serializable_score is not None and self.anomaly_threshold is not None:
                # STG-NF score here is normality-like (-NLL), so lower values are more anomalous.
                alert = bool(serializable_score <= float(self.anomaly_threshold))
            ts = float(frame_idx / fps) if fps > 0 else None
            frame_results.append(
                {
                    "frame_index": int(frame_idx),
                    "timestamp_sec": ts,
                    "score": serializable_score,
                    "alert": alert,
                    "anomaly_threshold": self.anomaly_threshold,
                    "track_scores": per_track_scores,
                    "num_person_detections": int(len(assignments)),
                }
            )

            # Visualization.
            for track_id, det in assignments:
                x1, y1, x2, y2 = det.bbox
                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"ID {track_id} conf {det.conf:.2f}",
                    (int(x1), max(18, int(y1) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                # Draw keypoints (small dots).
                for x, y, c in det.keypoints17:
                    if c < 0.1:
                        continue
                    cv2.circle(frame, (int(x), int(y)), 2, (255, 200, 0), -1)

            sorted_track_scores = sorted(per_track_scores.items(), key=lambda kv: kv[1])
            if sorted_track_scores:
                top_scores = " ".join([f"{tid}:{score:.3f}" for tid, score in sorted_track_scores[:3]])
            else:
                top_scores = "n/a"
            max_hist = max((len(hist) for hist in track_histories.values()), default=0)

            score_text = "STG score: n/a" if serializable_score is None else f"STG score: {serializable_score:.4f}"
            state_text = "State: warming up"
            state_color = (0, 215, 255)
            if serializable_score is not None:
                if self.anomaly_threshold is None:
                    state_text = "State: score-only (no threshold)"
                    state_color = (0, 215, 255)
                elif alert:
                    state_text = f"State: ALERT (<= {self.anomaly_threshold:.4f})"
                    state_color = (0, 0, 255)
                else:
                    state_text = f"State: normal (> {self.anomaly_threshold:.4f})"
                    state_color = (0, 255, 0)

            overlay_lines = [
                (score_text, (0, 255, 255)),
                (state_text, state_color),
                (
                    f"Frame: {frame_idx}  Persons: {len(assignments)}  Tracks-ready: {len(per_track_scores)}",
                    (255, 255, 255),
                ),
                (
                    f"Track scores(min first): {top_scores}",
                    (255, 230, 140),
                ),
                (
                    f"Pose: {self.pose_estimator.active_backend}  Warmup: {max_hist}/{self.seg_len}",
                    (255, 255, 0),
                ),
            ]
            for i, (line_text, line_color) in enumerate(overlay_lines):
                cv2.putText(
                    frame,
                    line_text,
                    (20, 34 + (i * 28)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62 if i > 1 else 0.70,
                    line_color,
                    2,
                    cv2.LINE_AA,
                )

            if writer is not None:
                writer.write(frame)

            if display:
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    break

            frame_idx += 1

        cap.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyWindow(window_name)

        score_payload = {
            "video_path": str(video_p.resolve()),
            "dataset": self.dataset,
            "model_export_dir": str(self.inferencer.export_dir.resolve()),
            "pose_backend_requested": self.pose_estimator.backend,
            "pose_backend_active": self.pose_estimator.active_backend,
            "segment_length": int(self.seg_len),
            "fps": fps,
            "frame_count_processed": int(len(frame_results)),
            "results": frame_results,
            "pose_json_path": str(pose_json_path.resolve()),
            "output_video": str(output_video_path.resolve()) if save_annotated_video else None,
        }
        score_json_path.write_text(json.dumps(score_payload, indent=2) + "\n")
        pose_json_path.write_text(json.dumps(clip_dict, indent=2) + "\n")

        return {
            "video": str(video_p.resolve()),
            "scores_json": str(score_json_path.resolve()),
            "pose_json": str(pose_json_path.resolve()),
            "output_video": (str(output_video_path.resolve()) if save_annotated_video else None),
            "frames_processed": int(len(frame_results)),
            "pose_backend": self.pose_estimator.active_backend,
        }

    def infer_sinth_samples(
        self,
        sinth_root: str,
        output_dir: str = str(Path(__file__).parent / "inference_outputs" / "video_online"),
        display: bool = True,
        save_annotated_video: bool = True,
        max_frames: Optional[int] = None,
    ) -> Dict[str, Dict]:
        samples = self._select_sinth_samples(Path(sinth_root))
        outputs = {}
        for split_name, video_path in samples.items():
            outputs[split_name] = self.infer_video(
                video_path=str(video_path),
                output_dir=output_dir,
                output_name=f"{split_name.lower()}_{video_path.stem}",
                display=display,
                save_annotated_video=save_annotated_video,
                max_frames=max_frames,
            )
        return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Online STG-NF video inference pipeline with MMPose stage")

    parser.add_argument("--model", type=str, required=True,
                        help="Model selector (export dir path/name/weights.pt).")
    parser.add_argument("--dataset", type=str, required=True, choices=list(SUPPORTED_DATASETS),
                        help="Dataset mode for STG-NF model config.")

    parser.add_argument("--video", type=str, default=None, help="Single video path for inference.")
    parser.add_argument("--run_sinth_samples", action="store_true",
                        help="Run one video from data/sinth/Normal and one from data/sinth/Shoplifting.")
    parser.add_argument("--sinth_root", type=str, default="data/sinth",
                        help="Root folder containing Normal/ and Shoplifting/ mp4 files.")

    parser.add_argument("--exports_root", type=str, default="stg_nf_official/exports")
    parser.add_argument("--device", type=str, default="mps", help="STG-NF device (e.g. mps/cpu/cuda:0)")
    parser.add_argument("--seg_len", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--pose_backend", type=str, default="rtmo", choices=list(SUPPORTED_POSE_BACKENDS),
                        help="Primary MMPose backend.")
    parser.add_argument("--pose_fallback_backend", type=str, default="rtmpose_s",
                        choices=list(SUPPORTED_POSE_BACKENDS),
                        help="Fallback backend if primary init fails.")
    parser.add_argument("--disable_pose_fallback", action="store_true",
                        help="Disable automatic fallback to --pose_fallback_backend.")
    parser.add_argument("--pose_device", type=str, default="cpu",
                        help="Pose model device (e.g. cpu/cuda:0/mps).")
    parser.add_argument("--pose_scope", type=str, default="mmpose")
    parser.add_argument("--pose_det_model", type=str, default=None,
                        help="Optional detector config/alias for top-down pose backends.")
    parser.add_argument("--pose_det_weights", type=str, default=None,
                        help="Optional detector weights path/URL for top-down pose backends.")
    parser.add_argument("--pose_kpt_thr", type=float, default=0.05,
                        help="Minimum mean keypoint confidence to keep a pose instance.")

    parser.add_argument("--track_iou_th", type=float, default=0.3)
    parser.add_argument("--track_max_missing", type=int, default=6)
    parser.add_argument("--anomaly_threshold", type=float, default=None,
                        help="If set, mark frames as ALERT when STG score <= threshold.")

    _default_output_dir = str(Path(__file__).parent / "inference_outputs" / "video_online")
    parser.add_argument("--output_dir", type=str, default=_default_output_dir)
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Optional cap for fast smoke tests.")
    parser.add_argument("--save_annotated_video", action="store_true",
                        help="Save an annotated mp4 alongside JSON outputs.")

    parser.add_argument("--display", action="store_true", help="Show live OpenCV window.")
    parser.add_argument("--no_display", dest="display", action="store_false", help="Disable live OpenCV window.")
    parser.set_defaults(display=True)

    args = parser.parse_args()

    if not args.run_sinth_samples and not args.video:
        parser.error("Provide --video for single-run mode, or set --run_sinth_samples.")

    try:
        pipeline = OnlineVideoInferencePipeline(
            model=args.model,
            dataset=args.dataset,
            exports_root=args.exports_root,
            device=args.device,
            seg_len=args.seg_len,
            batch_size=args.batch_size,
            pose_backend=args.pose_backend,
            pose_fallback_backend=args.pose_fallback_backend,
            pose_allow_fallback=not args.disable_pose_fallback,
            pose_device=args.pose_device,
            pose_scope=args.pose_scope,
            pose_det_model=args.pose_det_model,
            pose_det_weights=args.pose_det_weights,
            pose_kpt_thr=args.pose_kpt_thr,
            track_iou_th=args.track_iou_th,
            track_max_missing=args.track_max_missing,
            anomaly_threshold=args.anomaly_threshold,
        )

        if args.run_sinth_samples:
            result = pipeline.infer_sinth_samples(
                sinth_root=args.sinth_root,
                output_dir=args.output_dir,
                display=args.display,
                save_annotated_video=args.save_annotated_video,
                max_frames=args.max_frames,
            )
        else:
            result = pipeline.infer_video(
                video_path=args.video,
                output_dir=args.output_dir,
                display=args.display,
                save_annotated_video=args.save_annotated_video,
                max_frames=args.max_frames,
            )
    except RuntimeError as exc:
        parser.error(str(exc))

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
