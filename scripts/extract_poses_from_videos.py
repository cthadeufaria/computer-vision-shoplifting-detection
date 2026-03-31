#!/usr/bin/env python3
"""
Extract 17-keypoint COCO poses from a directory of MP4 videos and save as
PoseLift-compatible pickle files.

Uses YOLOv8-pose (ultralytics) for pose estimation — no mmcv/mmpose required.
Model weights are downloaded automatically on first run (~6 MB for yolov8n-pose).

Output format per file  (one .pkl per video):
    {frame_num: {person_id: [bbox(4,), keypoints(17,3)]}}
    keypoints columns: [x_pixel, y_pixel, confidence]

Output filenames follow the PoseLift convention  "sceneID_clipID.pkl" so the
resulting files can be read directly by PoseLiftDataset without modification.
Videos are enumerated in sorted order; the caller supplies the scene_id prefix
via --scene_id (default 0).

Usage:
    # sinth Normal
    python scripts/extract_poses_from_videos.py \\
        --input_dir  data/sinth/Normal \\
        --output_dir data/sinth/Pickle_files/Normal \\
        --device mps

    # test run on first 3 videos
    python scripts/extract_poses_from_videos.py \\
        --input_dir  data/sinth/Normal \\
        --output_dir data/sinth/Pickle_files/Normal \\
        --limit 3
"""

import argparse
import os
import pickle
import sys
import warnings
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

# Allow importing IoU tracker from pipeline/ regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "pipeline"))

from video_inference_pipeline import SimpleIoUTracker, PoseDetection


# ---------------------------------------------------------------------------
# YOLO-based pose estimator (replaces MMPosePoseEstimator)
# ---------------------------------------------------------------------------

class YOLOPoseEstimator:
    """
    Pose estimator backed by YOLOv8-pose (ultralytics).

    Returns the same List[PoseDetection] interface as MMPosePoseEstimator so
    the rest of the extraction pipeline is unchanged.

    Parameters
    ----------
    model_name : str
        One of: 'yolov8n-pose', 'yolov8s-pose', 'yolov8m-pose', 'yolov8l-pose'.
        Weights are downloaded automatically on first use.
    device : str
        'cpu', 'mps', 'cuda:0', etc.
    conf_thr : float
        Minimum person detection confidence to keep.
    """

    def __init__(
        self,
        model_name: str = "yolov8n-pose",
        device: str = "cpu",
        conf_thr: float = 0.25,
    ):
        from ultralytics import YOLO  # type: ignore

        self.model = YOLO(f"{model_name}.pt")
        self.device = device
        self.conf_thr = conf_thr

    def infer_pose(self, frame_bgr: np.ndarray) -> List[PoseDetection]:
        results = self.model(
            frame_bgr,
            device=self.device,
            verbose=False,
            conf=self.conf_thr,
        )
        out: List[PoseDetection] = []
        for result in results:
            if result.keypoints is None or result.boxes is None:
                continue
            for box, kps in zip(result.boxes, result.keypoints):
                bbox_xyxy = box.xyxy[0].cpu().numpy().astype(np.float32)  # (4,)
                det_conf  = float(box.conf[0].cpu())

                xy   = kps.xy[0].cpu().numpy().astype(np.float32)   # (17, 2)
                conf = kps.conf[0].cpu().numpy().astype(np.float32) # (17,)
                kp17 = np.concatenate([xy, conf[:, None]], axis=1)  # (17, 3)

                mean_conf  = float(conf.mean())
                pose_score = mean_conf * 3.0  # match STG-NF convention

                out.append(
                    PoseDetection(
                        bbox=tuple(bbox_xyxy.tolist()),
                        conf=det_conf,
                        keypoints17=kp17,
                        pose_score=pose_score,
                    )
                )
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sorted_video_paths(input_dir: str) -> list:
    """Return sorted list of video paths in input_dir."""
    exts = {".mp4", ".avi", ".mov", ".mkv"}
    paths = sorted(
        p for p in Path(input_dir).iterdir()
        if p.suffix.lower() in exts
    )
    return [str(p) for p in paths]


def _detect_vid_res(video_path: str) -> Tuple[int, int]:
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


def extract_video(
    video_path: str,
    estimator: YOLOPoseEstimator,
) -> dict:
    """
    Process a single video and return a pickle-format dict.

    Returns
    -------
    clip_data : {frame_num: {person_id: [bbox(4,), kp(17,3)]}}
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    tracker = SimpleIoUTracker(iou_th=0.3, max_missing=6)
    clip_data: dict = {}
    frame_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        detections = estimator.infer_pose(frame_bgr)
        if detections:
            assignments = tracker.update(frame_idx, detections)
            persons = {}
            for track_id, det in assignments:
                bbox_arr = np.array(det.bbox, dtype=np.float32)  # (4,)
                kp_arr   = det.keypoints17.copy()                  # (17, 3)
                persons[track_id] = [bbox_arr, kp_arr]
            if persons:
                clip_data[frame_idx] = persons

        frame_idx += 1

    cap.release()
    return clip_data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract COCO-17 poses from videos → PoseLift pickle format (uses YOLOv8-pose)"
    )
    parser.add_argument("--input_dir",  required=True, help="Directory containing input video files")
    parser.add_argument("--output_dir", required=True, help="Directory to write .pkl files")
    parser.add_argument("--scene_id",   type=int, default=0,
                        help="Scene ID prefix for output filenames (default: 0)")
    parser.add_argument("--device",     default="cpu",
                        help="Compute device: cpu / cuda:0 / mps  (default: cpu)")
    parser.add_argument("--model",      default="yolov8n-pose",
                        choices=["yolov8n-pose", "yolov8s-pose", "yolov8m-pose", "yolov8l-pose"],
                        help="YOLOv8-pose model size (default: yolov8n-pose)")
    parser.add_argument("--conf",       type=float, default=0.25,
                        help="Minimum detection confidence (default: 0.25)")
    parser.add_argument("--limit",      type=int, default=None,
                        help="Process at most this many videos (for testing)")
    args = parser.parse_args()

    video_paths = _sorted_video_paths(args.input_dir)
    if not video_paths:
        print(f"No video files found in {args.input_dir}")
        sys.exit(1)

    if args.limit is not None:
        video_paths = video_paths[: args.limit]

    os.makedirs(args.output_dir, exist_ok=True)

    w, h = _detect_vid_res(video_paths[0])
    print(f"Auto-detected video resolution: {w}x{h}")
    print(f"Initialising YOLOv8-pose ({args.model}) on device={args.device} ...")

    estimator = YOLOPoseEstimator(
        model_name=args.model,
        device=args.device,
        conf_thr=args.conf,
    )

    total_frames_with_poses = 0
    for idx, vpath in enumerate(video_paths, start=1):
        clip_id  = idx  # 1-based sequential clip ID
        out_name = f"{args.scene_id}_{clip_id}.pkl"
        out_path = os.path.join(args.output_dir, out_name)

        print(f"[{idx}/{len(video_paths)}] {Path(vpath).name} → {out_name}", end=" ... ", flush=True)

        clip_data = extract_video(vpath, estimator)

        with open(out_path, "wb") as fh:
            pickle.dump(clip_data, fh)

        n = len(clip_data)
        print(f"{n} frames with poses")
        total_frames_with_poses += n

    print(f"\nDone. Processed {len(video_paths)} videos → {args.output_dir}")
    print(f"Total frames with at least one person detected: {total_frames_with_poses}")


if __name__ == "__main__":
    main()
