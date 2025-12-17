# import torch
import os
import numpy as np
import pandas as pd
from ultralytics import YOLO
from dataset import BBox
from dataclasses import dataclass
from dataclass_csv import DataclassWriter
from tsai.all import *
from tsai.basics import *
from tsai.inference import load_learner


class Tracker:
    def __init__(self, model_path="./models/crowdhuman_yolov5m.pt"):
        # https://github.com/MahenderAutonomo/yolov5-crowdhuman?tab=readme-ov-file
        # self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)  # https://stackoverflow.com/questions/70167811/how-to-load-custom-model-in-pytorch
        self.model = YOLO("./models/yolov5mu.pt")

        self.anomalies = [
            "Abuse",
            "Arrest",
            "Arson",
            "Assault",
            "Burglary",
            "Explosion",
            "Fighting",
            "RoadAccidents",
            "Robbery",
            "Shooting",
            "Shoplifting",
            "Stealing",
            "Vandalism"
        ]

    def get_boxes(self, frame):
        # Get bounding boxes info for tracked hands  / reference for person only class tracking in ultralytics: https://github.com/orgs/ultralytics/discussions/10074
        results = self.model.track(frame, persist=True, show=False, classes=[0], verbose=False)  # TODO: implement DeepSort tracking for custom model / https://github.com/levan92/deep_sort_realtime

        return results[0].boxes

    def save_to_dataset(self, frame, i, n, label, name):
        boxes = self.get_boxes(frame)

        if not boxes.is_track:
            return

        if label in self.anomalies:
            is_anomaly = True
            path = "dataset/ucf-crime_dataset.csv"
        else:
            is_anomaly = False
            path = "dataset/ucf-crime_dataset-normal.csv"

        # Save bounding boxes into dataset
        data = [BBox(
            clip=i,
            name=name,
            frame=int(n),
            person=float(box.id),
            left=float(box.xywhn[0][0]),
            top=float(box.xywhn[0][1]),
            width=float(box.xywhn[0][2]),
            height=float(box.xywhn[0][3]),
            is_anomaly=is_anomaly,
            anomaly=label
        ) for box in boxes]

        # with open("ucf-crime_dataset.csv", "r", newline="") as f:  # TODO: do not write if same data is already in the dataset. find faster solution.
        #     reader = DataclassReader(f, BBox)

        #     l = [box for box in reader]

        #     for box in data:

        #         if box in l:
        #             return

        with open(path, "a", newline='') as f:
            writer = DataclassWriter(f, data, BBox)
            writer.write(skip_header=True)


class XceptionTime:  # TODO: implement XceptionTime model for time series classification of bounding box data.
    def __init__(self, csv_paths=None, seq_len=64, stride=32, min_frames=16):
        X, y, splits = get_classification_data('ECG200', split_data=False)
        self.X, self.y, self.splits = self._load_ucfcrime(csv_paths, seq_len=seq_len, stride=stride, min_frames=min_frames)

    def _load_ucfcrime(self, csv_paths, seq_len=64, stride=32, min_frames=16):
        """Create (X, y, splits) for TSClassifier from bounding-box CSVs.

        Returns
        -------
        X : np.ndarray (n_samples, n_channels, seq_len)
        y : np.ndarray (n_samples,)
        splits : tuple(list[int], list[int]) train/valid indices
        """

        dfs = []
        
        for path in csv_paths:            
            dfs.append(pd.read_csv(path))

        df = pd.concat(dfs, ignore_index=True)

        # Normalize dtypes
        df["frame"] = df["frame"].astype(int)
        df["is_anomaly"] = df["is_anomaly"].astype(bool)
        df.sort_values(["clip", "name", "frame", "person"], inplace=True)

        X = np.stack(df)  # (n_samples, seq_len, n_features)
        y = np.array(labels, dtype=int)  # 1 for anomaly, 0 for normal

        # tsai expects (n_samples, n_channels, seq_len)
        X = np.transpose(X, (0, 2, 1))

        # Stratified split
        splits = get_splits(y, valid_size=0.2, shuffle=True, stratify=y)

        return X, y, splits

    def train(self):
        tfms = [None, TSClassification()]
        batch_tfms = TSStandardize()
        clf = TSClassifier(self.X, self.y, splits=self.splits, path='models', arch="XceptionTime", tfms=tfms, batch_tfms=batch_tfms, metrics=accuracy)
        clf.fit_one_cycle(200, 3e-4)
        clf.export("XceptionTime.pkl")

    def infer(self):
        clf = load_learner("models/XceptionTime.pkl")
        probs, target, preds = clf.get_X_preds(self.X[self.splits[1]], self.y[self.splits[1]])