# Shoplifting Detection AI Vision Model

A simple computer vision project for detecting shoplifting using the UCF Crime dataset. The repo contains code to prepare a tabular dataset from video data and training code that consumes that tabular dataset.

## What this code does

- Scans the dataset folders (see below) and extracts features / metadata into a tabular CSV used for training.

## What's to be done

- Train a model using the prepared tabular CSV.
- Evaluate and run inference using the provided utilities.

## Quick start

1. Downlaod the UCF-Crime dataset at https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AABvnJSwZI7zXb8_myBA0CLHa?dl=0 
2. Put the dataset under the `dataset/` folder following the structure listed below. Create at least one of the two folders: `Shoplifting/` and/or `Shopping/` with mp4 videos inside. Note: Use the Anomaly_Train.txt file from this repository if using the `Shopping/` folder.
3. Create a virtual environment and activate it:
        
        python -m venv .venv
        source .venv/bin/activate   (Linux / Mac)
        .venv\Scripts\activate      (Windows)

4. Install dependencies:
        
        pip install -r requirements.txt

5. Create and populate the tabular dataset (preprocessing) by running:

        python3 preprocess.py

- This step uses YOLO model to classify and extract bounding boxes of people from the video frames, creates the tabular metadata and save it as csv file for further use.

## For future implementations
3. Train the model using the generated CSV:
        - python train.py --data data/tabular.csv --output_dir models/
4. Evaluate / run inference:
        python3 eval.py --model models/latest.pt --data data/tabular.csv

## Video Dataset Structure

Original dataset info: [UCF Crime Dataset](https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AABvnJSwZI7zXb8_myBA0CLHa?dl=0)

Expected dataset directory layout for shoplifting detection model:

```
dataset/
├─ Shoplifting/
├─ Shopping/
```

## License / Contact

- See repository root for license and contributor information.