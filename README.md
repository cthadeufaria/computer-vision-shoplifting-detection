# computer-vision-shoplifting-detection
computer-vision-shoplifting-detection

## Overview

A simple computer vision project for detecting shoplifting (and other crime classes) using the UCF Crime dataset. The repo contains code to prepare a tabular dataset from video data and training code that consumes that tabular dataset.

## What this code does

- Scans the dataset folders (see above) and extracts features / metadata into a tabular CSV used for training.
- Trains a model using the prepared tabular CSV.
- Evaluation and inference utilities are included to inspect results.

Important: training depends on the tabular dataset being populated first. The training pipeline currently assumes the preprocessing step has completed successfully.

## Requirements

- Python 3.8+
- Install dependencies:
    - pip install -r requirements.txt

## Quick start

1. Put the dataset under the `dataset/` folder following the structure already listed.
2. Populate the tabular dataset (preprocessing). Example:
        - python prepare_tabular.py --dataset dataset/ --output data/tabular.csv
        - This step extracts frames/features and creates `data/tabular.csv`.
3. Train the model using the generated CSV:
        - python train.py --data data/tabular.csv --output_dir models/
4. (Optional) Evaluate / run inference:
        - python eval.py --model models/latest.pt --data data/tabular.csv

Adjust script names/options if your repo uses different filenames or CLI flags.

## Notes

- If preprocessing fails or `data/tabular.csv` is missing/empty, training will not start.
- Preprocessing can be slow depending on the number/length of videos and available hardware.
- Use smaller subsets during development to iterate faster.

## Troubleshooting

- Check logs from the preprocessing step for missing files or permission errors.
- Verify Python environment and that all dependencies installed from `requirements.txt`.

## License / Contact

- See repository root for license and contributor information.
- For quick help, open an issue with the log output and the command you ran.

## Video Dataset Structure

Original dataset info: [UCF Crime Dataset (Kaggle)](https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset)

Expected directory layout:

```
dataset/
├─ Vandalism/
├─ Stealing/
├─ Shoplifting/
├─ Shooting/
├─ Robbery/
├─ RoadAccidents/
├─ Fighting/
├─ Explosion/
├─ Burglary/
├─ Assault/
├─ Arson/
├─ Arrest/
└─ Abuse/
```

Notes:
- Each folder contains videos for the corresponding crime class (one class per folder).
- Keep folder names consistent with the table-of-contents above to avoid label mismatches during preprocessing.
- If using a subset for development, copy only the folders you need and update the preprocessing script paths accordingly.
- Ensure you have permission to use the dataset and follow any license/usage terms on the Kaggle page.