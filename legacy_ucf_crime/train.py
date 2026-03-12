from model import XceptionTime
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parents[1]

    # Initialize and train the XceptionTime model
    csv_paths = [
        str(repo_root / "dataset" / "ucf-crime_dataset.csv"),  # anomalies
        str(repo_root / "dataset" / "ucf-crime_dataset-normal.csv"),  # normal
    ]
    
    xception_model = XceptionTime(csv_paths=csv_paths)
    xception_model.train()
    
    # Perform inference on the trained model
    xception_model.infer()


if __name__ == "__main__":
    main()
