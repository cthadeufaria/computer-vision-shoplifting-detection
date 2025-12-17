from model import XceptionTime


def main():
    # Initialize and train the XceptionTime model
    csv_paths = [
        "dataset/ucf-crime_dataset.csv",  # anomalies
        "dataset/ucf-crime_dataset-normal.csv",  # normal
    ]
    
    xception_model = XceptionTime(csv_paths=csv_paths)
    xception_model.train()
    
    # Perform inference on the trained model
    xception_model.infer()


if __name__ == "__main__":
    main()