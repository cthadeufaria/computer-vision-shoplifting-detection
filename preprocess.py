from model import Tracker
import cv2 as cv


def main():
    # Initialize the people tracking model
    people_tracker = Tracker()
    i = 0

    videos_to_process = [
        "Shoplifting",
        "Shopping"  # TODO: finish shopping category with videos part 2
    ]  # TODO: check usage of another dataset presented in the paper https://doi.org/10.3390/app13148341.

    with open('./dataset/Anomaly_Train.txt', 'r') as f:
        videos = f.read().split('\n')

    # Load the videos for saving bounding boxes into dataset
    for video in videos:
        i += 1
        print(f"Processing video: {i}")

        label = video.split('/')[0]
        print(label)
        name = video.split('/')[1]

        if label not in videos_to_process: 
            print(f"Skipping, {label}, {video}.")
            continue

        cap = cv.VideoCapture('./dataset/' + video)

        if not cap.isOpened():
            print(f"Failed to load video: {video}")
            continue

        while True:
            success, frame = cap.read()
            # cv.imshow("Video", frame)  # for debug

            n = cap.get(cv.CAP_PROP_POS_FRAMES)

            if not success:
                break

            # Perform object tracking and save results into dataset
            people_tracker.save_to_dataset(frame, i, n, label, name)

            # Wait for key press to exit
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()

    # TODO: use the tabular dataset for training the second stage model.


if __name__ == "__main__":
    main()