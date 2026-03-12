from model import PeopleTracker
import cv2 as cv


def main():
    # Load the videos for inference
    video_path = './challenge_hands/tarefas_cima.mp4'  # TODO: get correct videos
    cap = cv.VideoCapture(video_path)

    # Initialize the people tracking model
    people_tracker = PeopleTracker()
    people_tracker.create_dataset()
    
    if not cap.isOpened():
        print(f"Failed to load video: {video_path}")
        return

    while True:
        success, frame = cap.read()
        t = cap.get(cv.CAP_PROP_POS_MSEC)
        if not success:
            break
        
        # Perform object tracking and analyse results
        hand_detector.track(frame, t)

        # Wait for key press to exit
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()


if __name__ == "__main__":
    main()