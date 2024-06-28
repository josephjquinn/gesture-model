import os
import cv2
import pickle
import argparse
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def process_images(data_dir):
    IMAGE_FILES = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                file_path = os.path.join(root, file)
                IMAGE_FILES.append((file_path, root.split(os.sep)[-1]))

    landmark_data = []
    error = 0

    with mp_hands.Hands(
        static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7
    ) as hands:
        for idx, (file, label) in enumerate(IMAGE_FILES):
            image = cv2.flip(cv2.imread(file), 1)
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            print("Handedness:", results.multi_handedness)
            if not results.multi_hand_landmarks:
                error += 1
                continue

            image_height, image_width, _ = image.shape
            frame_landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * image_width)
                    y = int(landmark.y * image_height)
                    z = landmark.z
                    landmarks.append((x, y, z))
                frame_landmarks.append({"label": int(label), "landmarks": landmarks})

            landmark_data.append(frame_landmarks)

    output_file = "landmark_data.pickle"
    with open(output_file, "wb") as f:
        pickle.dump(landmark_data, f)

    print(f"Landmark data saved to {output_file}")
    print(f"Total files: {len(IMAGE_FILES)}")
    print(f"Number of landmarks: {len(landmark_data)}")
    print(f"Number of errors: {error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process images in a directory to extract hand landmarks."
    )
    parser.add_argument(
        "data_dir", type=str, help="Path to the directory containing image files"
    )

    args = parser.parse_args()

    process_images(args.data_dir)
