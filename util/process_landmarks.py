import os
import cv2
import pickle
import argparse
import mediapipe as mp

mp_hands = mp.solutions.hands


def process_images(data_dir):
    IMAGE_FILES = []
    for subdir in sorted(os.listdir(data_dir)):
        subdir_path = os.path.join(data_dir, subdir)
        if os.path.isdir(subdir_path):
            image_files = []
            for file in os.listdir(subdir_path):
                if file.endswith(".jpg") or file.endswith(".png"):
                    file_path = os.path.join(subdir_path, file)
                    image_files.append(file_path)
            IMAGE_FILES.append(image_files)

    landmark_data = []
    error = 0

    with mp_hands.Hands(
        static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7
    ) as hands:
        for idx, subdir_files in enumerate(IMAGE_FILES):
            for file in subdir_files:
                image = cv2.flip(cv2.imread(file), 1)
                results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                if not results.multi_hand_landmarks:
                    error += 1
                    continue
                print(file)

                image_height, image_width, _ = image.shape
                frame_landmarks = []
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        # Normalize coordinates between 0 and 1
                        x = landmark.x * image_width / image_width
                        y = landmark.y * image_height / image_height
                        z = landmark.z
                        landmarks.append((x, y, z))
                    frame_landmarks.append({"label": idx, "landmarks": landmarks})

                landmark_data.append(frame_landmarks)

    output_file = "landmark_data.pickle"
    with open(output_file, "wb") as f:
        pickle.dump(landmark_data, f)

    print(f"Landmark data saved to {output_file}")
    print(f"Total files: {sum(len(subdir_files) for subdir_files in IMAGE_FILES)}")
    print(f"Number of landmarks generated: {len(landmark_data)}")
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
