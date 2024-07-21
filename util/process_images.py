import os
import cv2
import argparse
from cvzone.HandTrackingModule import HandDetector

# Function to process images in a directory
def process_images(input_dir, offset=50):
    detector = HandDetector(maxHands=1, detectionCon=0.5)
    output_base_dir = os.path.join(os.path.dirname(input_dir), "cropped")
    os.makedirs(output_base_dir, exist_ok=True)

    for root, _, files in os.walk(input_dir):
        # Replicate directory structure in output
        relative_path = os.path.relpath(root, input_dir)
        output_dir = os.path.join(output_base_dir, relative_path)
        os.makedirs(output_dir, exist_ok=True)

        for file in files:
            if file.lower().endswith(("png", "jpg", "jpeg", "bmp", "tiff")):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)

                # Detect hand
                hands, _ = detector.findHands(img, flipType=False, draw=False)

                if hands:
                    hand = hands[0]
                    x, y, w, h = hand["bbox"]

                    # Crop image around hand
                    img_crop = img[
                        max(0, y - offset): y + h + offset,
                        max(0, x - offset): x + w + offset,
                    ]

                    # Save cropped image with same directory structure
                    output_path = os.path.join(output_dir, file)
                    cv2.imwrite(output_path, img_crop)

                    print(f"Processed: {img_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process images in a directory to extract hand landmarks."
    )
    parser.add_argument(
        "data_dir", type=str, help="Path to the directory containing image files"
    )

    args = parser.parse_args()

    process_images(args.data_dir)
