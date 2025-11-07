import os
import cv2
import argparse
from processing_utils import normalize_image, threshold_image, morfological

def main(image_dir, output_image_dir, threshold):
    os.makedirs(output_image_dir, exist_ok=True)
    for image in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image)
        normalized_image = normalize_image(image_path)
        binary_image = threshold_image(normalized_image, threshold)
        image_open_closed = morfological(binary_image)
        output_image_path = os.path.join(output_image_dir, image)
        cv2.imwrite(output_image_path, image_open_closed)
        print(f"Processed image saved in {output_image_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processing images")
    parser.add_argument("-i", "--input", default="images", help="Image input directory")
    parser.add_argument("-t", "--threshold", type=int, default=127, help="Threshold value")
    args = parser.parse_args()
    image_dir = args.input
    threshold = args.threshold
    output_image_dir = f"output_thres_{threshold}"
    main(image_dir, output_image_dir, threshold)
