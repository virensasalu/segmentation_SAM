import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_labelme_annotation(json_file):
    """Load annotation data from a LabelMe JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def draw_annotations(image, annotation):
    """Draw annotated polygons from the LabelMe JSON on the image."""
    for shape in annotation.get('shapes', []):
        points = np.array(shape['points'], dtype=np.int32)
        # Draw the polygon on the image (green color, thickness=2)
        cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)
    return image

def main():
    # Update these paths to your image and its corresponding LabelMe JSON file.
    image_path = "path/to/your/image.jpg"
    json_path = "path/to/your/annotation.json"

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load the image. Please check the path.")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load LabelMe annotation
    annotation = load_labelme_annotation(json_path)

    # Draw annotations
    annotated_image = draw_annotations(image.copy(), annotation)

    # Display the image with annotations
    plt.figure(figsize=(10, 10))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title("LabelMe Annotations")
    plt.show()

if __name__ == '__main__':
    main()
    