import easyocr
import cv2
import numpy as np
from PIL import Image

def extract_text_from_image(image_path):
    try:
        # Initialize EasyOCR reader
        reader = easyocr.Reader(['en', 'ch_sim'], gpu=False)  # Set gpu=True if GPU is available

        # Read the original image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image at {image_path}")
            return

        # Create a copy for annotations to preserve the original image
        annotated_image = image.copy()

        # Perform text recognition
        results = reader.readtext(image)

        # Print extracted text
        print("Extracted text:")
        extracted_texts = []
        for (bbox, text, prob) in results:
            print(f"Text: {text}, Confidence: {prob:.2f}")
            extracted_texts.append(f"{text} (Conf: {prob:.2f})")

            # Extract bounding box coordinates for annotated image
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = (int(top_left[0]), int(top_left[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))

            # Draw rectangle and text on the annotated image
            cv2.rectangle(annotated_image, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(annotated_image, text, (top_left[0], top_left[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Create a blank canvas for displaying extracted text
        text_canvas_height = max(100, 50 * len(extracted_texts))  # Dynamic height based on text lines
        text_canvas = np.ones((text_canvas_height, 600, 3), dtype=np.uint8) * 255  # White background
        for i, text in enumerate(extracted_texts):
            cv2.putText(text_canvas, text, (10, 30 + i * 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Save the annotated image
        output_path = 'output_image.jpg'
        cv2.imwrite(output_path, annotated_image)
        print(f"Annotated image saved to {output_path}")

        # Display original image, annotated image, and extracted text
        cv2.imshow('Original Image', image)
        cv2.imshow('Annotated Image', annotated_image)
        cv2.imshow('Extracted Text', text_canvas)

        # Wait for a key press and close all windows
        print("Press any key to close the windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        if "ANTIALIAS" in str(e):
            print("Hint: This error may be due to an outdated Pillow version or deprecated code.")
            print("Try running: pip install --upgrade pillow easyocr")
            print("Or downgrade Pillow: pip install pillow==9.5.0")

# Example usage
if __name__ == "__main__":
    # Replace with your image path
    image_path = r"C:\Users\28489\Desktop\paired\7\3.jpg"
    extract_text_from_image(image_path)