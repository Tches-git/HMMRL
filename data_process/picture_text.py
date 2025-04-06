import cv2
import numpy as np
import pytesseract
from PIL import Image

# Set the path to the Tesseract executable (Windows only; adjust if needed)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Example path for Windows

def preprocess_image(image):
    """Preprocess the image to improve OCR accuracy."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply adaptive thresholding to binarize the image
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # Denoise the image
    denoised = cv2.fastNlMeansDenoising(thresh)
    return denoised

def extract_text_from_image(image_path):
    try:
        # Read the original image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image at {image_path}")
            return

        # Create a copy for annotations
        annotated_image = image.copy()

        # Preprocess the image for better OCR results
        processed_image = preprocess_image(image)

        # Perform text recognition with Tesseract
        # Use PSM 6 (Assume a single uniform block of text) for better results; adjust as needed
        custom_config = r'--oem 3 --psm 6'
        results = pytesseract.image_to_data(processed_image, lang='eng+chi_sim', config=custom_config, output_type=pytesseract.Output.DICT)

        # Extract text and bounding boxes
        print("Extracted text:")
        extracted_texts = []
        for i in range(len(results['text'])):
            text = results['text'][i].strip()
            conf = float(results['conf'][i])
            if text and conf > 20:  # Filter low-confidence results
                print(f"Text: {text}, Confidence: {conf:.2f}")
                extracted_texts.append(f"{text} (Conf: {conf:.2f})")

                # Draw bounding box and text on the annotated image
                x, y, w, h = results['left'][i], results['top'][i], results['width'][i], results['height'][i]
                top_left = (x, y)
                bottom_right = (x + w, y + h)
                cv2.rectangle(annotated_image, top_left, bottom_right, (0, 255, 0), 2)
                cv2.putText(annotated_image, text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Create a blank canvas for displaying extracted text
        text_canvas_height = max(100, 50 * len(extracted_texts))
        text_canvas = np.ones((text_canvas_height, 600, 3), dtype=np.uint8) * 255  # White background
        for i, text in enumerate(extracted_texts):
            cv2.putText(text_canvas, text, (10, 30 + i * 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Resize images to a smaller size
        scale_percent = 50
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        resized_annotated_image = cv2.resize(annotated_image, dim, interpolation=cv2.INTER_AREA)
        text_canvas_resized = cv2.resize(text_canvas, (600, max(height, text_canvas.shape[0])), interpolation=cv2.INTER_AREA)

        # Combine all images into one canvas
        combined_height = max(resized_image.shape[0], resized_annotated_image.shape[0], text_canvas_resized.shape[0])
        combined_width = resized_image.shape[1] + resized_annotated_image.shape[1] + text_canvas_resized.shape[1]
        
        # Limit canvas size to fit screen
        max_display_width = 1600
        max_display_height = 900
        if combined_width > max_display_width or combined_height > max_display_height:
            scale = min(max_display_width / combined_width, max_display_height / combined_height)
            combined_width = int(combined_width * scale)
            combined_height = int(combined_height * scale)
            resized_image = cv2.resize(resized_image, (int(resized_image.shape[1] * scale), int(resized_image.shape[0] * scale)))
            resized_annotated_image = cv2.resize(resized_annotated_image, (int(resized_annotated_image.shape[1] * scale), int(resized_annotated_image.shape[0] * scale)))
            text_canvas_resized = cv2.resize(text_canvas_resized, (int(text_canvas_resized.shape[1] * scale), int(text_canvas_resized.shape[0] * scale)))

        combined_canvas = np.ones((combined_height, combined_width, 3), dtype=np.uint8) * 255
        combined_canvas[:resized_image.shape[0], :resized_image.shape[1]] = resized_image
        combined_canvas[:resized_annotated_image.shape[0], resized_image.shape[1]:resized_image.shape[1]+resized_annotated_image.shape[1]] = resized_annotated_image
        combined_canvas[:text_canvas_resized.shape[0], resized_image.shape[1]+resized_annotated_image.shape[1]:] = text_canvas_resized

        # Save the combined image
        output_path = 'output_image_tesseract.jpg'
        cv2.imwrite(output_path, combined_canvas)
        print(f"Combined image saved to {output_path}")

        # Display the combined image
        cv2.namedWindow('Combined Image', cv2.WINDOW_NORMAL)
        cv2.imshow('Combined Image', combined_canvas)
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Ensure Tesseract is installed and the path to tesseract.exe is correct.")
        print("Try running: pip install --upgrade pytesseract opencv-python")

# Example usage
if __name__ == "__main__":
    # Replace with your image path
    image_path = r"C:\Users\28489\Desktop\paired\8\1.jpg"
    extract_text_from_image(image_path)