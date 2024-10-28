import cv2
import pandas as pd
from pyzbar.pyzbar import decode
from datetime import datetime
import numpy as np

# Function to enhance the image quality
def enhance_image(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply GaussianBlur to reduce noise and improve focus on barcode
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # Apply sharpening kernel
    sharpening_kernel = np.array([[-1, -1, -1], 
                                  [-1,  9, -1], 
                                  [-1, -1, -1]])
    sharpened = cv2.filter2D(blurred, -1, sharpening_kernel)
    # Apply adaptive thresholding to improve barcode contrast
    enhanced = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    return enhanced

# Function to decode barcodes from a frame
def decode_barcodes(frame):
    barcodes = decode(frame)
    decoded_info = []
    for barcode in barcodes:
        # Extract the barcode text
        text = barcode.data.decode('utf-8')
        decoded_info.append(text)
    return decoded_info

# Set up a dataframe to store decoded barcodes
data = {"Timestamp": [], "Barcode": []}

# Initialize the video capture
cap = cv2.VideoCapture(0)  # '0' for default camera

try:
    while True:
        # Capture a frame from the video
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # Enhance the frame quality
        enhanced_frame = enhance_image(frame)

        # Decode barcodes from the enhanced frame
        barcodes = decode_barcodes(enhanced_frame)

        # If barcodes found, save to dataframe
        if barcodes:
            for barcode in barcodes:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                data["Timestamp"].append(timestamp)
                data["Barcode"].append(barcode)
                print(f"Barcode detected: {barcode} at {timestamp}")

        # Display the captured frame for feedback (optional)
        cv2.imshow('Barcode Reader', enhanced_frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
except Exception as e:
    print(f"Error: {e}")
finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    # Save data to Excel file
    df = pd.DataFrame(data)
    df.to_excel('barcodes.xlsx', index=False)
    print("Barcodes saved to barcodes.xlsx")

