import cv2
import pytesseract
from pyzbar.pyzbar import decode

# Specify the path to tesseract executable (if required)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Function to decode barcodes using pyzbar
def read_barcodes(image):
    barcodes = decode(image)
    barcode_data_list = []
    
    for barcode in barcodes:
        # Extract barcode data and type
        barcode_data = barcode.data.decode("utf-8")
        barcode_type = barcode.type
        
        # Print the barcode info
        print(f"Found {barcode_type} barcode: {barcode_data}")
         
        # Store the barcode data in a list
        barcode_data_list.append((barcode_type, barcode_data))
    
    return barcode_data_list

# Load the image where the barcodes are present
image_path = '6.jpg'
image = cv2.imread(image_path)

# Convert the image to grayscale for better OCR results
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Read barcodes using pyzbar
barcodes = read_barcodes(image)

# If no barcodes were detected by pyzbar, use Tesseract as a fallback
if not barcodes:
    print("No barcodes detected using pyzbar. Trying with Tesseract OCR...")
    
    # Use Tesseract to read text from the grayscale image
    custom_config = r'--oem 3 --psm 6'  # Configure Tesseract
    ocr_result = pytesseract.image_to_string(gray_image, config=custom_config)
    
    print(f"OCR Result from Tesseract:\n{ocr_result}")
else:
    print(f"Total barcodes detected: {len(barcodes)}")

# Show the image with the barcodes highlighted (optional)

cv2.destroyAllWindows()

