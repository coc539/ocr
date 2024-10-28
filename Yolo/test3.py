import cv2
from pyzbar.pyzbar import decode

def BarcodeReader(image):
    img = cv2.imread(image)
    detectedBarcodes = decode(img)

    min_width = 10  # Minimum width threshold for barcode region
    min_height = 10  # Minimum height threshold for barcode region
    
    if not detectedBarcodes:
        print("Barcode Not Detected or your barcode is blank/corrupted!")
    else:
        for barcode in detectedBarcodes:
            (x, y, w, h) = barcode.rect
            
            if w > min_width and h > min_height:
                cv2.rectangle(img, (x-10, y-10), (x + w+10, y + h+10), (255, 0, 0), 2)
                if barcode.data != "":
                    print(barcode.data)
                    print(barcode.type)

    # Save the image with the highlighted rectangle
    output_image = image.replace(".png", "_output.jpeg")
    cv2.imwrite(output_image, img)
    print("Output image saved as:", output_image)

    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image = "5.png"
    BarcodeReader(image)

for i in range :
    i = i+1