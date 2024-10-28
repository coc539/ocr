from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import customtkinter as tk
from ultralytics import YOLO
import cvzone
import math
import pytesseract
import os
from datetime import datetime
from openpyxl import Workbook
from pyzbar.pyzbar import decode

class WebcamApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        window.geometry("1200x785")
        window.minsize(1200, 785)

        # Initialize OpenCV's VideoCapture
        self.cap = cv2.VideoCapture(0)  # Change the number as per your webcam index
        
        self.desired_width = 1100  # Set your desired width
        self.desired_height = 580 
        
        # Initialize YOLO model
        self.model = YOLO("best.pt")
        self.class_names = ['Label']

        # Create folder to store detected label images
        self.output_folder = "detected_labels"
        os.makedirs(self.output_folder, exist_ok=True)

        # Initialize Excel file and workbook
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.excel_file = f"label_texts_{timestamp}.xlsx"
        self.workbook = Workbook()
        self.sheet = self.workbook.active
        self.sheet.title = "Extracted Texts"
        self.sheet.append(["Image Name", "Extracted Text"])  # Header for the Excel file

        main_frame = tk.CTkFrame(master=window)
        main_frame.pack(fill="both", expand="True")

        camera_frame = tk.CTkFrame(master=main_frame, width=840)  # 70% of 1200px
        camera_frame.pack(side="top", fill="both", expand=True)

        button_frame = tk.CTkFrame(master=main_frame,width=360)  # 30% of 1200px
        button_frame.pack(side="bottom", fill="x")
        
        self.lbl_video = tk.CTkLabel(camera_frame, text="", width=self.desired_width, height=self.desired_height)
        self.lbl_video.pack(fill="none", expand=True)
        
        self.btn_start_stop = tk.CTkButton(button_frame, text="Start", font=("comicsansms", 28), text_color="white", fg_color="#9F2B68", width=175, height=75, hover_color="#C3B1E1", command=self.toggle_capture)
        self.btn_start_stop.pack(padx=200,pady=50,side="left")

        button2 = tk.CTkButton(button_frame, text="Open File", font=("comicsansms", 28), text_color="white", fg_color="#9F2B68", width=175, height=75, hover_color="#C3B1E1", command=self.open_file)
        button2.pack(padx=200,pady=50,side="right")

        self.is_capturing = False
        self.update_video()

    def update_video(self):
        if self.is_capturing:
            ret, frame = self.cap.read()
            if ret:
                # Object Detection
                results = self.model(frame, stream=True)
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        w, h = x2 - x1, y2 - y1 
                        cvzone.cornerRect(frame, (x1, y1, w, h))

                        conf = math.ceil((box.conf[0] * 100)) / 100
                        cls = int(box.cls[0])
                        label = f'{self.class_names[cls]} {conf}'
                        cvzone.putTextRect(frame, label, (max(0, x1), max(35, y1)), scale=1, thickness=1)
                        
                        # Capture and save detected label image
                        label_image = frame[y1:y2, x1:x2]
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        image_path = os.path.join(self.output_folder, f'label_{timestamp}.png')
                        cv2.imwrite(image_path, label_image)
                        
                        # Perform Tesseract OCR on the saved image
                        extracted_text = self.extract_text_from_image(image_path)
                        print(f"Extracted Text: {extracted_text}")
                        
                        # Save image name and extracted text to Excel
                        self.save_to_excel(image_path, extracted_text)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.desired_width, self.desired_height))
                frame = Image.fromarray(frame)
                frame = ImageTk.PhotoImage(image=frame)
                self.lbl_video.configure(image=frame)
                self.lbl_video.image = frame

        self.window.after(10, self.update_video)

    def extract_text_from_image(self, image_path):
        # Use Tesseract to extract text from the saved image
        text = pytesseract.image_to_string(image_path)
        return text

    def save_to_excel(self, image_path, extracted_text):
        # Extract the image filename (without the full path) for the Excel record
        image_name = os.path.basename(image_path)
        # Append the image name and extracted text to the Excel file
        self.sheet.append([image_name, extracted_text])
        # Save the Excel file after each entry
        self.workbook.save(self.excel_file)

    def open_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", ".mp4;.avi")])
        if file_path:
            self.cap = cv2.VideoCapture(file_path)
            self.btn_start_stop.config(state=tk.NORMAL)
            messagebox.showinfo("File Opened", f"Opened file: {file_path}")

    def start_capture(self): 
        self.is_capturing = True
        self.btn_start_stop.config(text="Stop")
        self.capture_thread = threading.Thread(target=self.capture_frames)
        self.capture_thread.start()

    def stop_capture(self):
        self.is_capturing = False
        self.btn_start_stop.config(text="Start")
        self.cap.release()

    def toggle_capture(self):
        if self.is_capturing:
            self.stop_capture()
        else:
            self.start_capture()

    def capture_frames(self):
        while self.is_capturing:
            ret, frame = self.cap.read()

            if not ret:
                messagebox.showerror("Error", "Failed to capture frame from webcam.")
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        self.cap.release()

# Create a tkinter GUI window
window = tk.CTk()
app = WebcamApp(window, "Webcam Capture")

def close_app():
    window.cap.release()  # Release the webcam
    window.destroy()  # Destroy the tkinter window

# Bind the close_app function to the window close button
window.protocol("WM_DELETE_WINDOW", close_app)

# Handle closing of the tkinter window
def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        window.destroy()

window.protocol("WM_DELETE_WINDOW", on_closing)
                                                         
# Run the tkinter main loop
window.mainloop()
