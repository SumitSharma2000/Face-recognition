import cv2
import face_recognition
import numpy as np
import pandas as pd
import os
from datetime import datetime
from tkinter import Tk, Label, Button, messagebox

# Load known faces and their encodings
def load_known_faces(image_dir="known_faces"):
    known_encodings = []
    known_names = []

    # Get all image files from the known_faces directory
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for image_path in image_paths:
        # Load the image
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue

        # Convert the image to RGB (face_recognition expects RGB format)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Ensure the image is in the correct format
        if image_rgb.ndim == 3 and image_rgb.shape[2] == 3:  # RGB check
            # Find face encodings
            encodings = face_recognition.face_encodings(image_rgb)

            if encodings:
                known_encodings.append(encodings[0])
                # Assuming the name is part of the image filename (without the extension)
                known_names.append(image_path.split("\\")[-1].split(".")[0])
        else:
            print(f"Image {image_path} is not in the expected RGB format. Skipping.")

    return known_encodings, known_names

# Mark attendance in CSV
def mark_attendance(name):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    attendance = pd.read_csv("attendance.csv") if os.path.exists("attendance.csv") else pd.DataFrame(columns=["Name", "Date", "Time"])
    if not ((attendance["Name"] == name) & (attendance["Date"] == timestamp.split()[0])).any():
        attendance = attendance.append({"Name": name, "Date": timestamp.split()[0], "Time": timestamp.split()[1]}, ignore_index=True)
        attendance.to_csv("attendance.csv", index=False)
        messagebox.showinfo("Success", f"Attendance marked for {name}.")
    else:
        messagebox.showinfo("Info", f"Attendance for {name} is already marked today.")

# Open camera and show live feed
def capture_and_recognize():
    video_capture = cv2.VideoCapture(0)  # Open the camera

    if not video_capture.isOpened():
        messagebox.showerror("Error", "Unable to access the camera.")
        return

    while True:
        ret, frame = video_capture.read()  # Capture frame-by-frame
        if not ret:
            break

        # Display the frame in a window
        cv2.imshow('Camera - Press Q to Quit', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the window
    video_capture.release()
    cv2.destroyAllWindows()

# Reset attendance file
def reset_attendance():
    if os.path.exists("attendance.csv"):
        os.remove("attendance.csv")
    messagebox.showinfo("Reset", "Attendance has been reset.")

# View attendance records
def view_attendance():
    if os.path.exists("attendance.csv"):
        attendance = pd.read_csv("attendance.csv")
        messagebox.showinfo("Attendance Records", attendance.to_string(index=False))
    else:
        messagebox.showinfo("No Records", "No attendance records found.")

# Main UI
def main_ui():
    root = Tk()
    root.title("Face Recognition Attendance System")
    root.geometry("400x300")

    Label(root, text="Face Recognition Attendance", font=("Arial", 18)).pack(pady=20)
    Button(root, text="Mark Attendance", command=capture_and_recognize, width=20, height=2, bg="green", fg="white").pack(pady=10)
    Button(root, text="View Attendance", command=view_attendance, width=20, height=2, bg="blue", fg="white").pack(pady=10)
    Button(root, text="Reset Attendance", command=reset_attendance, width=20, height=2, bg="red", fg="white").pack(pady=10)
    Button(root, text="Exit", command=root.quit, width=20, height=2).pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main_ui()
