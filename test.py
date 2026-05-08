import cv2
import pickle
import os
import numpy as np
import csv
import time
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier

print("=== Smart Attendance System ===")
print("=" * 40)

# PATHS
DATA_FOLDER = "data"
ATTENDANCE_FOLDER = "Attendance"

# Load database
try:
    with open(os.path.join(DATA_FOLDER, 'faces_data.pkl'), 'rb') as f:
        FACES = pickle.load(f)
    with open(os.path.join(DATA_FOLDER, 'names.pkl'), 'rb') as f:
        LABELS = pickle.load(f)
    
    print(f"Database: {len(FACES)} samples for {len(set(LABELS))} person(s)")
    print("Registered persons:", set(LABELS))
except:
    print("ERROR: Database not found or corrupted!")
    print("Run: cd data && python add_faces.py")
    exit()

# Train classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Load face detector
facedetect = cv2.CascadeClassifier(os.path.join(DATA_FOLDER, 'haarcascade_frontalface_default.xml'))

# Start camera
video = cv2.VideoCapture(0)
print("\n✅ Camera ready")
print("   Press 'o' to save attendance")
print("   Press 'q' to quit")
print("=" * 40)

# Variables for attendance
last_save_time = time.time()
save_cooldown = 2  # Minimum seconds between saves

while True:
    ret, frame = video.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    
    current_names = []
    
    for (x, y, w, h) in faces:
        # Process face
        face_img = frame[y:y+h, x:x+w]
        resized_face = cv2.resize(face_img, (50, 50)).flatten().reshape(1, -1)
        
        try:
            # Predict with confidence
            prediction = knn.predict(resized_face)[0]
            probabilities = knn.predict_proba(resized_face)[0]
            confidence = max(probabilities)
            
            # Only accept if confidence > 70%
            if confidence > 0.7:
                current_names.append(prediction)
                
                # Draw on frame
                color = (0, 255, 0)  # Green
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.rectangle(frame, (x, y-35), (x+w, y), color, -1)
                cv2.putText(frame, f"{prediction} ({confidence:.0%})", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                # Low confidence - show as unknown
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                cv2.putText(frame, "Unknown", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
        except Exception as e:
            continue
    
    # Display frame
    cv2.imshow(f"Attendance - Detected: {len(set(current_names))}", frame)
    
    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    
    current_time = time.time()
    
    if key == ord('o'):  # Save attendance
        if current_time - last_save_time < save_cooldown:
            print(f"⚠️ Please wait {save_cooldown} seconds between saves")
            continue
            
        if current_names:
            # Get unique names from current frame
            unique_names = list(set(current_names))
            
            # Prepare data for saving
            date_str = datetime.now().strftime("%d-%m-%Y")
            time_str = datetime.now().strftime("%H:%M:%S")
            csv_file = os.path.join(ATTENDANCE_FOLDER, f"Attendance_{date_str}.csv")
            
            # Check existing entries for today
            existing_entries = []
            if os.path.exists(csv_file):
                try:
                    with open(csv_file, 'r') as f:
                        reader = csv.reader(f)
                        next(reader, None)  # Skip header
                        for row in reader:
                            if row and len(row) >= 3:  # Check if row has all 3 columns
                                existing_entries.append(row[0])
                except:
                    pass
            
            # Filter out already marked persons
            new_entries = []
            for name in unique_names:
                if name not in existing_entries:
                    new_entries.append(name)
            
            # Save to CSV - EXACTLY 3 COLUMNS
            if new_entries:
                file_exists = os.path.exists(csv_file)
                
                with open(csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        # WRITE HEADER WITH 3 COLUMNS
                        writer.writerow(['NAME', 'TIME', 'DATE'])
                    
                    # WRITE DATA WITH 3 COLUMNS
                    for name in new_entries:
                        writer.writerow([name, time_str, date_str])  # 3 VALUES
                
                print(f"\n📝 ATTENDANCE SAVED ({time_str})")
                print(f"   File: {csv_file}")
                print(f"   New entries: {len(new_entries)}")
                for name in new_entries:
                    print(f"   ✓ {name}")
                print("-" * 30)
            else:
                print(f"ℹ️ All detected persons already marked today")
                
        else:
            print("⚠️ No faces detected for attendance")
        
        last_save_time = current_time
    
    elif key == ord('q'):  # Quit
        print("\n👋 Closing system...")
        break

# Cleanup
video.release()
cv2.destroyAllWindows()
print("✅ System closed")
   