import cv2
import numpy as np
import os
import pandas as pd
import pickle
from ultralytics import YOLO  # type: ignore
from deepface import DeepFace
from datetime import datetime
import time
from typing import Dict, Any, List

# Initialize YOLO model for face detection
model = YOLO("yolov11n-face.pt")

# Directory for storing known faces
KNOWN_FACES_DIR = "known_faces_webcam"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# Cache file for face embeddings
EMBEDDINGS_CACHE_FILE = "face_embeddings_cache.pkl"

# Attendance file
attendance_file = "webcam_attendance.csv"
attendance_today = set()
today_date = datetime.now().strftime("%Y-%m-%d")

# Load today's attendance
if os.path.exists(attendance_file):
    try:
        df = pd.read_csv(attendance_file)
        if "Date" in df.columns and "Name" in df.columns:
            today_df = df[df["Date"] == today_date]
            attendance_today = set(today_df["Name"].values)
    except Exception as e:
        print(f"Error reading attendance file: {e}")

# Face embeddings cache
face_embeddings_cache = {}
last_recognition_time = {}
RECOGNITION_COOLDOWN = 5  # seconds between recognitions for same person

def load_known_faces():
    """Load and cache embeddings for all known faces"""
    print("Loading known faces...")
    global face_embeddings_cache
    
    # Check if cache file exists
    if os.path.exists(EMBEDDINGS_CACHE_FILE):
        try:
            with open(EMBEDDINGS_CACHE_FILE, "rb") as f:
                face_embeddings_cache = pickle.load(f)
            print(f"✓ Loaded embeddings from cache ({len(face_embeddings_cache)} people)")
            print("  Names loaded:", ", ".join(sorted(face_embeddings_cache.keys())))
            return face_embeddings_cache
        except Exception as e:
            print(f"⚠️  Error loading cache file: {e}")
            print("  Will rebuild cache from images...")
    
    # If cache doesn't exist, compute embeddings
    face_embeddings_cache = {}
    
    if not os.path.exists(KNOWN_FACES_DIR):
        print(f"No known faces directory found. Create folder: {KNOWN_FACES_DIR}")
        return
    
    print("Building face embeddings cache (this may take a while)...")
    for person_name in os.listdir(KNOWN_FACES_DIR):
        person_path = os.path.join(KNOWN_FACES_DIR, person_name)
        if os.path.isdir(person_path):
            embeddings = []
            for img_name in os.listdir(person_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(person_path, img_name)
                    try:
                        # Extract embedding
                        embedding_objs = DeepFace.represent(
                            img_path=img_path,
                            model_name="Facenet",
                            enforce_detection=False
                        )
                        if embedding_objs and isinstance(embedding_objs, list):
                            embedding_data: Dict[str, Any] = embedding_objs[0]  # type: ignore
                            embeddings.append(embedding_data["embedding"])
                            print(f"  ✓ Processed: {person_name}/{img_name}")
                    except Exception as e:
                        print(f"  ✗ Error loading {img_path}: {e}")
            
            if embeddings:
                # Store average embedding for each person
                face_embeddings_cache[person_name] = np.mean(embeddings, axis=0)
                print(f"  ✓ Created embedding for {person_name} (from {len(embeddings)} images)")
    
    # Save embeddings to cache
    if face_embeddings_cache:
        try:
            with open(EMBEDDINGS_CACHE_FILE, "wb") as f:
                pickle.dump(face_embeddings_cache, f)
            print(f"\n✓ Saved embeddings to cache file: {EMBEDDINGS_CACHE_FILE}")
        except Exception as e:
            print(f"⚠️  Error saving cache file: {e}")
    
    print(f"\n✓ Loaded {len(face_embeddings_cache)} people")
    return face_embeddings_cache

def update_face_cache(person_name, img_paths):
    """Update the cache with new face embeddings for a specific person"""
    global face_embeddings_cache
    embeddings = []
    
    print(f"Updating cache for {person_name}...")
    for img_path in img_paths:
        try:
            embedding_objs = DeepFace.represent(
                img_path=img_path,
                model_name="Facenet",
                enforce_detection=False
            )
            if embedding_objs and isinstance(embedding_objs, list):
                embedding_data: Dict[str, Any] = embedding_objs[0]  # type: ignore
                embeddings.append(embedding_data["embedding"])
                print(f"  ✓ Processed: {img_path}")
        except Exception as e:
            print(f"  ✗ Error processing {img_path}: {e}")
    
    if embeddings:
        # Update cache with average embedding
        face_embeddings_cache[person_name] = np.mean(embeddings, axis=0)
        try:
            with open(EMBEDDINGS_CACHE_FILE, "wb") as f:
                pickle.dump(face_embeddings_cache, f)
            print(f"✓ Updated cache for {person_name} with {len(embeddings)} images")
            return True
        except Exception as e:
            print(f"⚠️  Error updating cache file: {e}")
            return False
    else:
        print(f"⚠️  No valid embeddings found for {person_name}")
        return False

def mark_attendance(name):
    """Mark attendance for a person"""
    global attendance_today
    current_time = datetime.now().strftime("%H:%M:%S")
    
    if name not in attendance_today:
        attendance_today.add(name)
        new_entry = pd.DataFrame({
            "Name": [name], 
            "Date": [today_date], 
            "Time": [current_time]
        })
        
        write_header = not os.path.exists(attendance_file) or os.path.getsize(attendance_file) == 0
        new_entry.to_csv(attendance_file, mode="a", header=write_header, index=False)
        print(f"✓ Attendance marked for {name} at {current_time}")
        return True
    return False

def recognize_face(face_img):
    """Recognize face by comparing with known faces"""
    if len(face_embeddings_cache) == 0:
        return "Unknown", 0.0
    
    try:
        # Extract embedding from detected face
        embedding_objs = DeepFace.represent(
            img_path=face_img,
            model_name="Facenet",
            enforce_detection=False
        )
        
        if not embedding_objs:
            return "Unknown", 0.0
        
        embedding_data: Dict[str, Any] = embedding_objs[0]  # type: ignore
        test_embedding = np.array(embedding_data["embedding"])
        
        # Find best match
        best_match = "Unknown"
        best_distance = float('inf')
        
        for name, known_embedding in face_embeddings_cache.items():
            # Calculate cosine distance
            distance = np.linalg.norm(test_embedding - known_embedding)
            if distance < best_distance:
                best_distance = distance
                best_match = name
        
        # Threshold for recognition (adjust as needed)
        RECOGNITION_THRESHOLD = 10.0
        
        if best_distance < RECOGNITION_THRESHOLD:
            confidence = max(0, 100 - (best_distance * 10))
            return best_match, confidence
        else:
            return "Unknown", 0.0
            
    except Exception as e:
        print(f"Recognition error: {e}")
        return "Unknown", 0.0

def main():
    """Main function to run webcam attendance system"""
    
    # Load known faces
    load_known_faces()
    
    if len(face_embeddings_cache) == 0:
        print("\n⚠️  WARNING: No known faces found!")
        print(f"Please add face images to: {KNOWN_FACES_DIR}")
        print("Create folders with person names and add their photos inside.")
        print("\nExample structure:")
        print(f"  {KNOWN_FACES_DIR}/")
        print("    ├── John_Doe/")
        print("    │   ├── photo1.jpg")
        print("    │   └── photo2.jpg")
        print("    └── Jane_Smith/")
        print("        ├── photo1.jpg")
        print("        └── photo2.jpg")
        print("\nPress 'q' to quit or continue to test face detection only...")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam!")
        return
    
    # Set webcam properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("\n" + "="*60)
    print("WEBCAM ATTENDANCE SYSTEM STARTED")
    print("="*60)
    print("\nControls:")
    print("  'q' - Quit")
    print("  'r' - Reload known faces")
    print("  's' - Show attendance list")
    print("\n" + "="*60 + "\n")
    
    frame_count = 0
    process_every_n_frames = 3  # Process every 3rd frame for performance
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam")
            break
        
        frame_count += 1
        display_frame = frame.copy()
        
        # Process every Nth frame for face detection
        if frame_count % process_every_n_frames == 0:
            # Detect faces using YOLO
            results = model(frame, verbose=False)
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])
                        
                        if confidence > 0.5:  # Confidence threshold
                            # Extract face region
                            face_img = frame[y1:y2, x1:x2]
                            
                            if face_img.size > 0:
                                # Recognize face
                                name, recog_confidence = recognize_face(face_img)
                                
                                # Check cooldown for recognition
                                current_time = time.time()
                                can_recognize = True
                                
                                if name in last_recognition_time:
                                    if current_time - last_recognition_time[name] < RECOGNITION_COOLDOWN:
                                        can_recognize = False
                                
                                # Mark attendance if recognized and not recently marked
                                if name != "Unknown" and can_recognize:
                                    mark_attendance(name)
                                    last_recognition_time[name] = current_time
                                
                                # Draw bounding box
                                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                                
                                # Draw label
                                label = f"{name} ({recog_confidence:.1f}%)" if name != "Unknown" else "Unknown"
                                
                                # Add background for text
                                (text_width, text_height), _ = cv2.getTextSize(
                                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                                )
                                cv2.rectangle(
                                    display_frame, 
                                    (x1, y1 - text_height - 10), 
                                    (x1 + text_width, y1), 
                                    color, 
                                    -1
                                )
                                cv2.putText(
                                    display_frame, 
                                    label, 
                                    (x1, y1 - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.6, 
                                    (255, 255, 255), 
                                    2
                                )
        
        # Display attendance count
        info_text = f"Attendance Today: {len(attendance_today)} | Known Faces: {len(face_embeddings_cache)}"
        cv2.putText(
            display_frame, 
            info_text, 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 0), 
            2
        )
        
        # Show frame
        cv2.imshow('Webcam Attendance System', display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\nExiting...")
            break
        elif key == ord('r'):
            print("\nReloading known faces...")
            load_known_faces()
        elif key == ord('s'):
            print("\n" + "="*60)
            print("TODAY'S ATTENDANCE")
            print("="*60)
            if attendance_today:
                for i, name in enumerate(sorted(attendance_today), 1):
                    print(f"{i}. {name}")
            else:
                print("No attendance marked yet.")
            print("="*60 + "\n")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Final attendance summary
    print("\n" + "="*60)
    print("SESSION SUMMARY")
    print("="*60)
    print(f"Total attendance marked: {len(attendance_today)}")
    if attendance_today:
        print("\nAttendees:")
        for i, name in enumerate(sorted(attendance_today), 1):
            print(f"  {i}. {name}")
    print(f"\nAttendance saved to: {attendance_file}")
    print("="*60)

if __name__ == "__main__":
    main()
