"""
Test script to verify webcam and system components
"""
import cv2
import sys

print("="*60)
print("SYSTEM TEST")
print("="*60)

# Test 1: Check imports
print("\n1. Testing imports...")
try:
    import numpy as np
    print("   ✓ numpy")
    import pandas as pd
    print("   ✓ pandas")
    from ultralytics import YOLO  # type: ignore
    print("   ✓ ultralytics (YOLO)")
    from deepface import DeepFace
    print("   ✓ deepface")
    import cv2
    print("   ✓ opencv-python")
    print("   All packages installed correctly!")
except ImportError as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 2: Check YOLO model
print("\n2. Testing YOLO model...")
try:
    model = YOLO("yolov11n-face.pt")
    print("   ✓ YOLO model loaded successfully!")
except Exception as e:
    print(f"   ✗ Error loading YOLO model: {e}")
    sys.exit(1)

# Test 3: Check webcam
print("\n3. Testing webcam access...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("   ✗ Error: Could not open webcam!")
    print("   Check if webcam is connected and not used by another app.")
    sys.exit(1)

# Try to read a frame
ret, frame = cap.read()
if not ret:
    print("   ✗ Error: Could not read frame from webcam!")
    cap.release()
    sys.exit(1)

print(f"   ✓ Webcam opened successfully!")
print(f"   Resolution: {frame.shape[1]}x{frame.shape[0]}")

# Test 4: Test face detection
print("\n4. Testing face detection...")
print("   Position your face in front of webcam for 3 seconds...")

import time
face_detected = False
start_time = time.time()

while time.time() - start_time < 3:
    ret, frame = cap.read()
    if ret:
        results = model(frame, verbose=False)
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                face_detected = True
                break
        if face_detected:
            break

if face_detected:
    print("   ✓ Face detection working!")
else:
    print("   ⚠ No face detected (might be OK, check lighting)")

cap.release()

# Test 5: Check folders
print("\n5. Checking folder structure...")
import os

if os.path.exists("known_faces_webcam"):
    people = [d for d in os.listdir("known_faces_webcam") 
              if os.path.isdir(os.path.join("known_faces_webcam", d))]
    print(f"   ✓ Known faces folder exists")
    print(f"   Registered people: {len(people)}")
    if people:
        for person in people[:3]:  # Show first 3
            photo_count = len([f for f in os.listdir(os.path.join("known_faces_webcam", person)) 
                             if f.endswith(('.jpg', '.jpeg', '.png'))])
            print(f"     - {person} ({photo_count} photos)")
        if len(people) > 3:
            print(f"     ... and {len(people)-3} more")
else:
    print("   ⚠ No known faces registered yet")
    print("   Run 'python register_face.py' to add people")

print("\n" + "="*60)
print("TEST COMPLETE!")
print("="*60)
print("\n✓ System is ready to use!")
print("\nNext steps:")
print("  1. Register faces: python register_face.py")
print("  2. Start attendance: python webcam_attendance.py")
print("  OR")
print("  Run: quick_start.bat")
print("="*60)
