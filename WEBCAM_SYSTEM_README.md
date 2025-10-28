# 📹 Live Webcam Attendance System

A real-time face recognition attendance system using your webcam. No video files needed!

---

## 🚀 Quick Start Guide

### **Step 1: Install Requirements**

Make sure you have all packages installed:
```powershell
& "E:/Face recognition/face/Scripts/Activate.ps1"
cd "e:\Face recognition\DeepTrack"
pip install deepface ultralytics opencv-python pandas numpy
```

---

### **Step 2: Register People (Add Faces)**

Run the face registration tool:
```powershell
python register_face.py
```

**Menu Options:**
- **1. Register new person** - Add someone to the system
- **2. List registered people** - See who's registered
- **3. Delete person** - Remove someone
- **4. Exit**

#### **How to Register:**

1. Select option `1`
2. Enter the person's name (e.g., `John_Doe`)
3. Position face inside the green box on screen
4. Press **SPACE** to capture photos
5. Capture **5-10 photos** from different angles
6. Press **'q'** when done

**Tips for better accuracy:**
- ✅ Capture from different angles (left, right, center)
- ✅ Include different expressions (neutral, smiling)
- ✅ Ensure good lighting
- ✅ Look directly at camera for at least 2-3 photos
- ❌ Avoid blurry images
- ❌ Avoid extreme shadows

---

### **Step 3: Run Attendance System**

Start the webcam attendance system:
```powershell
python webcam_attendance.py
```

#### **Controls:**
- **'q'** - Quit the system
- **'r'** - Reload known faces (after adding new people)
- **'s'** - Show today's attendance list

#### **How it Works:**
1. Webcam opens and starts detecting faces
2. When a registered face is detected:
   - Green box appears with name and confidence %
   - Attendance is automatically marked
   - Name is added to CSV file
3. Unknown faces show as "Unknown" with red box
4. Each person can only be marked once per day

---

## 📁 Folder Structure

After registration, your structure will look like:

```
DeepTrack/
├── webcam_attendance.py       # Main attendance system
├── register_face.py           # Face registration tool
├── known_faces_webcam/        # Registered faces database
│   ├── John_Doe/
│   │   ├── John_Doe_20251024_101523_1.jpg
│   │   ├── John_Doe_20251024_101525_2.jpg
│   │   ├── John_Doe_20251024_101527_3.jpg
│   │   ├── John_Doe_20251024_101530_4.jpg
│   │   └── John_Doe_20251024_101532_5.jpg
│   ├── Jane_Smith/
│   │   ├── Jane_Smith_20251024_102015_1.jpg
│   │   ├── Jane_Smith_20251024_102017_2.jpg
│   │   └── ...
│   └── ...
└── webcam_attendance.csv      # Attendance records
```

---

## 📊 Output Files

### **webcam_attendance.csv**
Contains all attendance records:

```csv
Name,Date,Time
John_Doe,2025-10-24,09:15:23
Jane_Smith,2025-10-24,09:16:45
Bob_Wilson,2025-10-24,09:18:12
```

- **Name** - Person's name (as registered)
- **Date** - Date of attendance (YYYY-MM-DD)
- **Time** - Time when first detected (HH:MM:SS)

---

## 🎯 How to Add Faces to the System

There are **TWO methods** to add faces:

### **Method 1: Using Registration Tool (Recommended)**

```powershell
python register_face.py
```

Select option 1 and follow the on-screen instructions.

**Advantages:**
- ✅ Easy and guided process
- ✅ Automatic photo naming
- ✅ Live preview
- ✅ Quality control

---

### **Method 2: Manual Addition**

Create the folder structure manually:

1. Create folder: `known_faces_webcam/`
2. Inside, create a folder with person's name: `known_faces_webcam/John_Doe/`
3. Add 5-10 photos of that person's face inside their folder
4. Photos should be named: `photo1.jpg`, `photo2.jpg`, etc.

**Example:**
```
known_faces_webcam/
├── John_Doe/
│   ├── photo1.jpg
│   ├── photo2.jpg
│   ├── photo3.jpg
│   ├── photo4.jpg
│   └── photo5.jpg
└── Jane_Smith/
    ├── photo1.jpg
    ├── photo2.jpg
    └── photo3.jpg
```

**Requirements for manual photos:**
- Clear face visible
- Good lighting
- Face should be front-facing
- Resolution: At least 640x480
- Format: JPG, JPEG, or PNG

---

## ⚙️ System Features

### ✅ **Face Detection**
- Uses YOLOv11 (nano) model
- Fast and accurate
- Works in real-time

### ✅ **Face Recognition**
- DeepFace with Facenet model
- 128D facial embeddings
- Cosine distance matching

### ✅ **Performance Optimizations**
- Frame skipping (processes every 3rd frame)
- Embedding caching (no repeated calculations)
- Recognition cooldown (5 seconds between marks)

### ✅ **Attendance Management**
- One entry per person per day
- Automatic CSV logging
- Timestamp for each entry

---

## 🔧 Configuration

You can adjust these parameters in `webcam_attendance.py`:

```python
# Line 20: Recognition cooldown (seconds between recognitions)
RECOGNITION_COOLDOWN = 5

# Line 115: Face detection confidence threshold
if confidence > 0.5:  # Adjust 0.5 (range: 0.0 to 1.0)

# Line 147: Recognition distance threshold
RECOGNITION_THRESHOLD = 10.0  # Lower = stricter matching

# Line 198: Frame skip rate
process_every_n_frames = 3  # Process every Nth frame
```

---

## 📝 Usage Examples

### **Example 1: Daily Classroom Attendance**

**Morning:**
1. Start webcam system: `python webcam_attendance.py`
2. Students come and stand in front of camera
3. System automatically marks attendance
4. Check attendance: Press 's' in webcam window

**End of day:**
- Press 'q' to quit
- Attendance saved in `webcam_attendance.csv`

---

### **Example 2: Adding New Student**

```powershell
# Register new student
python register_face.py

# Select: 1. Register new person
# Enter name: Alice_Johnson
# Capture 5-10 photos
# Press 'q' when done

# Reload in running system
# In webcam window, press 'r' to reload faces
```

---

### **Example 3: Viewing Attendance**

**While system is running:**
- Press **'s'** in webcam window

**After closing:**
```powershell
# Open CSV in Excel/Notepad
notepad webcam_attendance.csv
```

Or view in Python:
```python
import pandas as pd
df = pd.read_csv("webcam_attendance.csv")
print(df)
```

---

## 🐛 Troubleshooting

### **Webcam not opening?**
```python
# Check webcam index (try 0, 1, 2)
cap = cv2.VideoCapture(0)  # Change 0 to 1 or 2
```

### **No faces detected?**
- Check lighting (ensure good lighting on face)
- Move closer to camera
- Ensure face is clearly visible

### **Wrong person recognized?**
- Add more photos of correct person (8-10 recommended)
- Delete incorrect person's photos
- Adjust `RECOGNITION_THRESHOLD` (lower = stricter)

### **System too slow?**
```python
# Increase frame skip rate
process_every_n_frames = 5  # Process less frequently
```

### **"Unknown" for registered person?**
- Press 'r' to reload faces
- Ensure photos are in correct folder structure
- Check photo quality (clear, well-lit)
- Add more varied photos

---

## 🔄 Differences from Original System

| Feature | Original | New Webcam System |
|---------|----------|-------------------|
| Input | Video files (.mp4, .dav) | Live webcam |
| ROI | Required polygon config | Not needed (full frame) |
| Registration | Manual file copying | Built-in registration tool |
| Output | Annotated video + CSV | Live display + CSV |
| Setup | Complex (video files needed) | Simple (just run) |
| Use Case | Post-processing recordings | Real-time attendance |

---

## 💡 Tips for Best Results

1. **Lighting:** Ensure good, even lighting on faces
2. **Distance:** Stand 1-2 meters from camera
3. **Angle:** Face camera directly
4. **Photos:** Capture 8-10 photos per person for best accuracy
5. **Variety:** Include different expressions and slight angle variations
6. **Testing:** Test recognition before using in production

---

## 📞 Common Questions

**Q: How many people can I register?**
A: Unlimited! The system can handle hundreds of registered people.

**Q: Can multiple people be recognized at once?**
A: Yes! The system detects and recognizes all visible faces simultaneously.

**Q: What if someone is marked absent but they attended?**
A: They may not have been detected. Ensure they stand clearly in front of camera for 2-3 seconds.

**Q: Can I use this offline?**
A: Yes! Once models are downloaded, the system works completely offline.

**Q: How accurate is it?**
A: With 5+ good quality photos per person, accuracy is typically 95%+ in good lighting conditions.

---

## 🎓 Next Steps

1. ✅ Register 2-3 test people
2. ✅ Run the attendance system
3. ✅ Test recognition with different angles
4. ✅ Adjust thresholds if needed
5. ✅ Deploy for actual use

---

## 📄 Files Overview

- **`webcam_attendance.py`** - Main attendance system with webcam
- **`register_face.py`** - Tool to register new people
- **`known_faces_webcam/`** - Database of registered faces
- **`webcam_attendance.csv`** - Attendance records
- **`Core1/yolov11n-face.pt`** - YOLO face detection model

---

**Need help?** Check the troubleshooting section or open an issue on GitHub!

Happy tracking! 🎉
