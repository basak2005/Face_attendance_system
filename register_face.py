import cv2
import os
import sys
import numpy as np
from datetime import datetime

# Directory for storing known faces
KNOWN_FACES_DIR = "known_faces_webcam"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

def register_new_person():
    """Register a new person by capturing photos from webcam"""
    
    print("\n" + "="*60)
    print("FACE REGISTRATION SYSTEM")
    print("="*60)
    
    # Get person's name
    person_name = input("\nEnter person's name (e.g., John_Doe): ").strip()
    
    if not person_name:
        print("Error: Name cannot be empty!")
        return
    
    # Create person's directory
    person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
    
    if os.path.exists(person_dir):
        print(f"\n⚠️  Person '{person_name}' already exists!")
        choice = input("Do you want to add more photos? (y/n): ").lower()
        if choice != 'y':
            return
    else:
        os.makedirs(person_dir)
    
    # Count existing photos
    existing_photos = len([f for f in os.listdir(person_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(person_dir) else 0
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam!")
        return
    
    # Set webcam properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("\n" + "="*60)
    print("INSTRUCTIONS:")
    print("="*60)
    print("1. Position your face in front of the camera")
    print("2. Press SPACE to capture a photo (capture 5-10 photos)")
    print("3. Try different angles and expressions")
    print("4. Press 'q' when done")
    print("="*60 + "\n")
    
    photo_count = existing_photos
    target_photos = 5
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam")
            break
        
        # Create display frame
        display_frame = frame.copy()
        
        # Draw guide rectangle
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        rect_size = 400
        
        cv2.rectangle(
            display_frame,
            (center_x - rect_size // 2, center_y - rect_size // 2),
            (center_x + rect_size // 2, center_y + rect_size // 2),
            (0, 255, 0),
            2
        )
        
        # Add instructions on frame
        cv2.putText(
            display_frame,
            f"Registering: {person_name}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        
        cv2.putText(
            display_frame,
            f"Photos captured: {photo_count}/{target_photos}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        cv2.putText(
            display_frame,
            "Position face inside green box",
            (10, height - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        
        cv2.putText(
            display_frame,
            "Press SPACE to capture | Press 'q' to finish",
            (10, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        
        # Show frame
        cv2.imshow('Face Registration', display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Space bar to capture
            photo_count += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{person_name}_{timestamp}_{photo_count}.jpg"
            filepath = os.path.join(person_dir, filename)
            
            cv2.imwrite(filepath, frame)
            print(f"✓ Photo {photo_count} saved: {filename}")
            
            # Flash effect
            white_frame = 255 * np.ones_like(frame)
            cv2.imshow('Face Registration', white_frame)
            cv2.waitKey(100)
            
        elif key == ord('q'):  # Quit
            if photo_count >= target_photos:
                print(f"\n✓ Registration complete for {person_name}!")
                print(f"  Total photos: {photo_count}")
                print(f"  Saved to: {person_dir}")
            else:
                print(f"\n⚠️  Only {photo_count} photos captured.")
                print(f"  Recommended: At least {target_photos} photos for better accuracy.")
                choice = input("Continue anyway? (y/n): ").lower()
                if choice == 'y':
                    print(f"✓ Registration saved with {photo_count} photos")
                else:
                    print("Registration cancelled")
                    # Optionally remove the directory if no photos
                    if photo_count == 0 and os.path.exists(person_dir):
                        os.rmdir(person_dir)
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("Next steps:")
    print("1. Run 'python webcam_attendance.py' to start attendance system")
    print("2. Press 'r' in the webcam window to reload faces")
    print("="*60 + "\n")

def list_registered_people():
    """List all registered people"""
    print("\n" + "="*60)
    print("REGISTERED PEOPLE")
    print("="*60)
    
    if not os.path.exists(KNOWN_FACES_DIR):
        print("No registrations found.")
        return
    
    people = [d for d in os.listdir(KNOWN_FACES_DIR) if os.path.isdir(os.path.join(KNOWN_FACES_DIR, d))]
    
    if not people:
        print("No people registered yet.")
    else:
        for i, person in enumerate(sorted(people), 1):
            person_dir = os.path.join(KNOWN_FACES_DIR, person)
            photo_count = len([f for f in os.listdir(person_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
            print(f"{i}. {person} ({photo_count} photos)")
    
    print("="*60 + "\n")

def delete_person():
    """Delete a registered person"""
    list_registered_people()
    
    person_name = input("Enter name to delete (or press Enter to cancel): ").strip()
    
    if not person_name:
        print("Cancelled.")
        return
    
    person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
    
    if not os.path.exists(person_dir):
        print(f"Error: '{person_name}' not found!")
        return
    
    confirm = input(f"Are you sure you want to delete '{person_name}'? (y/n): ").lower()
    
    if confirm == 'y':
        import shutil
        shutil.rmtree(person_dir)
        print(f"✓ Deleted '{person_name}'")
    else:
        print("Cancelled.")

def main():
    """Main menu"""
    
    while True:
        print("\n" + "="*60)
        print("FACE REGISTRATION MENU")
        print("="*60)
        print("1. Register new person")
        print("2. List registered people")
        print("3. Delete person")
        print("4. Exit")
        print("="*60)
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            register_new_person()
        elif choice == '2':
            list_registered_people()
        elif choice == '3':
            delete_person()
        elif choice == '4':
            print("\nGoodbye!")
            break
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()
