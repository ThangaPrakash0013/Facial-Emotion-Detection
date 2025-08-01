import cv2
from deepface import DeepFace
from PIL import Image
import numpy as np
import os

# Open built-in camera
cap = cv2.VideoCapture(0)

# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Map emotions to emoji image paths (relative paths)
emotion_emojis = {
    'angry': 'myenv\\angry.png',
    'disgust': 'myenv\\disgust.png',
    'fear': 'myenv\\fear.png',
    'happy': 'myenv\\happy.png',
    'sad': 'myenv\\sad.png',
    'surprise': 'myenv\\surprise.png',
    'neutral': 'myenv\\neutral.png'
}

# Load emoji images
emoji_images = {}
for emotion, path in emotion_emojis.items():
    if os.path.exists(path):
        emoji_images[emotion] = Image.open(path)
    else:
        print(f"Warning: Emoji image not found for {emotion} at {path}")

# Emotion scores averaging
emotion_scores = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}
num_frames = 3  # Reduced number of frames for faster response
frame_count = 0

# Variables to stabilize emotion display
stable_emotion = None
stable_emoji = None
stable_counter = 0
stable_threshold = 2  # Reduced threshold for faster updates

# Skip frames for faster processing
skip_frames = 1  # Analyze emotions every 2nd frame

while True:
    # Capture video frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Skip frames for faster processing
    if frame_count % (skip_frames + 1) != 0:
        frame_count += 1
        continue
    
    # Convert frame to grayscale for better detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(50, 50))

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Extract face ROI (Region of Interest)
        face_roi = frame[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (224, 224))  # Resize for better model performance

        try:
            # Analyze emotion using DeepFace
            analysis = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            
            # Accumulate emotion scores
            for emotion in emotion_scores:
                emotion_scores[emotion] += analysis[0]['emotion'][emotion]
            
            frame_count += 1

            # Average scores every `num_frames` frames
            if frame_count >= num_frames:
                for emotion in emotion_scores:
                    emotion_scores[emotion] /= num_frames
                
                # Get dominant emotion
                dominant_emotion = max(emotion_scores, key=emotion_scores.get)
                
                # Check if the dominant emotion is consistent
                if dominant_emotion == stable_emotion:
                    stable_counter += 1
                else:
                    stable_emotion = dominant_emotion
                    stable_counter = 0
                
                # Update display only if the emotion is stable for `stable_threshold` frames
                if stable_counter >= stable_threshold:
                    # Get corresponding emoji image
                    emoji_image = emoji_images.get(dominant_emotion)
                    
                    if emoji_image:
                        # Resize emoji image using LANCZOS resampling
                        emoji_image = emoji_image.resize((50, 50), Image.LANCZOS)
                        
                        # Convert PIL image to OpenCV format
                        emoji_cv2 = cv2.cvtColor(np.array(emoji_image), cv2.COLOR_RGB2BGR)
                        
                        # Overlay emoji next to the emotion name
                        frame[y-60:y-10, x+100:x+150] = emoji_cv2
                    
                    # Display emotion name
                    emotion_text = f"{dominant_emotion}"
                    cv2.putText(frame, emotion_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Reset counters
                emotion_scores = {e: 0 for e in emotion_scores}
                frame_count = 0
        
        except Exception as e:
            print("Error:", e)

    # Show the video frame with detected emotions
    cv2.imshow("Facial Emotion Detection System", frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()________________________________________
