import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque

# Load the trained model (input shape: 128x128x1)
model = tf.keras.models.load_model("/Users/parth/Projects/ML/drowsiness/best_eye_state_model.h5")


# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.8, min_tracking_confidence=0.8)

# Define eye landmark indices
LEFT_EYE_INDICES = [33, 133, 160, 159, 158, 153, 144, 145, 246]
RIGHT_EYE_INDICES = [362, 263, 385, 386, 387, 380, 373, 374, 466]

# EMA Buffer
eye_state_buffer = deque(maxlen=12)
alpha = 0.4  # EMA weight

# Webcam
cap = cv2.VideoCapture(0)

def preprocess_eye(eye):
    if eye.size == 0:
        return None

    eye_gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    eye_gray = clahe.apply(eye_gray)

    eye_gray = cv2.bilateralFilter(eye_gray, 7, 75, 75)

    eye_resized = cv2.resize(eye_gray, (128, 128))  # ✅ Match model input size
    eye_normalized = eye_resized / 255.0
    eye_input = np.expand_dims(eye_normalized, axis=(0, -1))  # (1, 128, 128, 1)

    return eye_input

def auto_adjust_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    enhanced = cv2.merge((l, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

def get_eye_bbox(face_landmarks, indices, img_width, img_height, scale=1.4):
    x_min, y_min = img_width, img_height
    x_max, y_max = 0, 0
    for i in indices:
        x = int(face_landmarks.landmark[i].x * img_width)
        y = int(face_landmarks.landmark[i].y * img_height)
        x_min, y_min = min(x_min, x), min(y_min, y)
        x_max, y_max = max(x_max, x), max(y_max, y)

    eye_width = x_max - x_min
    eye_height = y_max - y_min
    x_min = max(0, x_min - int(eye_width * (scale - 1) / 2))
    x_max = min(img_width, x_max + int(eye_width * (scale - 1) / 2))
    y_min = max(0, y_min - int(eye_height * (scale - 1) / 2))
    y_max = min(img_height, y_max + int(eye_height * (scale - 1) / 2))

    return x_min, y_min, x_max, y_max

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = auto_adjust_contrast(frame)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            lx1, ly1, lx2, ly2 = get_eye_bbox(face_landmarks, LEFT_EYE_INDICES, w, h)
            rx1, ry1, rx2, ry2 = get_eye_bbox(face_landmarks, RIGHT_EYE_INDICES, w, h)

            left_eye_img = preprocess_eye(frame[ly1:ly2, lx1:lx2])
            right_eye_img = preprocess_eye(frame[ry1:ry2, rx1:rx2])

            if left_eye_img is not None and right_eye_img is not None:
                if left_eye_img.shape != (1, 128, 128, 1):
                    print("⚠️ Left eye shape mismatch:", left_eye_img.shape)
                    continue
                if right_eye_img.shape != (1, 128, 128, 1):
                    print("⚠️ Right eye shape mismatch:", right_eye_img.shape)
                    continue

                left_pred = model.predict(left_eye_img, verbose=0)[0][0]
                right_pred = model.predict(right_eye_img, verbose=0)[0][0]

                if eye_state_buffer:
                    prev_left, prev_right = eye_state_buffer[-1]
                    left_pred = alpha * left_pred + (1 - alpha) * prev_left
                    right_pred = alpha * right_pred + (1 - alpha) * prev_right

                eye_state_buffer.append((left_pred, right_pred))

                left_final = "Open" if left_pred > 0.7 else "Closed"
                right_final = "Open" if right_pred > 0.7 else "Closed"

                color_left = (0, 255, 0) if left_final == "Open" else (0, 0, 255)
                color_right = (0, 255, 0) if right_final == "Open" else (0, 0, 255)

                # Display Left Eye
                cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), color_left, 2)
                cv2.putText(frame, f"Left: {left_final}", (lx1, ly1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_left, 2)

                # Display Right Eye
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), color_right, 2)
                cv2.putText(frame, f"Right: {right_final}", (rx1, ry1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_right, 2)

    cv2.imshow("Real-time Eye State Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()