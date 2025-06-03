import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model
from collections import deque

# Load CNN model
model = load_model("/Users/parth/Projects/ML/drowsiness/best_yawn_model.keras")

# Constants
MAR_THRESHOLD = 0.6
YAWN_FRAMES_THRESHOLD = 10
YAWN_COOLDOWN_FRAMES = 20
CNN_INPUT_SIZE = (96, 96)

# Use more complete set of mouth landmarks (outer + inner lips)
MOUTH_LANDMARKS = list(set([
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324,
    318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267,
    269, 270, 409, 415, 310, 311, 312, 13
]))

# Initialize variables
mar_buffer = deque(maxlen=5)
yawn_frame_count = 0
cooldown_counter = 0
yawning = False

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

def calculate_mar(landmarks, image_w, image_h):
    try:
        # Two vertical distances (outer + inner)
        top_outer = np.array([landmarks[13].x * image_w, landmarks[13].y * image_h])
        bottom_outer = np.array([landmarks[14].x * image_w, landmarks[14].y * image_h])
        top_inner = np.array([landmarks[312].x * image_w, landmarks[312].y * image_h])
        bottom_inner = np.array([landmarks[317].x * image_w, landmarks[317].y * image_h])

        # Two horizontal distances (outer + inner)
        left_outer = np.array([landmarks[78].x * image_w, landmarks[78].y * image_h])
        right_outer = np.array([landmarks[308].x * image_w, landmarks[308].y * image_h])
        left_inner = np.array([landmarks[82].x * image_w, landmarks[82].y * image_h])
        right_inner = np.array([landmarks[312].x * image_w, landmarks[312].y * image_h])

        vertical_dist = (np.linalg.norm(top_outer - bottom_outer) + np.linalg.norm(top_inner - bottom_inner)) / 2
        horizontal_dist = (np.linalg.norm(left_outer - right_outer) + np.linalg.norm(left_inner - right_inner)) / 2

        return vertical_dist / horizontal_dist
    except:
        return 0.0

def is_hand_covering_mouth(hand_landmarks, mouth_center, image_w, image_h):
    for hand in hand_landmarks:
        for lm in hand.landmark:
            x, y = int(lm.x * image_w), int(lm.y * image_h)
            if abs(x - mouth_center[0]) < 50 and abs(y - mouth_center[1]) < 50:
                return True
    return False

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_h, image_w, _ = frame.shape

    face_results = face_mesh.process(rgb_frame)
    hand_results = hands.process(rgb_frame)

    if face_results.multi_face_landmarks:
        landmarks = face_results.multi_face_landmarks[0].landmark

        # --- MAR Calculation ---
        mar = calculate_mar(landmarks, image_w, image_h)
        mar_buffer.append(mar)
        smoothed_mar = np.mean(mar_buffer)

        # --- Mouth Cropping for CNN ---
        mouth_coords = [landmarks[i] for i in MOUTH_LANDMARKS]
        mouth_points = np.array([[int(p.x * image_w), int(p.y * image_h)] for p in mouth_coords])
        x, y, w, h = cv2.boundingRect(mouth_points)

        # Add padding to get a larger, square ROI
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = w + 2 * padding
        h = h + 2 * padding

        mouth_img = frame[y:y + h, x:x + w]

        if mouth_img.size != 0:
            mouth_img_resized = cv2.resize(mouth_img, CNN_INPUT_SIZE)
            mouth_img_rgb = cv2.cvtColor(mouth_img_resized, cv2.COLOR_BGR2RGB)
            mouth_input = mouth_img_rgb.astype("float32") / 255.0
            mouth_input = np.expand_dims(mouth_input, axis=0)
            cnn_pred = model.predict(mouth_input, verbose=0)[0]
            cnn_yawning_prob = float(cnn_pred)  # Single output model (probability)
        else:
            cnn_yawning_prob = 0

        # --- Hand-over-Mouth Detection ---
        if hand_results.multi_hand_landmarks:
            hand_covering_mouth = is_hand_covering_mouth(hand_results.multi_hand_landmarks,
                                                         (x + w // 2, y + h // 2),
                                                         image_w, image_h)
        else:
            hand_covering_mouth = False

        # --- Final Decision Logic ---
        mar_based_yawning = smoothed_mar > MAR_THRESHOLD
        cnn_yawning = cnn_yawning_prob > 0.6

        if hand_covering_mouth:
            yawning = False
            yawn_frame_count = 0
        elif cnn_yawning:
            yawning = True
            cooldown_counter = YAWN_COOLDOWN_FRAMES
        elif mar_based_yawning and cooldown_counter == 0:
            yawn_frame_count += 1
            if yawn_frame_count >= YAWN_FRAMES_THRESHOLD:
                yawning = True
                cooldown_counter = YAWN_COOLDOWN_FRAMES
        else:
            yawn_frame_count = max(0, yawn_frame_count - 1)
            yawning = False

        if cooldown_counter > 0:
            cooldown_counter -= 1

        # --- Drawing on Frame ---
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

        if hand_covering_mouth:
            cv2.putText(frame, "Mouth Covered!", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        elif yawning:
            cv2.putText(frame, "Yawning Detected!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            cv2.putText(frame, "Not Yawning", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow("Hybrid Yawning Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
