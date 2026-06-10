import cv2
import pickle
import numpy as np
import time
import urllib.request
import os
import mediapipe as mp

# ── Load model ──────────────────────────────────────────────
with open('./model/sign_model.pkl', 'rb') as f:
    model = pickle.load(f)

# ── MediaPipe setup ─────────────────────────────────────────
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = './hand_landmarker.task'
if not os.path.exists(model_path):
    print('Downloading hand landmark model...')
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task',
        model_path
    )

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

# ── State ────────────────────────────────────────────────────
current_word = []
last_letter = ''
hold_start = None
cooldown_until = 0
HOLD_DURATION = 1.5
CONFIDENCE_THRESHOLD = 0.55

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print('\nSign Language Translator Running!')
print('Controls: SPACE = save word | BACKSPACE = delete letter | Q = quit\n')

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = landmarker.detect(mp_image)

        predicted_letter = ''
        confidence = 0.0
        now = time.time()
        in_cooldown = now < cooldown_until

        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]

            # Draw hand connections
            connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
            landmark_coords = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

            for connection in connections:
                start = landmark_coords[connection.start]
                end = landmark_coords[connection.end]
                cv2.line(frame, start, end, (0, 200, 100), 2)
            for coord in landmark_coords:
                cv2.circle(frame, coord, 5, (255, 255, 255), -1)

            # Extract features
            x_coords = [lm.x for lm in hand_landmarks]
            y_coords = [lm.y for lm in hand_landmarks]
            x_min, y_min = min(x_coords), min(y_coords)

            features = []
            for lm in hand_landmarks:
                features.append(lm.x - x_min)
                features.append(lm.y - y_min)

            proba = model.predict_proba([features])[0]
            confidence = max(proba)
            predicted_letter = model.classes_[np.argmax(proba)]

            # ── Hold-to-confirm logic ─────────────────────────
            if not in_cooldown:
                if predicted_letter == last_letter:
                    if hold_start is None:
                        hold_start = now
                    hold_elapsed = now - hold_start
                    progress = min(hold_elapsed / HOLD_DURATION, 1.0)

                    # Progress bar
                    bar_x, bar_y = w // 2 - 120, 20
                    cv2.rectangle(frame, (bar_x, bar_y),
                                  (bar_x + 240, bar_y + 12), (60, 60, 60), -1)
                    cv2.rectangle(frame, (bar_x, bar_y),
                                  (bar_x + int(240 * progress), bar_y + 12),
                                  (50, 220, 120), -1)

                    if hold_elapsed >= HOLD_DURATION and confidence > CONFIDENCE_THRESHOLD:
                        current_word.append(predicted_letter)
                        print(f'Letter confirmed: {predicted_letter}')
                        # Reset everything cleanly
                        last_letter = ''
                        hold_start = None
                        cooldown_until = now + 1.0  # 1 second cooldown before next letter

                else:
                    # New letter detected — reset hold
                    last_letter = predicted_letter
                    hold_start = None
            else:
                # Show cooldown bar in blue
                remaining = cooldown_until - now
                progress = remaining / 1.0
                bar_x, bar_y = w // 2 - 120, 20
                cv2.rectangle(frame, (bar_x, bar_y),
                              (bar_x + 240, bar_y + 12), (60, 60, 60), -1)
                cv2.rectangle(frame, (bar_x, bar_y),
                              (bar_x + int(240 * progress), bar_y + 12),
                              (80, 160, 255), -1)
                cv2.putText(frame, 'Next letter...', (bar_x, bar_y + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

            # Letter badge — green if confident, orange if low confidence
            badge_color = (50, 220, 120) if confidence > CONFIDENCE_THRESHOLD else (30, 140, 255)
            cv2.rectangle(frame, (w - 120, 20), (w - 20, 110), badge_color, -1)
            cv2.putText(frame, predicted_letter, (w - 100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)
            cv2.putText(frame, f'{confidence * 100:.0f}%', (w - 110, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        else:
            # No hand — reset hold but keep current word
            last_letter = ''
            hold_start = None

        # ── Bottom HUD ────────────────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 80), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        word_display = ''.join(current_word) + '_'
        cv2.putText(frame, 'Word:', (20, h - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
        cv2.putText(frame, word_display, (100, h - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, '[SPACE] save word   [BKSP] delete   [Q] quit',
                    (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

        cv2.imshow('Sign Language Translator', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            if current_word:
                print(f'Word saved: {"".join(current_word)}')
                current_word = []
                last_letter = ''
                hold_start = None
        elif key == 8:  # Backspace
            if current_word:
                current_word.pop()

cap.release()
cv2.destroyAllWindows()