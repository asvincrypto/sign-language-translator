import os
import pickle
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

# New API for mediapipe 0.10+
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

import urllib.request

model_path = './hand_landmarker.task'
if not os.path.exists(model_path):
    print('Downloading hand landmark model...')
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task',
        model_path
    )
    print('Downloaded!')

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

DATA_DIR = './data'
data = []
labels = []

print('Processing images...')

with HandLandmarker.create_from_options(options) as landmarker:
    for label in os.listdir(DATA_DIR):
        label_path = os.path.join(DATA_DIR, label)
        if not os.path.isdir(label_path):
            continue

        print(f'Processing letter: {label}')

        for img_file in os.listdir(label_path):
            img_path = os.path.join(label_path, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            result = landmarker.detect(mp_image)

            if result.hand_landmarks:
                hand_landmarks = result.hand_landmarks[0]
                x_coords = [lm.x for lm in hand_landmarks]
                y_coords = [lm.y for lm in hand_landmarks]

                x_min = min(x_coords)
                y_min = min(y_coords)

                features = []
                for lm in hand_landmarks:
                    features.append(lm.x - x_min)
                    features.append(lm.y - y_min)

                data.append(features)
                labels.append(label)

os.makedirs('./model', exist_ok=True)

with open('./model/sign_model_data.pkl', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f'\nDone! {len(data)} samples saved across {len(set(labels))} letters.')