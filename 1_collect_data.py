import os
import cv2

CLASSES = CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']  # starting with just 5 letters for now
DATA_DIR = './data'
IMAGES_PER_CLASS = 100

os.makedirs(DATA_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)

for label in CLASSES:
    class_dir = os.path.join(DATA_DIR, label)
    os.makedirs(class_dir, exist_ok=True)

    print(f'\nGet ready to show gesture for: {label}')
    print('Press Q when ready to start capturing...')

    while True:
        ret, frame = cap.read()
        cv2.putText(frame, f'Ready for: {label}  -- Press Q to start', (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Collector', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    count = 0
    while count < IMAGES_PER_CLASS:
        ret, frame = cap.read()
        cv2.putText(frame, f'Capturing {label}: {count}/{IMAGES_PER_CLASS}', (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)
        cv2.imshow('Collector', frame)
        cv2.imwrite(os.path.join(class_dir, f'{count}.jpg'), frame)
        count += 1
        cv2.waitKey(25)

    print(f'Done collecting for: {label}')

cap.release()
cv2.destroyAllWindows()
print('\nAll done! Check your data/ folder.')