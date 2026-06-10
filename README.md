# Sign Language Translator

I built this to explore how computer vision can be used to bridge communication gaps. It translates American Sign Language (ASL) hand gestures into text in real time using just a laptop webcam.

You hold an ASL gesture in front of your camera for 1.5 seconds and the letter appears on screen. Keep going and it builds words.

Under the hood it detects 21 landmarks on your hand every frame, normalizes their positions, and runs them through a Random Forest classifier trained on your own gesture data.

---

## Built with

Python, OpenCV, MediaPipe, scikit-learn

---

## Notes

- Collect data in the same lighting you plan to use the app in
- Each person should collect their own gesture data for best accuracy
- Similar letters like M, N, S, T require careful hand positioning

  ---

Run the scripts in order:

```bash
python 1_collect_data.py       # Capture gesture images (A-Z, ~30 mins)
python 2_create_dataset.py     # Extract hand landmarks
python 3_train_model.py        # Train the classifier
python 4_realtime_translator.py  # Run the live translator
```
