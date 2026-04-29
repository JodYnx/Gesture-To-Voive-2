import os
import cv2
import numpy as np
import mediapipe as mp

# =========================
# المسارات
# =========================
DATA_PATH = "data/videos"
SAVE_PATH = "data/landmarks"
MODEL_PATH = "hand_landmarker.task"

# =========================
# إعداد MediaPipe
# =========================
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2   
)

landmarker = HandLandmarker.create_from_options(options)

# =========================
# استخراج اللاند ماركس
# =========================
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("DATA_PATH not found")

os.makedirs(SAVE_PATH, exist_ok=True)

for word in os.listdir(DATA_PATH):

    word_folder = os.path.join(DATA_PATH, word)

    if not os.path.isdir(word_folder):
        continue

    save_word_folder = os.path.join(SAVE_PATH, word)
    os.makedirs(save_word_folder, exist_ok=True)

    for video in os.listdir(word_folder):

        video_path = os.path.join(word_folder, video)

        if not video.lower().endswith((".mp4", ".avi", ".mov")):
            continue

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Failed to open {video_path}")
            continue

        sequence = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=frame_rgb
            )

            results = landmarker.detect(mp_image)

            # =========================
            # تجهيز اليدين (126 feature)
            # =========================
            left_hand = np.zeros(21 * 3)
            right_hand = np.zeros(21 * 3)

            if results.hand_landmarks:

                for idx, hand_landmarks in enumerate(results.hand_landmarks):

                    handedness = results.handedness[idx][0].category_name

                    hand_array = []

                    for lm in hand_landmarks:
                        hand_array.extend([lm.x, lm.y, lm.z])

                    if handedness == "Left":
                        left_hand = np.array(hand_array)

                    elif handedness == "Right":
                        right_hand = np.array(hand_array)

            # دمج اليدين
            frame_landmarks = np.concatenate([left_hand, right_hand])  # 126
            sequence.append(frame_landmarks)

        cap.release()

        if len(sequence) == 0:
            print(f"No hand detected in {video}")
            continue

        sequence = np.array(sequence)

        filename = os.path.splitext(video)[0]
        save_file_path = os.path.join(save_word_folder, filename)

        np.save(save_file_path, sequence)

        print(f"Saved: {save_file_path}.npy")

print("All landmarks extracted successfully.")