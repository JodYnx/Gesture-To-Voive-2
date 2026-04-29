import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import joblib
import pygame
import time

# =========================
# تحميل المودل والـ scaler
# =========================
model = tf.keras.models.load_model("best_sign_model.keras")  
scaler = joblib.load("scaler.save")

# =========================
# الكلمات
# =========================
labels = [
    "alsalam_kum",
    "kaif_halik",
    "kam_alsaah",
    "momkin_tosaedni",
    "sabah_alkhair",
    "win_alhamam",
    "win_altawaree"
]

# =========================
# الصوت (حل مشاكل WASAPI)
# =========================
try:
    pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
except:
    print("Audio init failed")

audio_files = {
    "alsalam_kum": "audio/Sara_alsalam_kum.mp3",
    "kaif_halik": "audio/Sara_kaif_halik.mp3",
    "kam_alsaah": "audio/Sara_kam_alsaah.mp3",
    "momkin_tosaedni": "audio/Sara_momkin_tosaedni.mp3",
    "sabah_alkhair": "audio/Sara_sabah_alkhair.mp3",
    "win_alhamam": "audio/Sara_win_alhamam.mp3",
    "win_altawaree": "audio/Sara_win_altawaree.mp3"
}

# =========================
# MediaPipe (يدين)
# =========================
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=2,  # مهم
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# =========================
# الكاميرا
# =========================
cap = cv2.VideoCapture(0)

sequence = []
predictions = []
sentence = []

SEQUENCE_LENGTH = 30
last_audio_time = 0
COOLDOWN = 2  # ثانيتين بين كل صوت

# =========================
# تشغيل الصوت
# =========================
def play_audio(word):
    global last_audio_time

    if time.time() - last_audio_time < COOLDOWN:
        return

    if word in audio_files:
        try:
            pygame.mixer.music.load(audio_files[word])
            pygame.mixer.music.play()
            last_audio_time = time.time()
        except:
            print("Audio error:", word)

# =========================
# تشغيل الكاميرا
# =========================
while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        continue

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    left_hand = np.zeros(63)
    right_hand = np.zeros(63)

    if results.multi_hand_landmarks and results.multi_handedness:

        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):

            label = handedness.classification[0].label

            hand_array = []
            for lm in hand_landmarks.landmark:
                hand_array.extend([lm.x, lm.y, lm.z])

            if label == "Left":
                left_hand = np.array(hand_array)

            elif label == "Right":
                right_hand = np.array(hand_array)

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    # دمج اليدين (126 feature)
    frame_features = np.concatenate([left_hand, right_hand])
    sequence.append(frame_features)

    if len(sequence) > SEQUENCE_LENGTH:
        sequence.pop(0)

    if len(sequence) == SEQUENCE_LENGTH:

        input_data = np.array(sequence)

        # scaler
        input_data = scaler.transform(input_data)
        input_data = input_data.reshape(1, SEQUENCE_LENGTH, 126)

        prediction = model.predict(input_data, verbose=0)[0]

        predicted_class = np.argmax(prediction)
        confidence = prediction[predicted_class]

        if confidence > 0.75:  # شددنا الشرط

            word = labels[predicted_class]
            predictions.append(word)

            # فلترة الاهتزاز
            if len(predictions) > 12:

                recent = predictions[-12:]

                if recent.count(word) > 8:

                    if len(sentence) == 0 or word != sentence[-1]:

                        sentence.append(word)

                        print(f"Detected: {word} ({confidence:.2f})")

                        play_audio(word)

    # عرض النص
    cv2.rectangle(frame, (0, 0), (640, 60), (0, 0, 0), -1)

    text = " ".join(sentence[-3:])

    cv2.putText(
        frame,
        text,
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    cv2.imshow("Sign Language Recognition", frame)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()