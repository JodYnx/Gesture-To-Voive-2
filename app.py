from flask import Flask, render_template, request, jsonify
from keras.models import load_model
import numpy as np
import mediapipe as mp
import joblib
import cv2
import base64
import time
from io import BytesIO

app = Flask(__name__)

# ==========================================
# 1. تحميل النموذج والبيانات والفلاتر القياسية
# ==========================================
model = load_model("best_sign_model.keras")
scaler = joblib.load("scaler.save")

labels = {
    "alsalam_kum": ["السلام عليكم", "Peace be upon you"],
    "win_alhamam": ["وين دورة المياه", "Where is the restroom"],
    "win_altawaree": ["وين الطوارئ", "Where is emergency"]
}

class_names = ["alsalam_kum", "win_alhamam", "win_altawaree"]

# الإعدادات الموزونة: الحفاظ على الـ 30 فريم لسلامة الموديل مع تسريع الفلاتر
MOTION_THRESHOLD = 0.01
CONF_THRESHOLD = 0.55
SEQUENCE_LENGTH = 30     # رجعناها 30 منعاً لانهيار أبعاد الموديل (Shape Error)
COOLDOWN = 1.5           # سرعة استجابة الصوت

# ذاكرة الحسابات المستمرة
sequence = []
predictions_history = []
last_valid = np.zeros(126)
last_audio_time = 0
current_sentence = "..."
current_sentence_en = "Waiting for gesture"

# ==========================================
# 2. إعداد ميديا بايب
# ==========================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==========================================
# 3. دالات الفلاتر المساعدة
# ==========================================
def normalize_hand(hand_array):
    hand = hand_array.reshape(21, 3)
    wrist = hand[0]
    hand = hand - wrist
    max_val = np.max(np.abs(hand))
    if max_val > 0:
        hand = hand / max_val
    return hand.flatten()

def calculate_motion(seq):
    if len(seq) < 2:
        return 0
    diff = np.abs(seq[-1] - seq[-2])
    return np.mean(diff)

# ==========================================
# 4. مسارات صفحات الويب (Routes)
# ==========================================
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/translator")
def translator():
    return render_template("translator.html")

@app.route("/predict", methods=["POST"])
def predict():
    global sequence, last_valid, predictions_history, last_audio_time, current_sentence, current_sentence_en

    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"arabic": current_sentence, "english": current_sentence_en, "playAudio": False})

    image_data = data["image"].split(",")[1]
    image_bytes = base64.b64decode(image_data)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    left_hand = None
    right_hand = None

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label

            hand_array = []
            for lm in hand_landmarks.landmark:
                hand_array.extend([lm.x, lm.y, lm.z])

            hand_array = np.array(hand_array)
            hand_array = normalize_hand(hand_array)

            if label == "Left":
                left_hand = hand_array
            elif label == "Right":
                right_hand = hand_array

    lh_data = left_hand if left_hand is not None else np.zeros(63)
    rh_data = right_hand if right_hand is not None else np.zeros(63)

    frame_features = np.concatenate([lh_data, rh_data])

    if np.all(frame_features == 0):
        frame_features = last_valid
    else:
        last_valid = frame_features

    sequence.append(frame_features)
    if len(sequence) > SEQUENCE_LENGTH:
        sequence.pop(0)

    should_play_audio = False

    # التحليل يبدأ فوراً عند توفر الـ 30 فريم المتوافقة مع الموديل
    if len(sequence) == SEQUENCE_LENGTH:
        motion = calculate_motion(sequence)

        if motion >= MOTION_THRESHOLD:
            # الترتيب الافتراضي
            input_data1 = np.array(sequence)
            input_data_scaled1 = scaler.transform(input_data1)
            input_data_reshaped1 = input_data_scaled1.reshape(1, SEQUENCE_LENGTH, 126)

            prediction = model.predict(input_data_reshaped1, verbose=0)[0]
            predicted_class = np.argmax(prediction)
            confidence = prediction[predicted_class]

            # الفحص المعكوس لحل مشكلة المرآة في الويب
            if confidence < CONF_THRESHOLD and (left_hand is not None or right_hand is not None):
                reversed_sequence = []
                for seq_frame in sequence:
                    lf = seq_frame[:63]
                    rf = seq_frame[63:]
                    reversed_sequence.append(np.concatenate([rf, lf]))
                
                input_data2 = np.array(reversed_sequence)
                input_data_scaled2 = scaler.transform(input_data2)
                input_data_reshaped2 = input_data_scaled2.reshape(1, SEQUENCE_LENGTH, 126)

                prediction2 = model.predict(input_data_reshaped2, verbose=0)[0]
                predicted_class2 = np.argmax(prediction2)
                confidence2 = prediction2[predicted_class2]

                if confidence2 > confidence:
                    predicted_class = predicted_class2
                    confidence = confidence2

            # آلية التصويت فائقة السرعة والخفيفة جداً
            if predicted_class < len(class_names) and confidence > CONF_THRESHOLD:
                word = class_names[predicted_class]
                predictions_history.append(word)

                # نفحص آخر 3 تنبؤات فقط؛ إذا تكررت الإشارة مرتين متتاليتين، نعتمدها وننطقها فوراً دون تأخير
                if len(predictions_history) > 3:
                    recent = predictions_history[-3:]
                    if recent.count(word) >= 2:
                        if current_sentence == "..." or word != predictions_history[-4]:
                            current_sentence = labels[word][0]
                            current_sentence_en = labels[word][1]
                            
                            if time.time() - last_audio_time > COOLDOWN:
                                should_play_audio = True
                                last_audio_time = time.time()

    return jsonify({
        "arabic": current_sentence,
        "english": current_sentence_en,
        "playAudio": should_play_audio
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
