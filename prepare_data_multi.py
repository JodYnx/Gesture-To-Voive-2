import os
import numpy as np

# =========================
# الإعدادات
# =========================
DATA_PATH = "data/landmarks"
SEQUENCE_LENGTH = 30
FEATURE_SIZE = 126   

AUGMENTATIONS = 2   

# =========================
# التحقق من المسار
# =========================
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("Landmarks folder not found")

# =========================
# Flip Augmentation (الأهم)
# =========================
def flip_sequence(seq):

    flipped_sequence = []

    for frame in seq:

        left = frame[:63].copy()
        right = frame[63:].copy()

        left = left.reshape(21, 3)
        right = right.reshape(21, 3)

        # عكس X
        left[:, 0] = 1 - left[:, 0]
        right[:, 0] = 1 - right[:, 0]

        # تبديل اليدين
        new_left = right
        new_right = left

        new_frame = np.concatenate([
            new_left.flatten(),
            new_right.flatten()
        ])

        flipped_sequence.append(new_frame)

    return np.array(flipped_sequence)


# =========================
# Noise Augmentation
# =========================
def noise_sequence(seq):

    noise = np.random.normal(0, 0.01, seq.shape)
    shift = np.random.uniform(-0.02, 0.02)

    return seq + noise + shift


# =========================
# تجهيز البيانات
# =========================
X = []
y = []

labels = sorted([
    label for label in os.listdir(DATA_PATH)
    if os.path.isdir(os.path.join(DATA_PATH, label))
])

label_map = {label: idx for idx, label in enumerate(labels)}

print("Label mapping:", label_map)

for label in labels:

    label_folder = os.path.join(DATA_PATH, label)

    for file in os.listdir(label_folder):

        if not file.endswith(".npy"):
            continue

        file_path = os.path.join(label_folder, file)

        seq = np.load(file_path)

        # تحقق
        if seq.shape[0] == 0 or seq.shape[1] != FEATURE_SIZE:
            print(f"Skipped invalid file: {file}")
            continue

        # =========================
        # Padding / Truncation
        # =========================
        if len(seq) >= SEQUENCE_LENGTH:
            seq = seq[:SEQUENCE_LENGTH]
        else:
            padding = np.zeros((SEQUENCE_LENGTH - len(seq), FEATURE_SIZE))
            seq = np.vstack((seq, padding))

        # =========================
        # الأصل
        # =========================
        X.append(seq)
        y.append(label_map[label])

        # =========================
        # Flip (مهم جدًا)
        # =========================
        flipped = flip_sequence(seq)
        X.append(flipped)
        y.append(label_map[label])

        # =========================
        # Noise Augmentation
        # =========================
        for _ in range(AUGMENTATIONS):
            aug = noise_sequence(seq)
            X.append(aug)
            y.append(label_map[label])


# =========================
# تحويل إلى numpy
# =========================
X = np.array(X)
y = np.array(y)

print("Final Dataset shape:", X.shape)
print("Labels shape:", y.shape)

# =========================
# حفظ الملفات
# =========================
np.save("X_multi.npy", X)
np.save("y_multi.npy", y)

print("Data preparation completed successfully.")