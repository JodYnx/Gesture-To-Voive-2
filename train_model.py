import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# =========================
# 1) تحميل البيانات
# =========================
X = np.load("X_multi.npy")
y = np.load("y_multi.npy")

print("Original X shape:", X.shape)
print("Original y shape:", y.shape)

num_samples, sequence_length, num_features = X.shape
num_classes = len(np.unique(y))

# =========================
# 2) Split
# =========================
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Train shape:", X_train.shape)
print("Validation shape:", X_val.shape)

# =========================
# 3) Normalization
# =========================
scaler = StandardScaler()

X_train_reshaped = X_train.reshape(-1, num_features)
X_val_reshaped = X_val.reshape(-1, num_features)

scaler.fit(X_train_reshaped)

import joblib
joblib.dump(scaler, "scaler.save")

X_train = scaler.transform(X_train_reshaped).reshape(-1, sequence_length, num_features)
X_val = scaler.transform(X_val_reshaped).reshape(-1, sequence_length, num_features)

# =========================
# 4) One-hot
# =========================
y_train = to_categorical(y_train, num_classes=num_classes)
y_val = to_categorical(y_val, num_classes=num_classes)

# =========================
# 5) Model (محسّن)
# =========================
model = Sequential()

model.add(LSTM(128, return_sequences=True, input_shape=(sequence_length, num_features)))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(LSTM(64))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Dense(64, activation="relu"))
model.add(Dropout(0.3))

model.add(Dense(num_classes, activation="softmax"))

# =========================
# 6) Compile
# =========================
model.compile(
    optimizer=Adam(learning_rate=0.0005),  # أبطأ = تعلم أدق
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# 7) Callbacks
# =========================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-5
)

checkpoint = ModelCheckpoint(
    "best_sign_model.keras",   #  فورمات حديث
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)

# =========================
# 8) Training
# =========================
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=150,
    batch_size=16,   # أكبر = استقرار أفضل
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1
)

# =========================
# 9) تقييم
# =========================
val_loss, val_acc = model.evaluate(X_val, y_val)
print(f"\nFinal Validation Accuracy: {val_acc * 100:.2f}%")

# =========================
# 10) حفظ
# =========================
model.save("sign_model_final.keras")

print("Training finished and model saved.")