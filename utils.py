import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic

def extract_landmarks(results):
    pose = np.array([[res.x, res.y, res.z] 
                     for res in results.pose_landmarks.landmark]).flatten() \
                     if results.pose_landmarks else np.zeros(33*3)

    left_hand = np.array([[res.x, res.y, res.z] 
                          for res in results.left_hand_landmarks.landmark]).flatten() \
                          if results.left_hand_landmarks else np.zeros(21*3)

    right_hand = np.array([[res.x, res.y, res.z] 
                           for res in results.right_hand_landmarks.landmark]).flatten() \
                           if results.right_hand_landmarks else np.zeros(21*3)

    return np.concatenate([pose, left_hand, right_hand])