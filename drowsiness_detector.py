import cv2
import mediapipe as mp
import numpy as np
from math import hypot
import os

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)

# Eye landmark indices (MediaPipe Face Mesh)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

MOUTH = [13, 14, 78, 308, 82, 87, 312, 317]

def eye_aspect_ratio(eye_points, landmarks, frame_width, frame_height):
    points = []
    for point in eye_points:
        x = int(landmarks[point].x * frame_width)
        y = int(landmarks[point].y * frame_height)
        points.append((x, y))

    # Vertical distances
    v1 = hypot(points[1][0] - points[5][0], points[1][1] - points[5][1])
    v2 = hypot(points[2][0] - points[4][0], points[2][1] - points[4][1])

    # Horizontal distance
    h = hypot(points[0][0] - points[3][0], points[0][1] - points[3][1])

    ear = (v1 + v2) / (2.0 * h)
    return ear

def mouth_aspect_ratio(landmarks, frame_width, frame_height):
    # Top and bottom lip
    top = landmarks[13]
    bottom = landmarks[14]

    # Left and right mouth corners
    left = landmarks[78]
    right = landmarks[308]

    top = (int(top.x * frame_width), int(top.y * frame_height))
    bottom = (int(bottom.x * frame_width), int(bottom.y * frame_height))
    left = (int(left.x * frame_width), int(left.y * frame_height))
    right = (int(right.x * frame_width), int(right.y * frame_height))

    vertical = hypot(top[0] - bottom[0], top[1] - bottom[1])
    horizontal = hypot(left[0] - right[0], left[1] - right[1])

    mar = vertical / horizontal
    return mar
MAR_THRESHOLD = 0.6
YAWN_FRAMES = 15
yawn_counter = 0

frame_counter = 0
EAR_THRESHOLD = 0.23
CLOSED_FRAMES = 20
alarm_on = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    frame_height, frame_width, _ = frame.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            left_ear = eye_aspect_ratio(LEFT_EYE, landmarks, frame_width, frame_height)
            right_ear = eye_aspect_ratio(RIGHT_EYE, landmarks, frame_width, frame_height)

            avg_ear = (left_ear + right_ear) / 2.0
            mar = mouth_aspect_ratio(landmarks, frame_width, frame_height)

            if mar > MAR_THRESHOLD:
                yawn_counter += 1
            else:
                yawn_counter = 0

            if yawn_counter > YAWN_FRAMES:
                cv2.putText(frame, "YAWNING ALERT !!!", (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

                os.system("afplay /System/Library/Sounds/Sosumi.aiff")

            if avg_ear < EAR_THRESHOLD:
                frame_counter += 1
            else:
                frame_counter = 0

            if frame_counter > CLOSED_FRAMES:
                cv2.putText(frame, "DROWSY ALERT !!!", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

                if not alarm_on:
                    os.system("afplay /System/Library/Sounds/Sosumi.aiff")
                    alarm_on = True
            else:
                alarm_on = False

            cv2.putText(frame, f"EAR: {round(avg_ear, 2)}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()