import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

=cap = cv2.VideoCapture(0)

def count_fingers(hand_landmarks):
    tips_ids = [4, 8, 12, 16, 20]
    fingers = []

    if hand_landmarks.landmark[tips_ids[0]].x < hand_landmarks.landmark[tips_ids[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    for id in range(1, 5):
        if hand_landmarks.landmark[tips_ids[id]].y < hand_landmarks.landmark[tips_ids[id] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers.count(1)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # نقلب الصورة بالعكس
    h, w, c = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            fingers_up = count_fingers(hand_landmarks)

            if fingers_up == 5:
                action = "Yes"
            elif fingers_up == 2:
                action = "Next"
            elif fingers_up == 1:
                action = "Back"
            else:
                action = "..."

            cv2.putText(frame, action, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                        2, (0, 255, 0), 3)

    cv2.imshow("Hand Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
