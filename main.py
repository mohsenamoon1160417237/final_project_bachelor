import cv2
import mediapipe as mp
from math import hypot
import numpy as np
import osascript


def main(cap, max_num_hands: int):
    with mp_hands.Hands(
            model_complexity=0,
            max_num_hands=max_num_hands,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Initially set finger count to 0 for each cap
            finger_count = 0

            lm_list = []
            if results.multi_hand_landmarks:

                for hand_landmarks in results.multi_hand_landmarks:
                    for id, lm in enumerate(hand_landmarks.landmark):  # adding counter and returning it
                        # Get finger joint points
                        h, w, _ = image.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lm_list.append([id, cx, cy])  # adding to the empty list 'lmList'

                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    # Get hand index to check label (left or right)
                    hand_index = results.multi_hand_landmarks.index(hand_landmarks)
                    hand_label = results.multi_handedness[hand_index].classification[0].label

                    # Set variable to keep landmarks positions (x and y)
                    hand_land_marks = []

                    # Fill list with x and y positions of each landmark
                    for landmarks in hand_landmarks.landmark:
                        hand_land_marks.append([landmarks.x, landmarks.y])

                    # Test conditions for each finger: Count is increased if finger is
                    #   considered raised.
                    # Thumb: TIP x position must be greater or lower than IP x position,
                    #   deppeding on hand label.
                    if hand_label == "Left" and hand_land_marks[4][0] > hand_land_marks[3][0]:
                        finger_count = finger_count + 1
                    elif hand_label == "Right" and hand_land_marks[4][0] < hand_land_marks[3][0]:
                        finger_count = finger_count + 1

                    # Other fingers: TIP y position must be lower than PIP y position,
                    #   as image origin is in the upper left corner.
                    if hand_land_marks[8][1] < hand_land_marks[6][1]:  # Index finger
                        finger_count = finger_count + 1
                    if hand_land_marks[12][1] < hand_land_marks[10][1]:  # Middle finger
                        finger_count = finger_count + 1
                    if hand_land_marks[16][1] < hand_land_marks[14][1]:  # Ring finger
                        finger_count = finger_count + 1
                    if hand_land_marks[20][1] < hand_land_marks[18][1]:  # Pinky
                        finger_count = finger_count + 1

                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

            if len(lm_list):
                # getting the value at a point
                # x      #y
                x1, y1 = lm_list[4][1], lm_list[4][2]  # thumb
                x2, y2 = lm_list[8][1], lm_list[8][2]  # index finger
                # creating circle at the tips of thumb and index finger
                cv2.circle(image, (x1, y1), 13, (255, 0, 0), cv2.FILLED)  # image #fingers #radius #rgb
                cv2.circle(image, (x2, y2), 13, (255, 0, 0), cv2.FILLED)  # image #fingers #radius #rgb
                cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0),
                         3)  # create a line b/w tips of index finger and thumb

                length = hypot(x2 - x1, y2 - y1)  # distance b/w tips using hypotenuse
                # from numpy we find our length,by converting hand range in terms of volume range ie b/w -63.5 to 0
                # vol = np.interp(length, [30, 350], [volMin, volMax])
                volbar = np.interp(length, [30, 350], [400, 150])
                volper = np.interp(length, [30, 350], [0, 100])
                osascript.osascript("set volume output volume {}".format(volper))
                # print(vol, int(length))
                # volume.SetMasterVolumeLevel(vol, None)

                # Hand range 30 - 350
                # Volume range -63.5 - 0.0
                # creating volume bar for volume level
                cv2.rectangle(image, (50, 150), (85, 400), (0, 0, 255),
                              4)  # vid ,initial position ,ending position ,rgb ,thickness
                cv2.rectangle(image, (50, int(volbar)), (85, 400), (0, 0, 255), cv2.FILLED)
                cv2.putText(image, f"{int(volper)}%", (10, 40), cv2.FONT_ITALIC, 1, (0, 255, 98), 3)

            # Display finger count
            cv2.putText(image, str(finger_count), (50, 600), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 10)

            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
            # cap.release()


if __name__ == "__main__":
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(0)
    main(cap, 4)
