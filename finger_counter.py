import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=6,
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

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:
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

        # Display finger count
        cv2.putText(image, str(finger_count), (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 10)

        # Display image
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
