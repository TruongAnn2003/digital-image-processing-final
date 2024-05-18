import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import math
import cv2
import mediapipe as mp
from pynput.keyboard import Controller
import streamlit as st

keyboard = Controller()
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_SIMPLEX

st.title('Control car by hand')
frame_placeholder = st.empty()

def draw_text(text, bg_color):
    text_position = (50, 50)
    (text_width, text_height), baseline = cv2.getTextSize(text, font, 0.9, 2)
    rect_top_left = (text_position[0] - 5, text_position[1] - text_height - 5)
    rect_bottom_right = (text_position[0] + text_width + 5, text_position[1] + baseline + 5)

    cv2.rectangle(image, rect_top_left, rect_bottom_right, bg_color, -1)

    cv2.putText(image, text, text_position, font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print('Cannot open camera frame')
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)

        results = hands.process(image)
        imageHeight, imageWidth, _ = image.shape

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.waitKey(1)
        co = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                for point in mp_hands.HandLandmark:
                    if str(point) == "HandLandmark.WRIST":
                        normalizedLandmark = hand_landmarks.landmark[point]
                        pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark.x, normalizedLandmark.y, imageWidth, imageHeight)

                        try:
                            co.append(list(pixelCoordinatesLandmark))
                        except:
                            continue

        # no hand wrist detected
        print(co)
        if len(co) == 0:
            draw_text('Not playing', (0, 0, 255))
        # 2 hands were detected
        if len(co) == 2:
            xm, ym = (co[0][0] + co[1][0]) / 2, (co[0][1] + co[1][1]) /2 

            radius = 150

            try:
                m = (co[1][1] - co[0][1]) / (co[1][0] - co[0][0])  
            except:
                continue         
        
            m11 = math.floor(math.degrees(math.atan(m)))
            cv2.circle(img=image, center=(int(xm), int(ym)), radius=radius, color=(195, 255, 62), thickness=22)

            if (co[1][0] > co[0][0] and co[1][1] > co[0][1] and co[1][1] - co[0][1] > 65 and m11 <= 24) or (co[0][0] > co[1][0] and co[0][1] > co[1][1] and co[0][1] - co[1][1] > 65 and m11 <= 24):
                keyboard.release('s')
                keyboard.release('a')
                keyboard.press('d')

                draw_text('Turn right.', (0, 255, 0))

            elif (co[0][0] > co[1][0] and co[1][1] > co[0][1] and co[1][1] - co[0][1] > 65 and m11 >= -24) or (co[1][0] > co[0][0] and co[0][1] > co[1][1] and co[0][1] - co[1][1] > 65 and m11 >= -24):
                keyboard.release('s')
                keyboard.release('a')
                keyboard.press('a')

                draw_text('Turn left.', (0, 255, 0))
            else:
                print(math.degrees(math.atan(m)))
                keyboard.release('s')
                keyboard.release('a')
                keyboard.release('d')
                keyboard.press('w')

                draw_text('Keeping straight', (0, 255, 0))
        
        frame_placeholder.image(image, channels='BGR')

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()




    