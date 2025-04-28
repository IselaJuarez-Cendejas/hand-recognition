import cv2
import mediapipe as mp
import numpy as np
import time
#try:
from hand_direction import ( determine_hand_direction, find_boundary_landmarks, calculate_angle_between_points, display_hand_info )
#except ImportError:
 #   raise ImportError("The module 'hand_direction' could not be found. Ensure 'hand_direction.py' exists in the same directory or provide the correct path.")

CAM_W = 1280
CAM_H = 720
TEXT_COLOR = (243,236,27)
LM_COLOR = (102,255,255)
LINE_COLOR = (51,51,51)

# A hand detector based on mediapipe, it can detect hands and return several features of hands:
#   'label'         - handedness of hands, 'left', 'right'
#   'landmarks'     - the coordinates of 21 hand joints
#   'wrist_angle'   - angle of <index finger mcp, wrist, pinky mcp>
#   'direction'     - the direction that a hand is pointing, 'up', 'down', 'left', 'right'
#   'facing'        - the facing of hands, 'front', 'back' ('front' means the palm is facing the camera)
#   'boundary'      - the boundary joints from 'up', 'down', 'left', 'right'


class HandDetector:
    def __init__(self, mode=False, maxHands=1, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils # it gives small dots onhands total 20 landmark points
    def detect_hands(self, img):
        self.decoded_hands = None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            h, w, _ = img.shape
            num_hands = len(self.results.multi_hand_landmarks)
            self.decoded_hands = [None] * num_hands

            for i in range(num_hands):
                self.decoded_hands[i] = dict()
                lm_list = list()
                handedness = self.results.multi_handedness[i]
                hand_landmarks = self.results.multi_hand_landmarks[i]
                wrist_z = hand_landmarks.landmark[0].z

                for lm in hand_landmarks.landmark:
                    cx = int(lm.x * w)
                    cy = int(lm.y * h)
                    cz = int((lm.z - wrist_z) * w)
                    lm_list.append([cx, cy, cz])

                label = handedness.classification[0].label.lower()
                lm_array = np.array(lm_list)
                direction, facing = determine_hand_direction(lm_array, label)
                boundary = find_boundary_landmarks(lm_array)
                wrist_angle_joints = lm_array[[5, 0, 17]]
                wrist_angle = calculate_angle_between_points(wrist_angle_joints)

                self.decoded_hands[i]['label'] = label
                self.decoded_hands[i]['landmarks'] = lm_array
                self.decoded_hands[i]['wrist_angle'] = wrist_angle
                self.decoded_hands[i]['direction'] = direction
                self.decoded_hands[i]['facing'] = facing
                self.decoded_hands[i]['boundary'] = boundary

        return self.decoded_hands

    def draw_landmarks(self, img):
        w = img.shape[1]
        t = int(w / 500)
        if self.results.multi_hand_landmarks:
            for landmarks in self.results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(img, landmarks, self.mp_hands.HAND_CONNECTIONS,
                                               self.mp_drawing.DrawingSpec(color=LM_COLOR, thickness=3*t, circle_radius=t),
                                               self.mp_drawing.DrawingSpec(color=LINE_COLOR, thickness=t, circle_radius=t))

    def findHands(self,img,draw=True):
        # Send rgb image to hands
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB) # process the frame
        #     print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    #Draw dots and connect them
                    self.mpDraw.draw_landmarks(img,handLms,
                                               self.mpHands.HAND_CONNECTIONS)

def main():
    max_hands = 2  # Convert max_hands to an integer
    cap = cv2.VideoCapture(0)
    cap.set(3, CAM_W)
    cap.set(4, CAM_H)
    detector = HandDetector()  # Pass max_hands as an integer
    ptime = 0
    ctime = 0

    while True:
        _, img = cap.read()
        img = cv2.flip(img, 1)
        detector.detect_hands(img)
        detector.findHands(img)
        if detector.decoded_hands:
            for hand in detector.decoded_hands:
                display_hand_info(img, hand)

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        cv2.putText(img, f'FPS: {int(fps)}', (50,50), 0, 0.8,
                    TEXT_COLOR, 2, lineType=cv2.LINE_AA)

        cv2.imshow('Hand detection', img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
