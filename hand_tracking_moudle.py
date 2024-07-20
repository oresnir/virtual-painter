import cv2
import mediapipe as mp
import time
import math


class HandDetector:
    def __init__(self, mode=False, max_hands=2, detection_con=0.5, track_con=0.5, model_complex=1):
        self.lm_list = None
        self.results = None
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con
        self.model_complex = model_complex

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.max_hands, self.model_complex,
                                        self.detection_con, self.track_con)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hand_num=0, draw=True):
        x_list = []
        y_list = []
        bbox = []
        self.lm_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_num]
            for id, lm in enumerate(my_hand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                x_list.append(cx)
                y_list.append(cy)
                # print(id, cx, cy)
                self.lm_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            x_min, x_max = min(x_list), max(x_list)
            y_min, y_max = min(y_list), max(y_list)
            bbox = x_min, y_min, x_max, y_max

            if draw:
                cv2.rectangle(img, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (0, 255, 0), 2)
        return self.lm_list, bbox

    def fingers_up(self):
        fingers = []
        # Thumb
        if self.thumb_up():
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for id in range(1, 5):
            if self.lm_list[self.tipIds[id]][2] < self.lm_list[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

            # totalFingers = fingers.count(1)

        return fingers

    def thumb_up(self, threshold=120):
        thumb_to_pinkie_x = self.lm_list[self.tipIds[0]][1] - self.lm_list[self.tipIds[4]][1]
        thumb_to_pinkie_y = self.lm_list[self.tipIds[0]][2] - self.lm_list[self.tipIds[4]][2]
        if math.fabs(thumb_to_pinkie_x) > threshold:
            # Thumb is on the right
            if thumb_to_pinkie_x > 0:
                return self.lm_list[self.tipIds[0]][1] > self.lm_list[self.tipIds[0] - 1][1]
            # Thumb is on the left
            else:
                return self.lm_list[self.tipIds[0]][1] < self.lm_list[self.tipIds[0] - 1][1]

        elif math.fabs(thumb_to_pinkie_y) > threshold:
            # Thumb is on the bottom
            if thumb_to_pinkie_y > 0:
                return self.lm_list[self.tipIds[0]][2] > self.lm_list[self.tipIds[0] - 1][2]
            # Thumb is on the top
            else:
                return self.lm_list[self.tipIds[0]][2] < self.lm_list[self.tipIds[0] - 1][2]
        else:
            return False

    def find_distance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lm_list[p1][1:]
        x2, y2 = self.lm_list[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
            length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]


# def main():
#     p_time, c_time = 0, 0
#     cap = cv2.VideoCapture(0)
#     detector = HandDetector()
#     while True:
#         success, img = cap.read()
#         img = detector.find_hands(img)
#         lm_list, bbox = detector.find_position(img)

#         c_time = time.time()
#         fps = 1 / (c_time - p_time)
#         p_time = c_time

#         cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
#         cv2.imshow("Image", img)
#         cv2.waitKey(1)


# if __name__ == "__main__":
#     main()
