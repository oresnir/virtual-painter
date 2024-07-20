import math

import cv2
import numpy as np
import os
import hand_tracking_moudle as htm
from consts import folder_path, img_width, img_height, BrushSize, Colors, opacity, Shapes, Modes

color_index = {Colors.RED.value: 3, Colors.GREEN.value: 4, Colors.BLUE.value: 5,
               Colors.YELLOW.value: 6, Colors.BLACK.value: 7}


def selection_mode():
    global header, draw_color, draw_shape
    # Checking for the click
    if y1 < 125:
        if 100 < x1 < 250:
            header = header_images[1]
            draw_shape = Shapes.RECTANGLE.value
        if 280 < x1 < 380:
            header = header_images[2]
            draw_shape = Shapes.CIRCLE.value
        if 400 < x1 < 500:
            header = header_images[3]
            draw_color = Colors.RED.value
        elif 550 < x1 < 650:
            header = header_images[4]
            draw_color = Colors.GREEN.value
        elif 700 < x1 < 800:
            header = header_images[5]
            draw_color = Colors.BLUE.value
        elif 850 < x1 < 950:
            header = header_images[6]
            draw_color = Colors.YELLOW.value
        elif 1000 < x1 < 1100:
            header = header_images[7]
            draw_color = Colors.BLACK.value


def drawing_mode():
    global xp, yp, start, end, mode, last_mode, draw_shape, header
    cv2.circle(img, (x1, y1), 15, draw_color, cv2.FILLED)
    if xp == 0 and yp == 0:
        xp, yp = x1, y1
    if draw_shape == Shapes.LINE.value:
        cv2.line(img, (xp, yp), (x1, y1), draw_color, thickness)
        if draw_color == Colors.YELLOW.value:
            cv2.line(img_highlight, (xp, yp), (x1, y1), draw_color, thickness)
        elif draw_color == Colors.BLACK.value:
            cv2.line(img_highlight, (xp, yp), (x1, y1), draw_color, thickness)
            cv2.line(img_canvas, (xp, yp), (x1, y1), draw_color, thickness)
        else:
            cv2.line(img_canvas, (xp, yp), (x1, y1), draw_color, thickness)

    elif draw_shape == Shapes.RECTANGLE.value:
        if mode != last_mode:
            start, end = x1, y1
        cv2.rectangle(img, (start, end), (x1, y1), draw_color, thickness)
    else:
        if mode != last_mode:
            start, end = x1, y1
        radius = int(math.sqrt(math.pow(start-x1, 2) + math.pow(end-y1, 2)))
        cv2.circle(img, ((start + x1)//2, (end + y1)//2), radius, draw_color, thickness)

    xp, yp = x1, y1


def update_thickness():
    global thickness
    if sum(fingers) == 3 and fingers[1] and fingers[2] and fingers[3]:
        thickness = BrushSize.THIN.value
    elif sum(fingers) == 4 and not fingers[0]:
        thickness = BrushSize.REGULAR.value
    elif sum(fingers) == 5:
        thickness = BrushSize.THICK.value


def main():
    global header_images, img, header, draw_color, draw_shape, img_canvas, thickness, fingers, img_highlight
    global xp, yp, start, end, x1, y1, x2, y2, x, mode, last_mode
    cap, detector, rounds = set_up()
    clear_canvas()
    while True:
        rounds += 1
        # Import image
        success, img = cap.read()
        img = cv2.flip(img, 1)

        # Find Hand Landmarks
        img = detector.find_hands(img)
        position = detector.find_position(img, draw=False)

        if len(position) > 0 and len(position[0]) > 0:
            lm_list = position[0]

            # tip of index and middle fingers
            x1, y1 = lm_list[8][1:]
            x2, y2 = lm_list[12][1:]

            # Check which fingers are up
            fingers = detector.fingers_up()

            # Selection Mode - Two finger are up
            if fingers[1] and fingers[2]:
                xp, yp = 0, 0
                mode = Modes.SELECT.value
                selection_mode()

            # Drawing Mode - Index finger is up
            elif fingers[1] and not fingers[2]:
                mode = Modes.DRAW.value
                drawing_mode()

            if last_mode == Modes.DRAW.value and mode == Modes.SELECT.value:
                keep_shape()

            # Update Thickness Mode
            if sum(fingers) >= 3 and rounds % 10 == 0:
                update_thickness()

            # Clear Canvas when all fingers are down
            if sum(fingers) == 0:
                clear_canvas()

            # update last mode
            last_mode = mode

        edit_img()

        # Setting the header image
        img[0:125, 0:1280] = header

        # Adding the highlight
        img = cv2.addWeighted(img_highlight, opacity, img, 1 - opacity, 0)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


def edit_img():
    global img, img_canvas
    img_gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, img_inv)
    img = cv2.bitwise_or(img, img_canvas)


def set_up():
    global header_images, img, header, draw_color, draw_shape, xp, yp, start, end, mode, last_mode, thickness
    rounds = 0
    my_list = os.listdir(folder_path)
    header_images = []
    for img_path in my_list:
        img = cv2.imread(f'{folder_path}/{img_path}')
        header_images.append(img)
    header = header_images[0]
    draw_color = Colors.RED.value
    draw_shape = Shapes.LINE.value
    cap = cv2.VideoCapture(0)
    cap.set(3, img_width)
    cap.set(4, img_height)
    detector = htm.HandDetector(detection_con=0.65, max_hands=1)
    xp, yp, start, end = 0, 0, 0, 0
    mode, last_mode = Modes.DRAW.value, Modes.DRAW.value
    thickness = BrushSize.REGULAR.value
    return cap, detector, rounds


def clear_canvas():
    global img_canvas, img_highlight
    img_canvas = np.zeros((img_height, img_width, 3), np.uint8)
    img_highlight = np.zeros((img_height, img_width, 3), np.uint8)


def keep_shape():
    global draw_shape, header, img_canvas, draw_color, end, header_images, start, thickness, x1, y1, img_highlight
    if draw_shape == Shapes.RECTANGLE.value:
        if draw_color == Colors.YELLOW.value:
            cv2.rectangle(img_highlight, (start, end), (x1, y1), draw_color, thickness)
        else:
            cv2.rectangle(img_canvas, (start, end), (x1, y1), draw_color, thickness)
    elif draw_shape == Shapes.CIRCLE.value:
        radius = int(math.sqrt(math.pow(start-x1, 2) + math.pow(end-y1, 2)))
        if draw_color == Colors.YELLOW.value:
            cv2.circle(img_highlight, ((start + x1) // 2, (end + y1) // 2), radius, draw_color, thickness)
        else:
            cv2.circle(img_canvas, ((start + x1) // 2, (end + y1) // 2), radius, draw_color, thickness)

    draw_shape = Shapes.LINE.value
    header = header_images[color_index[draw_color]]


main()
