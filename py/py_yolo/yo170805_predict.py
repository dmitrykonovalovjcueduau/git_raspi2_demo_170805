from __future__ import print_function
import numpy as np
import cv2
from os.path import expanduser
import os

home = expanduser("~")

FROM_CV2 = os.path.join(home, 'darknet', 'from_cv2.jpg')
FROM_DARKNET = os.path.join(home, 'darknet', 'predictions.jpg')

def show_webcam():
    cam = cv2.VideoCapture(0)
    while True:
        cv2.destroyAllWindows()
        ret_val, img = cam.read()

        save_input_to_darknet(img)
        call_darknet()
        show_from_darknet()

        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cam.release()
    cv2.destroyAllWindows()

def main():
    show_webcam()


def show_from_darknet():
    print('show_from_darknet()...')
    img = cv2.imread(FROM_DARKNET, 0)
    cv2.imshow('image', img)


def save_input_to_darknet(img):
    print('save_input_to_darknet()...')
    if os.path.isfile(FROM_CV2):
        os.remove(FROM_CV2)
    if os.path.isfile(FROM_DARKNET):
        os.remove(FROM_DARKNET)
    cv2.imwrite(FROM_CV2, img)


def call_darknet():
    print('call_darknet()...')
    from subprocess import call
    import os
    os.chdir(os.path.join(home, 'darknet'))
    # ./darknet detector test cfg/voc.data cfg/tiny-yolo-voc.cfg tiny-yolo-voc.weights data/dog.jpg
    call(["./darknet", "detector", "test", "cfg/voc.data",
          "cfg/tiny-yolo-voc.cfg", "tiny-yolo-voc.weights", "from_cv2.jpg"])


if __name__ == '__main__':
    main()


