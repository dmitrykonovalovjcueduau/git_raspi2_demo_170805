from __future__ import print_function
import numpy as np
import cv2
from os.path import expanduser

home = expanduser("~")

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
    import os
    os.chdir(os.path.join(home, 'darknet'))
    img = cv2.imread('predictions.jpg', 0)
    cv2.imshow('image', img)


def save_input_to_darknet(img):
    print('save_input_to_darknet()...')
    import os
    os.chdir(os.path.join(home, 'darknet'))
    cv2.imwrite('from_cv2.jpg', img)


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


