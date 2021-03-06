from __future__ import print_function
import numpy as np
import cv2
# http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html

def main():
    import subprocess
    from subprocess import call
    import os
    from os.path import expanduser
    home = expanduser("~")

    subprocess.call("ls")
    os.chdir(os.path.join(home, 'darknet'))
    # os.chdir(os.path.join(home, 'lib/darknet'))
    subprocess.call("ls")

    # ./darknet detector test cfg/voc.data cfg/tiny-yolo-voc.cfg tiny-yolo-voc.weights data/dog.jpg
    call(["./darknet", "detector", "test", "cfg/voc.data",
          "cfg/tiny-yolo-voc.cfg", "tiny-yolo-voc.weights", "data/dog.jpg"])
    # show_webcam()


if __name__ == '__main__':
    main()


