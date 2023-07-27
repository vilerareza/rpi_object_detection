import argparse
import time
from picamera2 import Picamera2
import cv2 as cv
import dlib


def start_camera(flip = True, res=(640,480)):

    # define a video capture object
    cam = Picamera2()
    # Configure the camera
    config = cam.create_preview_configuration(main={"size": res, "format": "BGR888"})
    cam.configure(config)
    cam.start()
    print ('Camera is running')
    # dlib face detector
    face_detector = dlib.get_frontal_face_detector()


    while(True):

        try:
            #t1 = time.time()
            # Read the frame
            frame = cam.capture_array()
            # Frame conversion to gray
            img_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            # Flip
            if flip:
                img_gray = cv.rotate(img_gray, cv.ROTATE_180)

            # Face detection
            rects = face_detector(img_gray, 0)

            if len(rects) != 0:
                print (f'Face detected: {len(rects)}')
        
            #t2 = time.time()
            #print (f'frame_time: {t2-t1}')

        except Exception as e:
            print (e)
            # On error, release the camera object
            cam.stop()
            break


def main(predictor_path = '.', res = (640, 480)):
    
    print ('Starting...')

    # Start camera
    start_camera()


if __name__ == '__main__':

    # Argument handler
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', type = str, default = '../shape_predictor_68_face_landmarks.dat', required = False)

    # Parsing
    args = parser.parse_args()
    predictor_path = args.pred_path

    # Run
    main(predictor_path)
