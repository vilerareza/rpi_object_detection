import argparse
import time
from picamera2 import Picamera2
import cv2 as cv
import dlib
from pygame import mixer as audio_mixer


def start_camera(flip = True, res=(640,480), audio_out=None):

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
                if audio_out is not None:
                    # Play the audio
                    audio_out.music.play()
        
            #t2 = time.time()
            #print (f'frame_time: {t2-t1}')

        except Exception as e:
            print (e)
            # On error, release the camera object
            cam.stop()
            break


def main(predictor_path = '.', res = (640, 480), audio_out = None):
    
    print ('Starting...')

    # Start camera
    start_camera(audio_out=audio_out)


if __name__ == '__main__':

    # Argument handler
    parser = argparse.ArgumentParser()
    # parser.add_argument('--pred_path', type = str, default = '../shape_predictor_68_face_landmarks.dat', required = False)
    parser.add_argument('--audio_enabled', type = bool, default = True, required = False)
    parser.add_argument('--audio_file_path', type = str, default = 'audio_test.mp3', required = False)
    parser.add_argument('--audio_vol', type = float, default = 0.7, required = False)

    # Parsing
    args = parser.parse_args()
    # predictor_path = args.pred_path
    audio_enabled = args.audio_enabled
    audio_file_path = args.audio_file_path
    audio_vol = args.audio_vol

    if audio_enabled:
        try:
            # Initialize pygame mixer
            audio_mixer.init()
            # Loading the audio
            audio_mixer.music.load(audio_file_path)
            # Setting the volume
            audio_mixer.music.set_volume(audio_vol)
            # Play the audio file

        except Exception as e:
            print (f'Error {e}: Check if the audio file path {audio_file_path} is correct')
            # Exit the program
            exit()
    else:
        audio_out = None

    # Run
    main(audio_out=audio_mixer)
