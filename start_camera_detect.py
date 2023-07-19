import time
from picamera2 import Picamera2
import cv2 as cv
import numpy as np
import tflite_runtime.interpreter as tflite
from utils import create_label_dict


# Path to tflite model
model_path = 'efficientdet_lite0.tflite'
# Path to test image
test_image_path = 'test_data/test_img_1.jpg'
# Path to test image annotated with prediction bounding box
test_image_bbox_path = 'test_data/test_img_1_bbox.jpg'
# Detection score threshold
det_score_thres = 0.2
# Path to id2name txt file
id2name_path = 'labelmap.txt'


def create_detector(model_path):
    # Initialize the object detector
    detector = tflite.Interpreter(model_path)
    detector.allocate_tensors()
    return detector


def start_camera(flip = True, res=(640,480), model_path = '.', id2name_path = '.', det_score_thres=0.2):

    # Creating detector
    detector = create_detector(model_path)  
    detector_output = detector.get_output_details()
    detector_input = detector.get_input_details()[0]

    # Create dictionary to map class ID to class name
    id2name_dict = create_label_dict(id2name_path)

    # Initialize the camera
    cam = Picamera2()
    # Configure the camera
    config = cam.create_preview_configuration(main={"size": res, "format": "BGR888"})
    cam.configure(config)
    # Start the camera
    cam.start()

    while(True):

        try:
            t1 = time.time()
            # Read the frame
            frame = cam.capture_array()

            ''' Preprocess '''
            # Convert BGR to RGB
            frame = frame[:,:,::-1]
            # The EfficientDet model require the input size to be (320 x 320) 
            frame = cv.resize(frame,(320,320))
            # Flip
            if flip:
               frame = cv.rotate(frame, cv.ROTATE_180)

            frame = np.expand_dims(frame, axis=0)

            ''' Run onject detection '''
            detector.set_tensor(detector_input['index'], frame)
            detector.invoke()
            bboxes = detector.get_tensor(detector_output[0]['index'])[0]
            class_id = detector.get_tensor(detector_output[1]['index'])[0]
            scores = detector.get_tensor(detector_output[2]['index'])[0]

            if len(bboxes) != 0:
                for i in range(len(bboxes)):
                    if scores[i] >= det_score_thres:
                        try:
                            print (id2name_dict[class_id[i]])
                        except:
                            print (f'Class name does not exist for label ID {class_id[i]}') 

            t2 = time.time()
            print (f'frame_time: {t2-t1}')


        except Exception as e:
            print (e)
            # On error, release the camera object
            cam.stop()
            break

def main():
    start_camera(model_path=model_path, id2name_path=id2name_path)

if __name__ == '__main__':
    main()

