import time
from picamera2 import Picamera2
import cv2 as cv
import numpy as np
import tflite_runtime.interpreter as tflite
from utils import visualize, create_label_dict


# Path to tflite model
model_path = 'model_obj_detection.tflite'
# Detection score threshold
det_score_thres = 0.6
# Path to id2name txt file
id2name_path = 'labelmap_new.txt'


def create_detector(model_path):
    # Initialize the object detector
    detector = tflite.Interpreter(model_path)
    # Allocate memory for the model's input `Tensor`s
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
    print ('Camera is running')

    while(True):

        try:
            t1 = time.time()
            # Read the frame
            frame_ori = cam.capture_array()

            ''' Preprocess '''
            # Convert BGR to RGB
            frame = frame_ori.copy()
            frame = frame[:,:,::-1]
            # The EfficientDet model require the input size to be (320 x 320) 
            frame = cv.resize(frame,(384,384))
            # Flip
            if flip:
               frame = cv.rotate(frame, cv.ROTATE_180)

            frame = np.expand_dims(frame, axis=0)

            ''' Run object detection '''
            detector.set_tensor(detector_input['index'], frame)
            detector.invoke()
            bboxes = detector.get_tensor(detector_output[0]['index'])[0]
            class_id = detector.get_tensor(detector_output[1]['index'])[0]
            scores = detector.get_tensor(detector_output[2]['index'])[0]

            # Check if any object is detected
            if len(bboxes) != 0:
                for i in range(len(bboxes)):

                    # Check if score is above threshold
                    if scores[i] >= det_score_thres:

                        # Print deteced objects and scores on the terminal
                        try:
                            print (f'{(id2name_dict[class_id[i]]).strip()}, Score: {scores[i]}')
                        except:
                            print (f'Class name does not exist for label ID {class_id[i]}') 

                        # Create annotated image to visualize
                        frame_ori = visualize(frame_ori, 
                                              bboxes, 
                                              class_id, 
                                              scores, 
                                              det_score_thres, 
                                              id2name_dict)

            # Display the resulting frame
            cv.imshow('frame', frame_ori)

            # the 'q' button is set as the
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

            t2 = time.time()
            #print (f'frame_time: {t2-t1}')


        except Exception as e:
            print (e)
            # On error, release the camera object
            cam.stop()
            break

def main():
    start_camera(model_path=model_path, id2name_path=id2name_path, det_score_thres=det_score_thres)

if __name__ == '__main__':
    main()

