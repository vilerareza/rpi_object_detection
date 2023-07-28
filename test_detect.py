import cv2 as cv
import numpy as np
import tflite_runtime.interpreter as tflite
from utils import visualize, create_label_dict

# Path to tflite model
model_path = 'model_obj_detection.tflite'
#model_path = 'lite-model_efficientdet_lite0_detection_metadata_1.tflite'
# Path to test image
test_image_path = 'test_data/test_img_1.jpg'
# Path to test image annotated with prediction bounding box
test_image_bbox_path = 'test_data/test_img_1_bbox.jpg'
# Score threshold
score_threshold = 0.2
# Path to label txt
label_path = 'labelmap_new.txt'


def run(model, test_img_path, label_dict) -> None:

    # Initialize the object detector
    interpreter = tflite.Interpreter(model)
    interpreter.allocate_tensors()
    interpreter_output = interpreter.get_output_details()
    interpreter_input = interpreter.get_input_details()[0]

    # Reading image from file
    img_ori = cv.imread(test_img_path, 1)
    img = img_ori.copy()

    # Convert BGR to RGB
    img = img[:,:,::-1]
    # The EfficientDet model require the input size to be (320 x 320) 
    img = cv.resize(img,(384,384))
    img = np.expand_dims(img, axis=0)
    # Run object detection estimation using the model.
    interpreter.set_tensor(interpreter_input['index'], img)
    interpreter.invoke()
    bboxes = interpreter.get_tensor(interpreter_output[0]['index'])[0]
    class_id = interpreter.get_tensor(interpreter_output[1]['index'])[0]
    scores = interpreter.get_tensor(interpreter_output[2]['index'])[0]

    img_ann = visualize(img_ori, bboxes, class_id, scores, score_threshold, label_dict)
    cv.imwrite(test_image_bbox_path, img_ann)


def main():

    # Creating label dictionary for class name
    label_dict = create_label_dict(label_path)
    # Run inference
    run(model_path, test_image_path, label_dict)


if __name__ == '__main__':
    main()
