#!/bin/bash

if [ $# -eq 0 ]; then
  DATA_DIR="./models"
  mkdir ${DATA_DIR}
else
  DATA_DIR="$1"
fi

# Install Python dependencies
# python3 -m pip install pip --upgrade
# python3 -m pip install -r requirements.txt

# Efficient Det Lite 0
# MODEL_PATH='https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/object_detection/rpi/lite-model_efficientdet_lite0_detection_metadata_1.tflite' 
# Efficient Det Lite 1
MODEL_PATH='https://tfhub.dev/tensorflow/lite-model/efficientdet/lite1/detection/default/1?lite-format=tflite'
# Efficient Det Lite 3
# MODEL_PATH='https://tfhub.dev/tensorflow/lite-model/efficientdet/lite3/detection/default/1?lite-format=tflite'
# Efficient Det Lite 4
# MODEL_PATH='https://tfhub.dev/tensorflow/lite-model/efficientdet/lite4/detection/default/2?lite-format=tflite'

# Mobilenet v2 RetinaNet
# MODEL_PATH='https://tfhub.dev/google/lite-model/qat/mobilenet_v2_retinanet_256/1?lite-format=tflite'
# Mobilenet v1
# MODEL_PATH='https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/default/1?lite-format=tflite'

# Download TF Lite models
# FILE=${DATA_DIR}/model_obj_detection.tflite
# if [ ! -f "$FILE" ]; then
#   curl \
#     -L ${MODEL_PATH} \
#     -o ${FILE}
# fi

# echo -e "Downloaded files are in ${DATA_DIR}"
