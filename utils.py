# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility functions to display the pose detection results."""

import cv2

_MARGIN = 25  # pixels
_ROW_SIZE = 10  # pixels
_FONT_SIZE = 3
_FONT_THICKNESS = 3
_TEXT_COLOR = (0, 255, 0)  # green


def visualize(img, bboxes, class_id, scores, score_thres, label_dict):
  # Draws bounding boxes on the input image and return it.

  factor_w = img.shape[1]
  factor_h = img.shape[0]

  for i in range(len(bboxes)):

    if scores[i] >= score_thres:
      try:
        print (label_dict[class_id[i]])
      except:
        print (f'Class name does not exist for label ID {class_id[i]}') 
      
      # Draw bounding_box
      start_point = (int(bboxes[i][1]*factor_w), int(bboxes[i][0]*factor_h))
      end_point = int(bboxes[i][3]*factor_w ), int(bboxes[i][2]*factor_h)
      cv2.rectangle(img, start_point, end_point, _TEXT_COLOR, 2)

      # Draw label and score
      class_name = (label_dict[class_id[i]]).strip()
      score = str(round(scores[i], 2))[:4]
      result_text = f'{class_name}: {score}'
      text_location = (_MARGIN + int(bboxes[i][1]*factor_w),
                      _MARGIN + _ROW_SIZE + int(bboxes[i][0]*factor_h))
      cv2.putText(img, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                  _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)

  return img

def create_label_dict(label_file_path):
  # Create a dictionary that maps class ID to class name
  label_dict = {}
  try:
    with open(label_file_path) as f:
        i = 0
        for row in f:
            label_dict[i] = row
            i+=1
  except:
     print ('Error when reading label file')
  finally:
     return label_dict
