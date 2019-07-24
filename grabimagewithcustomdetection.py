import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import Dot

import tensorflow as tf
import time
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
import PIL.ImageGrab
import mss
import cv2
import Keyboard
import pyautogui

import mouse
import win32api,win32con
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

# This is needed to display the images.


from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

sct = mss.mss()
title = "Detection"
fps = 0
display_time = 2
start_time = time.time()

monitor = {"top": 80, "left": 0, "width": 840, "height": 640}
width = 800
height = 640




mouse1 = mouse.Mouse()



# What model to download.
MODEL_NAME = 'inference_graph'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = 'C://Users//Arthur//tensorflow//models//research//object_detection/inference_graph//frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = "C://Users//Arthur//PycharmProjects//AiFortnite//detection.pbtxt"
NUM_CLASSES = 5

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]

            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


"""printscreen_numpy = np.array(PIL.ImageGrab.grab(bbox=(0, 40, 800, 640)))

last_time = time.time()
print("loop take {} seconds".format(time.time() - last_time))
last_time = time.time()
new_screen = process_img(printscreen_numpy)
cap = cv2.imshow("winndowTest", printscreen_numpy)
"""

def move(x,y):
    x = int(mid_x*width)
    y = int(mid_y*height+height/9)
    print(x, y)
    Dot.dotrun(x, y)
    x1,y1 = win32api.GetCursorPos()
    x-=x1
    y-=y1
    Keyboard.MouseMoveTo(int(x/2),int(y/2))
    pyautogui.mouseUp(button="right")
    pyautogui.click(button="left")


   # if (x,y) != (400,321):
   # win32api.mouse_event(win32con.MOUSEEVENTF_MOVE |
                  #  win32con.MOUSEEVENTF_ABSOLUTE,int(x/width*65535.0),int(y/height*65535.0))
    #win32api.mouse_event(win32con.MOUSEEVENTF_MOVE |
                         #win32con.MOUSEEVENTF_ABSOLUTE, int(0), int(0))









def movement(boxes):
   if len([category_index.get(value) for index, value in enumerate(classes[0]) if scores[0, index] > 0.7]) != 0:
        try:
            print(pyautogui.position())
            Keyboard.MouseMoveTo(boxes[0][1]+boxes[0][3]/2,(boxes[0][2]+boxes[0][1])/2)


        except IndexError:
            print(" ")

   else:
       try:
           Keyboard.MouseMoveTo(-((boxes[0][1] + boxes[0][3]) / 2), -((boxes[0][2] + boxes[0][1]) / 2))



       except IndexError:
           print(" ")






# press = Keyboard.PressKey(Keyboard.W)
# else:
# Keyboard.ReleaseKey(Keyboard.W)

# click = Keyboard.PressKey(K)


with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:
            # Get raw pixels from the screen, save it to a Numpy array
            image_np = np.array(sct.grab(monitor))
            # To get real color we do this:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Visualization of the results of a detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=3)

            coordinates = vis_util.return_coordinates(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                min_score_thresh=0.90)
            win32api.SetCursorPos((383,321))
            #print(win32api.GetCursorPos())



            array_values =[]
            for i,b in enumerate(boxes[0]):
                #scores[0][i] >=0.6
                if scores[0][i] >=0.8:
                    mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                    mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                    array_values.append([mid_x,mid_y])

            if len(array_values)>0:
                move(array_values[0][0],array_values[0][1])



            # Show image with detection
            print(coordinates)
            cv2.imshow(title, cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
           # movement(coordinates)





            # Bellow we calculate our FPS
            fps += 1
            TIME = time.time() - start_time
            if (TIME) >= display_time:
                print("FPS: ", fps / (TIME))
                fps = 0
                start_time = time.time()
            # Press "q" to quit
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break
