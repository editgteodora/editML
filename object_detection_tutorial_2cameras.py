#!/usr/bin/env python
# coding: utf-8

# # Object Detection Demo
# Welcome to the object detection inference walkthrough!  This notebook will walk you step by step through the process of using a pre-trained model to detect objects in an image. Make sure to follow the [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) before you start.

# # Imports


from itertools import combinations
import itertools

import time
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import flask
from threading import Thread
import datetime

from time import sleep
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from flask import Flask
from flask import jsonify
from flask import request

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')


# ## Env setup





# ## Object detection imports
# Here are the imports from the object detection module.



from utils import label_map_util

from utils import visualization_utils as vis_util


# # Model preparation 

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_FROZEN_GRAPH` to point to a new .pb file.  
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.




# What model to download.
# MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
# MODEL_FILE = MODEL_NAME + '.tar.gz'
# DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
#inference_graph-ssd_mobilenet_v1_fpn_coco-25000
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = '/home/ucenik/Documents/TensorFlow/models/research/object_detection/data/inference_graph-ssd_inception_v2_coco-147037/frozen_inference_graph.pb'
# PATH_TO_FROZEN_GRAPH = '/home/ucenik/Documents/inference_graph-ssdlite-20190719T081050Z-001/inference_graph-ssdlite/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'numbers.pbtxt')


# ## Download Model# In[ ]:


# opener = urllib.request.URLopener()
# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
# tar_file = tarfile.open(MODEL_FILE)
# for file in tar_file.getmembers():
#   file_name = os.path.basename(file.name)
#   if 'frozen_inference_graph.pb' in file_name:
#     tar_file.extract(file, os.getcwd())


# ## Load a (frozen) Tensorflow model into memory.

last_recognized_time = datetime.datetime.now()

def start_server_thread():

  app = Flask(__name__)

  @app.route('/', methods = ['GET'])
  def server():
    response = flask.jsonify(ID=last_recognized, DateTime=last_recognized_time)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response
    
  if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9999)

server_thread = Thread(target=start_server_thread)
server_thread.start()

def count(list, element):
  c = 0
  for i in range(len(list)):
    checking = list[i][0]
    if(checking == element):
      c+=1
  return c

def most_frequent(list1): 
    counter = 0
    num = list1[0] 
      
    for i in list1: 
        curr_frequency = count(list1,i) 
        if(curr_frequency > counter): 
            counter = curr_frequency 
            num = i 
  
    return num 

# def post_to_server(container_id):
#   current_container_id = jsonify(id=container_id,health_status="healthy")

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine


category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


# ## Helper code



def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# # Detection




# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)





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
            detection_masks, detection_boxes, image.shape[1], image.shape[2])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: image})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.int64)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict


# caps = [cv2.VideoCapture(4),cv2.VideoCapture(2)]
cap_indexes = [2, 4]

from webcam_threaded import WebcamVideoStream

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    recognizedTriplets = []
    
    last_recognized = ()
    cameras = [WebcamVideoStream(src=i) for i in cap_indexes]
    #cameras = [cv2.VideoCapture(i) for i in cap_indexes]
    # for cap in cameras:
    #     cap.set(cv2.CAP_PROP_FPS, 20)
    import time
    for cap in cameras:
        cap.start()
    #cameras[0].start()
    #cameras[1].start()
    #cameras[2].start()
    
    while True:
        # theader 1 = thread.run(detect, camera)
        # theader 2 = thread.run(detect, camera)
        # thread1.join()
        # thread2.join()

        for i, cap in enumerate(cameras):
            pocelo = time.time()
            ret, image_np = cap.read()
            check_receive = time.time()
            print('zavrsilo {} {}'.format(i, check_receive - pocelo))
            
            
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      #print(boxes)
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')

            classes = detection_graph.get_tensor_by_name('detection_classes:0')

            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

      # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            key_boxes = []
            for x in range(len(boxes[0])):
                if scores[0][x] > 0.15:#POKUSATI SMANJITI TODO
                    key_boxes.append((boxes[0][x], scores[0][x], classes[0][x], num_detections[0]))
      
      # if len(key_boxes) == 3:
      #   key_boxes.sort(key=lambda x:x[0][1], reverse=False)
      #   container_id = str(int(key_boxes[0][2])) + str(int(key_boxes[1][2])) + str(int(key_boxes[2][2]))
      #   print(container_id)

            key_boxes.sort(key=lambda x:x[0][1], reverse=False)
      


      #parametri delta1
            deltaDistance = 0.05
            deltaTriangle = 0.005
        
            triplets = []
            for triplet in combinations(key_boxes, 3):
        #TODO definisati dozvoljenu razliku za udaljenost i provjeriti je li razlika medju tackama unutar trojke manja od nje
                isDist = (abs(abs(triplet[0][0][1] - triplet[1][0][1]) - abs(triplet[1][0][1] - triplet[2][0][1]))) <= deltaDistance

        #print(abs(abs(triplet[0][0][1] - triplet[1][0][1]) - abs(triplet[1][0][1] - triplet[2][0][1])))

        #TODO definisati dozvoljenu povrsinu, naci povrsinu trougla za tri tacke i provjeriti je li manja od dozvoljene
                area = abs(triplet[0][0][1]*(triplet[1][0][0]-triplet[2][0][0])+triplet[1][0][1]*(triplet[2][0][0] - triplet[0][0][0])+triplet[2][0][1]*(triplet[0][0][0] - triplet[1][0][0]))/2

        #print(abs(triplet[0][0][1]*(triplet[1][0][0]-triplet[2][0][0])+triplet[1][0][1]*(triplet[2][0][0] - triplet[0][0][0])+triplet[2][0][1]*(triplet[0][0][0] - triplet[1][0][0]))/2)
                isArea = area <= deltaTriangle
        
                if(isArea and isDist):
                    recognizedTriplets.append(triplet)
          #container_id = str(int(triplet[0][2])) + str(int(triplet[1][2])) + str(int(triplet[2][2]))
          #print(container_id)
          
  ## end

      # #TODO pokusati izdvojiti najcescu trojku koja zadovoljava uslove
            mostAppeared = ()
            if( len(recognizedTriplets) > 3):
                mostAppeared = most_frequent(recognizedTriplets)
                container_id = str(int(mostAppeared[0][2])) + str(int(mostAppeared[1][2])) + str(int(mostAppeared[2][2]))
                if(last_recognized!=container_id):
                    last_recognized = container_id
                    last_recognized_time = datetime.datetime.now()
                recognizedTriplets = []
          
      

      #if(correctTriplets):
      #print(correctTriplets[0][0][1])

      # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=1,
                min_score_thresh=.15)

            cv2.imshow('object detection ' + str(i + 1), cv2.resize(image_np, (800,640)))

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

#800,640


