# Car_detection_using_YOLO
Car Detection using YOLO algorithm and implementing basic deep learning convolutional networks and applying optimizing algorithms such as Non-Max Suppression.

import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
from PIL import ImageFont, ImageDraw, Image
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor

from tensorflow.keras.models import load_model
from yad2k.models.keras_yolo import yolo_head
from yad2k.utils.utils import draw_boxes, get_colors_for_classes, scale_boxes, read_classes, read_anchors, preprocess_image

get_ipython().run_line_magic('matplotlib', 'inline')

def yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold = .6):
    box_scores = box_confidence*box_class_probs
    box_classes = tf.math.argmax(box_scores, axis=-1)
    box_class_scores = tf.math.reduce_max(box_scores, axis =-1)
    filtering_mask = box_class_scores >= threshold
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)
    return scores, boxes, classes


tf.random.set_seed(10)
box_confidence = tf.random.normal([19, 19, 5, 1], mean=1, stddev=4, seed = 1)
boxes = tf.random.normal([19, 19, 5, 4], mean=1, stddev=4, seed = 1)
box_class_probs = tf.random.normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1)
scores, boxes, classes = yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold = 0.5)

print("scores[2] = " + str(scores[2].numpy()))
print("boxes[2] = " + str(boxes[2].numpy()))
print("classes[2] = " + str(classes[2].numpy()))
print("scores.shape = " + str(scores.shape))
print("boxes.shape = " + str(boxes.shape))
print("classes.shape = " + str(classes.shape))


def iou(box1, box2):
    (box1_x1, box1_y1, box1_x2, box1_y2) = box1
    (box2_x1, box2_y1, box2_x2, box2_y2) = box2
    xi1 = max(box1_x1,box2_x1)
    yi1 = max(box1_y1,box2_y1)
    xi2 = min(box1_x2,box2_x2)
    yi2 = min(box1_y2,box2_y2)
    inter_width = xi2 - xi1
    inter_height =  yi2 - yi1
    inter_area = max(inter_width,0)*max(inter_height,0)
    box1_area = (box1_x2 - box1_x1)*(box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1)*(box2_y2 - box2_y1)
    union_area = box1_area + box2_area - inter_area
    iou = inter_area/union_area
    return iou


# BEGIN UNIT TEST

box1 = (2, 1, 4, 3)
box2 = (1, 2, 3, 4)
print("iou for intersecting boxes = " + str(iou(box1, box2)))

box1 = (1,2,3,4)
box2 = (5,6,7,8)
print("iou for non-intersecting boxes = " + str(iou(box1,box2)))

box1 = (1,1,2,2)
box2 = (2,2,3,3)
print("iou for boxes that only touch at vertices = " + str(iou(box1,box2)))

box1 = (1,1,3,3)
box2 = (2,3,3,4)
print("iou for boxes that only touch at edges = " + str(iou(box1,box2)))


def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):
    boxes = tf.cast(boxes, dtype=tf.float32)
    scores = tf.cast(scores, dtype=tf.float32)

    nms_indices = []
    classes_labels = tf.unique(classes)[0]
    
    for label in classes_labels:
        filtering_mask = classes == label
        boxes_label = tf.boolean_mask(boxes, filtering_mask)
        scores_label = tf.boolean_mask(scores, filtering_mask)
        
        if tf.shape(scores_label)[0] > 0:  
            nms_indices_label = tf.image.non_max_suppression(
                    boxes_label,
                    scores_label,
                    max_boxes,
                    iou_threshold=iou_threshold) 
            selected_indices = tf.squeeze(tf.where(filtering_mask), axis=1)
            nms_indices.append(tf.gather(selected_indices, nms_indices_label))

    # Flatten the list of indices and concatenate
    nms_indices = tf.concat(nms_indices,axis=0)
    scores = tf.gather(scores, nms_indices)
    boxes = tf.gather(boxes, nms_indices)
    classes = tf.gather(classes, nms_indices)
    sort_order = tf.argsort(scores, direction='DESCENDING').numpy()
    scores = tf.gather(scores, sort_order[0:max_boxes])
    boxes = tf.gather(boxes, sort_order[0:max_boxes])
    classes = tf.gather(classes, sort_order[0:max_boxes])

    return scores, boxes, classes



scores = np.array([0.855, 0.828])
boxes = np.array([[0.45, 0.2,  1.01, 2.6], [0.42, 0.15, 1.7, 1.01]])
classes = np.array([0, 1])

print(f"iou:    \t{iou(boxes[0], boxes[1])}")

scores2, boxes2, classes2 = yolo_non_max_suppression(scores, boxes, classes, iou_threshold = 0.9)

scores2, boxes2, classes2 = yolo_non_max_suppression(scores, boxes, classes, iou_threshold = 0.1)

classes = np.array([0, 0])

scores2, boxes2, classes2 = yolo_non_max_suppression(scores, boxes, classes, iou_threshold = 0.15)


print(f"scores:  \t{scores2.numpy()}")
print(f"boxes:  \t{boxes2.numpy()}")     
print(f"classes:\t{classes2.numpy()}")


scores2, boxes2, classes2 = yolo_non_max_suppression(scores, boxes, [0, 0], iou_threshold = 0.9)

from unit_tests import test_yolo_non_max_suppression

test_yolo_non_max_suppression(yolo_non_max_suppression)

def yolo_boxes_to_corners(box_xy, box_wh):
    """Convert YOLO box predictions to bounding box corners."""
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    return tf.keras.backend.concatenate([
        box_mins[..., 1:2],  # y_min
        box_mins[..., 0:1],  # x_min
        box_maxes[..., 1:2],  # y_max
        box_maxes[..., 0:1]  # x_max
    ])



def yolo_eval(yolo_outputs, image_shape = (720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    box_xy, box_wh, box_confidence, box_class_probs = yolo_outputs
    boxes = yolo_boxes_to_corners(box_xy, box_wh)
    scores, boxes, classes = yolo_filter_boxes(boxes, box_confidence, box_class_probs, score_threshold )
    boxes = scale_boxes(boxes, image_shape)
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)
    return scores, boxes, classes


# BEGIN UNIT TEST
tf.random.set_seed(10)
yolo_outputs = (tf.random.normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),
                tf.random.normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),
                tf.random.normal([19, 19, 5, 1], mean=1, stddev=4, seed = 1),
                tf.random.normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1))
scores, boxes, classes = yolo_eval(yolo_outputs)
print("scores[2] = " + str(scores[2].numpy()))
print("boxes[2] = " + str(boxes[2].numpy()))
print("classes[2] = " + str(classes[2].numpy()))
print("scores.shape = " + str(scores.numpy().shape))
print("boxes.shape = " + str(boxes.numpy().shape))
print("classes.shape = " + str(classes.numpy().shape))

class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
model_image_size = (608, 608) # Same as yolo_model input layer size
yolo_model = load_model("model_data/", compile=False)
yolo_model.summary()

def predict(image_file):
    image, image_data = preprocess_image("images/" + image_file, model_image_size = (608, 608))
    yolo_model_outputs = yolo_model(image_data)
    yolo_outputs = yolo_head(yolo_model_outputs, anchors, len(class_names))
    out_scores, out_boxes, out_classes = yolo_eval(yolo_outputs, [image.size[1],  image.size[0]], 10, 0.3, 0.5)
    print('Found {} boxes for {}'.format(len(out_boxes), "images/" + image_file))
    colors = get_colors_for_classes(len(class_names))
    draw_boxes(image, out_boxes, out_classes, class_names, out_scores)
    image.save(os.path.join("out", image_file), quality=100)
    output_image = Image.open(os.path.join("out", image_file))
    imshow(output_image)
    return out_scores, out_boxes, out_classes
