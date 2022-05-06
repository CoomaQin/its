# Dependencies
import os
import time
import cv2
import torch
from torchvision import models
import numpy as np
from numpy import random
from datetime import datetime, timezone, timedelta
from collections import deque
import math
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Modules
from my_utils.general import check_img_size, non_max_suppression, scale_coords, plot_one_box, set_logging
from my_utils.torch_utils import time_synchronized

# Deep Sort
from deep_sort.sort.detection import Detection
from deep_sort.sort.tracker import Tracker
from deep_sort.sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.deep.feature_extractor import Extractor

# YOLO Constants
YOLO_IMG_SIZE = 640
YOLO_CONF_THRES = 0.65
YOLO_IOU_THRES = 0.5

# Sort Constants
# N_INIT = 3          # Consecutive frames need to confirm object
# MAX_AGE = 150          # Max tracking window
# MAX_IOU_DIST= 0.3   # IOU Threshold

N_INIT = 3         # Consecutive frames need to confirm object
MAX_AGE = 150          # Max tracking window
MAX_IOU_DIST = 3  # IOU Threshold
MAX_DIST = 10
NN_BUDGET = 100

# Retrained with only 8 labels
# person, bicycle, car, motorcycle, bus, train, truck, traffic light
OBJECT_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7]

# Colors
PERSON_COLOR = (255, 0, 0)
VEHICLE_COLOR = (102, 102, 0)
TRAIN_COLOR = (255, 0, 0)
OTHER_COLOR = (160, 160, 160)
TRESPASS_COLOR = (0, 0, 255)
COLORS = np.random.randint(0, 255, (3000, 3))

label_mappings = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "bus",
    5: "train",
    6: "truck",
    7: "traffic light"
}


def load_model(weight_file):
  model = torch.hub.load('ultralytics/yolov5', 'custom', weight_file)
  return model


def draw_bboxes(
    orig_img,
    objects,
    classnames,
    colors=None
  ):
    if colors is None:
      colors = [[random.randint(0, 255) for _ in range(3)]
                                for _ in range(len(classnames))]
    for *xyxy, conf, cls in reversed(objects):
      label = '%s %.2f' % (classnames[int(cls)], conf)
      plot_one_box(xyxy, orig_img, label=label,
                   color=colors[int(cls)], line_thickness=1)


def draw_one_bbox(
    orig_img,
    object,
    label,
    color
  ):
    *xyxy, conf, cls = object
    plot_one_box(xyxy, orig_img, label=label, color=color, line_thickness=2)


def draw_one_bbox_raw(
    orig_img,
    bbox,
    label,
    color
  ):
    plot_one_box(bbox, orig_img, label=label, color=color, line_thickness=2)


def ROI_points_to_contour(
    ROI_points
  ):
    contour = []
    for point in ROI_points:
      contour.append([int(point['x']), int(point['y'])])
    contour = np.array(contour)
    return contour


def draw_ROI(
    orig_img,
    contour
  ):
    cv2.drawContours(orig_img, [contour], -1, (0, 255, 0), 3)


def draw(orig_img, track):
  tlbr = track.to_tlbr()
  track_id = track.track_id
  cls_id = track.latest_detection_oid
  cv2.rectangle(orig_img, (int(tlbr[0]), int(tlbr[1])), (int(
      tlbr[2]), int(tlbr[3])), tuple(COLORS[int(track_id)].tolist()), 2)
  cv2.putText(orig_img, f"{track_id} {label_mappings[int(cls_id)]}", (int(tlbr[0]), int(
      tlbr[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, tuple(COLORS[int(track_id)].tolist()), 2)


def format_track(track):
  # output [trackid, clsid, x1, y1, x2, y2]
  ans = [track.track_id, track.latest_detection_oid] 
  ans.extend(track.to_tlbr())
  return ans



def is_in_ROI(
    obj,
    contour,
    is_ROW=False
  ):
    *xyxy, _, _ = obj
    xyxy = [int(i) for i in xyxy]
    x1, y1, x2, y2 = xyxy
    centerX = int((x1 + x2) / 2)

    if is_ROW:  # For ROW, take center of object
      bottomY = int((y1 + y2) / 2)
    else:  # Else take bottom center of object
      bottomY = max(y1, y2)

    retval = cv2.pointPolygonTest(contour, (centerX, bottomY), False)
    is_in = True if retval >= 0 else False
    return is_in  # (is_in, centerX, bottomY)


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
  # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
  shape = img.shape[:2]  # current shape [height, width]
  if isinstance(new_shape, int):
    new_shape = (new_shape, new_shape)

  # Scale ratio (new / old)
  r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
  if not scaleup:  # only scale down, do not scale up (for better test mAP)
    r = min(r, 1.0)

  # Compute padding
  ratio = r, r  # width, height ratios
  new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
  dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
      new_unpad[1]  # wh padding
  if auto:  # minimum rectangle
    dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
  elif scaleFill:  # stretch
    dw, dh = 0.0, 0.0
    new_unpad = (new_shape[1], new_shape[0])
    ratio = new_shape[1] / shape[1], new_shape[0] / \
        shape[0]  # width, height ratios

  dw /= 2  # divide padding into 2 sides
  dh /= 2

  if shape[::-1] != new_unpad:  # resize
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
  top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
  left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
  img = cv2.copyMakeBorder(img, top, bottom, left, right,
                           cv2.BORDER_CONSTANT, value=color)  # add border
  return img, ratio, (dw, dh)


def preprocess_img_for_resnet(orig_img):
  img = orig_img / 255.0
  img = cv2.resize(img, (224, 224))
  img = img[:, :, ::-1].transpose(2, 0, 1)
  img = np.ascontiguousarray(img)
  return img


def preprocess_img_for_yolo(orig_img):
  img = letterbox(orig_img, new_shape=(640, 640))[0]
  img = img[:, :, ::-1].transpose(2, 0, 1)
  img = np.ascontiguousarray(img)
  return img


def create_Detections(orig_img, yolo_detections, feature_model, use_resnet=False):
  Detections = []

  for obj in yolo_detections:
    obj_internalized = obj
    tlwh = (obj_internalized[0], obj_internalized[1], obj_internalized[2] -
            obj_internalized[0], obj_internalized[3] - obj_internalized[1])
    confidence = obj_internalized[4]

    # Get features
    patch = orig_img[int(obj_internalized[1]):int(obj_internalized[3]), int(obj_internalized[0]):int(obj_internalized[2]),
                         :] if use_resnet else orig_img[int(obj_internalized[1]):int(obj_internalized[3]), int(obj_internalized[0]):int(obj_internalized[2])]
    if use_resnet:
      img_for_resnet = preprocess_img_for_resnet(patch)
      img_for_resnet = torch.from_numpy(img_for_resnet).cuda().half()
      img_for_resnet = img_for_resnet.unsqueeze(0)
      features = feature_model(img_for_resnet)
      features = features.squeeze()
      features = features.cpu().numpy()
    else:
      features = feature_model([patch])[0]

    detection = Detection(tlwh, confidence, features, oid=obj[5])

    Detections.append(detection)

  return Detections


# main loop
if __name__ == '__main__':

  wname = "m5_500"

  # # built-in resnet model
  # resnet = models.resnet18(pretrained=True)
  # resnet.cuda().half()
  # resnet.eval()
  # # Extract all layers but last to form a new model which last layer is the avgpool layer that has 512 outputs of feature vector
  # modules = list(resnet.children())[:-1]
  # resnet_features = torch.nn.Sequential(*modules)

  # customized model
  extractor = Extractor("deep_sort/deep/checkpoint/ckpt.t7", use_cuda=True)

  # YOLO model
  yolo_model = load_model(f'./weights/{wname}.pt')
  yolo_model.conf = 0.5

  # Deep Sort Tracker
  metric = NearestNeighborDistanceMetric('cosine', MAX_DIST, budget=NN_BUDGET)
  tracker = Tracker(metric, n_init=N_INIT, max_age=MAX_AGE,
                    max_iou_distance=MAX_IOU_DIST)

  for vname in ["inward2.mp4"]:
    output_path = "./outputs/"
    video_url = f"./videos/{vname}"
    video = cv2.VideoCapture(video_url)

    CODEC_fourcc = "mp4v"
    fps = video.get(cv2.CAP_PROP_FPS)
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output = cv2.VideoWriter(
        f'outputs/{vname}-{wname}.mp4', cv2.VideoWriter_fourcc(*CODEC_fourcc), 30, (w, h))

    with torch.no_grad():
      # whole screen
      zone_points = [{
              "x": 1,
              "y": 1
          }, {
              "x": w - 1,
              "y": 1
          }, {
              "x": w - 1,
              "y": h - 1
          }, {
              "x": 1,
              "y": h - 1
      }]

      contour = ROI_points_to_contour(zone_points)
      idx = 0
      info = []
      while True:
        ret, frame = video.read()
        if ret == True:
          t0 = time.time()
          yolo_results = yolo_model(frame)
          # yolo_results.print()

          yolo_results = yolo_results.xyxy[0].tolist() if len(yolo_results.xyxy[0]) > 0 else [] # yolo_results.xyxy has len 1 even if no detection
          # Only keep detections that are in the ROI
          in_roi_detections = []
          
          for obj in yolo_results:
            if int(obj[5]) != 7:
              in_roi_detections.append(obj)

          Detections = create_Detections(frame, in_roi_detections, extractor)

          tracker.predict()
          tracker.update(Detections)

          t1 = time.time()
          # curr_info = []
          for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
              continue
            
            draw(frame, track)
            # curr_info.append(format_track(track))
          # info.append(curr_info)

          output.write(frame)
        
        else:
          # info = np.array(info)
          # np.save("./od_info", info)
          output.release()
          break
