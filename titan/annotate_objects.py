# Load Necessary Libraries

# General
from typing import List, Tuple

# Data Handling
import numpy as np

# Image Processing
import cv2

# Model
import torch

# Default values to be used if nothing else is specified
DEVICE = 'cuda' # device
NUM_IMAGES_PER_PROMPT = 4 # Number of images to be generated per prompt
STRENGTH = 0.3 # The noise to add to original image
NUM_INFERENCE_STEPS = 50 # Number of inference steps to the Diffusion Model
NAME_OF_DATASET = 'COCO Stable Diffusion 2 Dataset' # Name of the generated dataset
SAVE_AFTER_NUM_IMAGES = 1 # Number of images after which the annotation and caption files will be saved
NMS_OVERLAP_THRESHOLD = 0.5 # Non-Max Suppression Threshold
BLUR_KERNEL_SIZE = (5, 5) # The size of the kernel to be used for blurring the heatmap before binary thresholding
PRE_NMS_SMALL_SEGMENT_THRESH = 30 # For filtering small segments, higher the threshold more smaller contours will be allowed (before applying NMS)
PRE_NMS_SMALL_BOX_THRESH = 30 # For filtering small bboxes, higher the threshold more smaller boxes will be allowed (before applying NMS)
SMALL_SEGMENT_THRESH = 30 # For filtering small segments, higher the threshold more smaller segments will be allowed (Within a bbox operation for each category in an image)
SMALL_BOX_THRESH = 30 # For filtering small boxes, higher the threshold more small boxes will be allowed (Between bboxes operation for each category in an image)

class ObjectAnnotator:

  def __init__(
    self,
    nms_overlap_threshold: float = NMS_OVERLAP_THRESHOLD,
    blur_kernel_size: Tuple[int, int] = BLUR_KERNEL_SIZE,
    pre_nms_small_segment_thresh: int = PRE_NMS_SMALL_SEGMENT_THRESH,
    pre_nms_small_box_thresh: int = PRE_NMS_SMALL_BOX_THRESH,
    small_segment_thresh: int = SMALL_SEGMENT_THRESH,
    small_box_thresh: int = SMALL_BOX_THRESH
    ):

    self.nms_overlap_threshold = nms_overlap_threshold
    self.blur_kernel_size = blur_kernel_size
    self.pre_nms_small_segment_thresh = pre_nms_small_segment_thresh
    self.pre_nms_small_box_thresh = pre_nms_small_box_thresh
    self.small_segment_thresh = small_segment_thresh
    self.small_box_thresh = small_box_thresh


  def non_max_suppression_fast(self, boxes):
    """
    Helper Function that performs Non-Max Suppression when given the bounding boxes.
    We merge multiple boxes together using Non-Max Suppression with a modification. Since, each box is corresponding to a segment. Now, when we delete boxes, we keep track 
    which box it is merged to. Now, if we use the boundary of this box as a boundary for the object there would be segments spreading out of the box. So, instead I enlarge the 
    box to the extreme segment which merged to this box by NMS 
    Greatly reduces too many bounding boxes.
    """

    # if there are no boxes, return an empty list
    if len(boxes) == 0:
      return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
      boxes = boxes.astype("float")
    # initialize the list of picked indexes 
    pick = []
    pick2idx = dict()
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
      # grab the last index in the indexes list and add the
      # index value to the list of picked indexes
      last = len(idxs) - 1
      i = idxs[last]
      pick.append(i)
      # find the largest (x, y) coordinates for the start of
      # the bounding box and the smallest (x, y) coordinates
      # for the end of the bounding box
      xx1 = np.maximum(x1[i], x1[idxs[:last]])
      yy1 = np.maximum(y1[i], y1[idxs[:last]])
      xx2 = np.minimum(x2[i], x2[idxs[:last]])
      yy2 = np.minimum(y2[i], y2[idxs[:last]])
      # compute the width and height of the intersection bounding box
      w = np.maximum(0, xx2 - xx1 + 1)
      h = np.maximum(0, yy2 - yy1 + 1)
      # compute the ratio of overlap
      overlap = (w * h) / area[idxs[:last]]
      # delete all indexes from the index list that have
      to_delete = np.concatenate(([last], np.where(overlap >= self.nms_overlap_threshold)[0]))
      # the boxes to be deleted are mapped to a box so the box to which it is mapped
      # should also be in the dictionary else it will be omitted all together
      pick2idx[i] = [idxs[id] for id in list(to_delete)] + [i]
      # removing the bounding boxes and moving towards doing NMS on remaining ones
      idxs = np.delete(idxs, to_delete)

    # return only the bounding box idx that were picked
    return pick, pick2idx


  def wordheatmap_to_annotations(self, word_heatmap, start_annotation_id: int = 1, image_id: int = -1, word_cat_id: int = -1):
    """
    heat_map: daam WordHeatMap
    image_id: if any (required for COCO dataset format) defaults to -1 meaning not provided
    word_cat_id: if any (the id of the current word) defaults to -1 meaning not provided
    """

    # stores the annotations
    annotations = list()
    annotation_id = start_annotation_id

    # Casting heatmap from 0-1 floating range to 0-255 unsigned 8 bit integer
    heatmap = np.array(word_heatmap * 255, dtype = np.uint8)

    # Blur the heatmap for better thresholding
    blurred_heatmap = cv2.GaussianBlur(heatmap, BLUR_KERNEL_SIZE, 0)

    # Binary threshold of the above heatmap - serves as sort of semantic segmentation for the word
    thresh = cv2.threshold(blurred_heatmap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Find contours from the binary threshold
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    if len(cnts) == 0: # If no contours detected skip
      return None
    
    # Filtering contours based on their small area
    cnts_areas = [cv2.contourArea(cnt) for cnt in cnts]
    cnts_area_filter_threshold = np.max(cnts_areas) / self.pre_nms_small_segment_thresh
    filtered_cnts = [area >= cnts_area_filter_threshold for area in cnts_areas]
    cnts = [cnts[k] for k in range(len(cnts)) if filtered_cnts[k]]

    # Find bounding boxes from contours
    bboxes = np.zeros((len(cnts), 4))
    bbox_areas = list()
    for idx, c in enumerate(cnts):
      x,y,w,h = cv2.boundingRect(c)
      bbox_areas.append(w*h)
      bboxes[idx, :] = np.array([x,y, x+w, y+h])
    
    # Filtering bboxes based on their small area
    bbox_area_filter_threshold = np.max(bbox_areas) / self.pre_nms_small_box_thresh
    filtered_bboxes = [area >= bbox_area_filter_threshold for area in bbox_areas]
    bboxes = bboxes[filtered_bboxes, :]

    # Merge multiple box predictions using Non-Max Suppression
    picks, picks2idx = self.non_max_suppression_fast(bboxes)
    picks2idx = {pick:list(set(idxs)) for pick, idxs in picks2idx.items()}
    picks = list(picks2idx.keys())
    
    # stores filtered out boxes i.e. small boxes removed
    curr_word_annots = list()

    # Annotating the segmentation and the bounding boxes
    for pick in picks:
      # All segments in current pick
      segments = [list(cnts[k].squeeze().reshape(1, -1).squeeze()) for k in picks2idx[pick]]

      # Area of each segment in segments
      all_areas = [cv2.contourArea(cnts[k]) for k in picks2idx[pick]]

      # Finding discard threshold for small segments for current category in current image (Within pick small segment filtration)
      curr_pick_seg_small_filter = np.max(all_areas) / self.small_segment_thresh
      
      # stores the filtered out segments
      filtered_segments = list()

      # Filtering small segments in current pick for current word/object
      for seg_idx, ar in enumerate(all_areas):
        if ar >= curr_pick_seg_small_filter:
          filtered_segments.append(segments[seg_idx])

      # The area inside one annotation is sum of the area of all the segments that form it
      area = np.sum(all_areas)

      # Finding bounding box location and dimensions based on filtered segments
      x_segments = [x for segment in filtered_segments for x in segment[::2]]
      y_segments = [y for segment in filtered_segments for y in segment[1::2]]
      x = min(x_segments)
      y = min(y_segments)
      w = max(x_segments) - x
      h = max(y_segments) - y

      ann_det = { # Annotation details
          'segmentation': filtered_segments,
          'area': area,
          'iscrowd': 0,
          'image_id': image_id,
          'bbox': [x, y, w, h],
          'category_id': word_cat_id,
          'id': annotation_id,
      }
      annotation_id += 1
      curr_word_annots.append(ann_det)
    
    # Finding discard threshold for small boxes for current category in current image (Between picks small box filtration)
    curr_word_ann_small_filter = np.max([ann['area'] for ann in curr_word_annots]) / self.small_box_thresh
    
    # Filtering small box annotations for current word/object
    for ann_det in curr_word_annots:
      if ann_det['area'] >= curr_word_ann_small_filter:
        annotations.append(ann_det)

    return annotations

