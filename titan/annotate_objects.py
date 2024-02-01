# Load Necessary Libraries

# General
from typing import List, Tuple

# Data Handling
import numpy as np

# Image Processing
import cv2

# Plotting
import matplotlib.pyplot as plt
from .visualizer import showAnns

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

  # Adapted from: https://github.com/cocodataset/cocoapi/issues/153#issuecomment-1833866223 to handle contours with holes using a trick
  # it connects the holes with the outer polygon so that it becomes a single polygon.
  def is_clockwise(self, contour):
    """
    Finds whether a given `contour` is clockwise or not
    """
    value = 0
    num = len(contour)
    for i, point in enumerate(contour):
        p1 = contour[i]
        if i < num - 1:
          p2 = contour[i + 1]
        else:
          p2 = contour[0]
        value += (p2[0][0] - p1[0][0]) * (p2[0][1] + p1[0][1]);
    return value < 0

  def get_merge_point_idx(self, contour1, contour2):
    """
    Helper function for `merge_with_parent`
    Finds the point (index) of merging of two contours.
    """
    idx1 = 0
    idx2 = 0
    distance_min = -1
    for i, p1 in enumerate(contour1):
      for j, p2 in enumerate(contour2):
        distance = pow(p2[0][0] - p1[0][0], 2) + pow(p2[0][1] - p1[0][1], 2);
        if distance_min < 0:
          distance_min = distance
          idx1 = i
          idx2 = j
        elif distance < distance_min:
          distance_min = distance
          idx1 = i
          idx2 = j
    return idx1, idx2

  def merge_contours(self, contour1, contour2, idx1, idx2):
    """
    Helper funtion for `merge_with_parent`.
    Merge two contours.
    """
    contour = []
    for i in list(range(0, idx1 + 1)):
      contour.append(contour1[i])
    for i in list(range(idx2, len(contour2))):
      contour.append(contour2[i])
    for i in list(range(0, idx2 + 1)):
      contour.append(contour2[i])
    for i in list(range(idx1, len(contour1))):
      contour.append(contour1[i])
    contour = np.array(contour)
    return contour

  def merge_with_parent(self, contour_parent, contour):
    """
    Helper function for `mask2polygon`.
    Merge the `contour` (hole contour) with the `contour_parent`.
    """
    if not self.is_clockwise(contour_parent):
      contour_parent = contour_parent[::-1]
    if self.is_clockwise(contour):
      contour = contour[::-1]
    idx1, idx2 = self.get_merge_point_idx(contour_parent, contour)
    return self.merge_contours(contour_parent, contour, idx1, idx2)

  def mask2polygon(self, binary_image):
    """
    Helper function for `wordheatmap_to_annotation`.
    Returns the contours present in an image. Handles holes as well by merging them with parent contour.
    - `binary_image`: The binary image for which polygons needs to be returned for further proessing.
    """
    contours, hierarchies = cv2.findContours(binary_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    contours_approx = []
    polygons = []
    for contour in contours:
      epsilon = 0.001 * cv2.arcLength(contour, True)
      contour_approx = cv2.approxPolyDP(contour, epsilon, True)
      contours_approx.append(contour_approx)

    contours_parent = []
    for i, contour in enumerate(contours_approx):
      parent_idx = hierarchies[0][i][3]
      if parent_idx < 0 and len(contour) >= 3:
        contours_parent.append(contour)
      else:
        contours_parent.append([])

    for i, contour in enumerate(contours_approx):
      parent_idx = hierarchies[0][i][3]
      if parent_idx >= 0 and len(contour) >= 3:
        contour_parent = contours_parent[parent_idx]
        if len(contour_parent) == 0:
          continue
        contours_parent[parent_idx] = self.merge_with_parent(contour_parent, contour)

    contours_parent_tmp = []
    for contour in contours_parent:
      if len(contour) == 0:
        continue
      contours_parent_tmp.append(contour)

    return tuple(contours_parent_tmp)


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


  def wordheatmap_to_annotations(
    self, 
    word_heatmap, 
    start_annotation_id: int = 1, 
    image_id: int = -1, 
    word_cat_id: int = -1, 
    use_nms=True, 
    skip_small_filters=False):
    """
    TODO: Change the name at heatmap_to_annotations!
    
    word_heatmap: word heatmap or simply heatmap as an array --> The heatmap pixels must be in 0 to 1 range
    image_id: if any (required for COCO dataset format) defaults to -1 meaning not provided
    word_cat_id: if any (the id of the current word) defaults to -1 meaning not provided
    use_nms: True by default, if False does not apply nms
    skip_small_filters: False by default, if True does not apply filtering of small bboxes or segments
    """

    # stores the annotations
    annotations = list()
    annotation_id = start_annotation_id

    # Casting heatmap from 0-1 floating range to 0-255 unsigned 8 bit integer
    heatmap = np.array(word_heatmap * 255, dtype = np.uint8)

    # Blur the heatmap for better thresholding
    blurred_heatmap = cv2.GaussianBlur(heatmap, self.blur_kernel_size, 0)

    # Binary threshold of the above heatmap - serves as sort of semantic segmentation for the word
    thresh = cv2.threshold(blurred_heatmap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Find contours from the binary threshold
    cnts = self.mask2polygon(thresh)
    # the following 2 lines can be used but it does not detect holes present within an object
    # cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    if len(cnts) == 0: # If no contours detected skip
      return None
    
    # Filtering contours based on their small area
    if not skip_small_filters:
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
    if not skip_small_filters:
        bbox_area_filter_threshold = np.max(bbox_areas) / self.pre_nms_small_box_thresh
        filtered_bboxes = [area >= bbox_area_filter_threshold for area in bbox_areas]
        bboxes = bboxes[filtered_bboxes, :]

        # If any bounding box is removed above its corresponding contour should also be removed
        cnts = [cnts[k] for k in range(len(cnts)) if filtered_bboxes[k]]

    # Merge multiple box predictions using Non-Max Suppression
    if use_nms:
        picks, picks2idx = self.non_max_suppression_fast(bboxes)
        picks2idx = {pick:list(set(idxs)) for pick, idxs in picks2idx.items()}
        picks = list(picks2idx.keys())
    else:
        picks, picks2idx = list(range(len(bboxes))), dict([(p, [p]) for p in range(len(bboxes))])
    
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
      if not skip_small_filters:
          for seg_idx, ar in enumerate(all_areas):
            if ar >= curr_pick_seg_small_filter:
              filtered_segments.append(segments[seg_idx])
      else:
          filtered_segments = segments

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
          'score': 0.0 # Doesn't actually matter much (If you aren't using this field later in some downstream task)
      }
      annotation_id += 1
      curr_word_annots.append(ann_det)
    
    # Finding discard threshold for small boxes for current category in current image (Between picks small box filtration)
    curr_word_ann_small_filter = np.max([ann['area'] for ann in curr_word_annots]) / self.small_box_thresh
    
    # Filtering small box annotations for current word/object
    for ann_det in curr_word_annots:
      if not skip_small_filters:
          if ann_det['area'] >= curr_word_ann_small_filter:
            annotations.append(ann_det)
      else:
          annotations.append(ann_det)

    return annotations


  def show_annotations(self, image_arr, anns: List, draw_bbox=True, figsize=(10,10)):
    """
    Visualizes the Annotations
    The image on which annotations needs to be shown needs to be plotted before this is called otherwise no effect can be seen
    image_arr: numpy array form of the image
    anns: list of annotations in COCO format only the annotations
    """
    plt.figure(figsize=figsize)
    plt.imshow(image_arr)
    plt.axis('off')
    showAnns(anns, draw_bbox)