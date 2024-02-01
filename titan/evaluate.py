# Load Necessary Libraries
# Data Handling
import pandas as pd

# Matrix Manipulation
import numpy as np

def evaluate_mIoU(cocoGt, cocoDt, mode='segm', return_per_class_iou = False):
  """
  Evaluates the mIoU over a dataset for specified `mode` (defaults to segmentation).
  For each class, it finds the TP, FP, and FN over all the images. Then finds the IoU for each class
  using IoU_c = TP / (FP + TP + FN) and then averages over the IoU_c for all c to get the mIoU.
  - cocoGt: The ground truth Coco object. (instance of the COCO class in pycocotools)
  - cocoDt: The predicted Coco object. (instance of the COCO class in pycocotools)
  - mode: [defaults to `segm`] set it to either `segm` or `bbox`.
  - return_per_class_iou: [defaults to `False`] If `True` returns IoU for each class o.w. mean IoU
  Note: This function must be run only after making sure that each (img_id, cat_id) pair has only one ann_id. 
  If not, run `titan.utils.merge_img_cat_id`.
  """

  # Add a check for whether the image ids and cat ids for both the ground truth and detection match or not.

  # Check if mode is correct or not
  assert mode == 'segm' or mode == 'bbox'

  # Check if the id to category name mapping is same in both
  assert {cat['id']: cat['name'] for cat in cocoDt.cats.values()} == {cat['id']: cat['name'] for cat in cocoGt.cats.values()}

  # Check if the id to image name mapping is same in both
  assert {img['id']: img['file_name'] for img in cocoDt.imgs.values()} == {img['id']: img['file_name'] for img in cocoGt.imgs.values()}

  # Gets all the unique image ids and category ids
  img_ids = sorted(set([img['id'] for img in cocoGt.imgs.values()]))
  cat_ids = sorted(set([cat['id'] for cat in cocoGt.cats.values()]))

  # Create dataframe for easier handling and keeping track of TP, FP and FN
  tp_data = pd.DataFrame(columns=cat_ids, index=img_ids)
  fp_data = tp_data.copy(True)
  fn_data = tp_data.copy(True)

  # iterating over all the img_id and cat_id pairs
  for img_id in img_ids:
    for cat_id in cat_ids:

      # get the annotation id corresponding to the img_id and cat_id
      ann_id_gt = cocoGt.getAnnIds(imgIds=img_id, catIds=cat_id)
      ann_id_dt = cocoDt.getAnnIds(imgIds=img_id, catIds=cat_id)

      # if there is no annotation for a img_id and cat_id for both gt and dt then skip it.
      # as it means that class is not present in that image for both and hence is a True Negative (TN)
      # but we don't use TN in our calculation of IoU.
      if (len(ann_id_gt) == 0) and (len(ann_id_dt) == 0):
        continue
      
      # it means that something in detected for this class in the dt but that is not in gt. 
      # Hence a false positive
      if (len(ann_id_gt) == 0):

        ann_dt = cocoDt.loadAnns(ann_id_dt)[0]

        if mode == 'segm':
          mask_dt = cocoDt.annToMask(ann_dt)
          fp_data.loc[img_id, cat_id] = np.sum(mask_dt) # all the 1s in the detection mask are FPs

        else:
          fp_data.loc[img_id, cat_id] = int((ann_dt['bbox'][-1] + 1) * (ann_dt['bbox'][-2] + 1))

        continue
      
      # it means that something in present for that class in the gt but that is not in detection
      # Hence a false negative
      if (len(ann_id_dt) == 0):
          
        ann_gt = cocoGt.loadAnns(ann_id_gt)[0]

        if mode == 'segm':
          mask_gt = cocoGt.annToMask(ann_gt)
          fn_data.loc[img_id, cat_id] = np.sum(mask_gt) # all the 1s in the ground truth mask are FNs in the detection

        else:
          fn_data.loc[img_id, cat_id] = int((ann_gt['bbox'][-1] + 1) * (ann_gt['bbox'][-2] + 1))

        continue

      # get the corresponding ground truth and detection mask and find
      # all three TP, FP and FN. We reshape it for easier calculation
      ann_gt = cocoGt.loadAnns(ann_id_gt)[0]
      ann_dt = cocoDt.loadAnns(ann_id_dt)[0]

      if mode == 'segm':
        mask_gt = cocoGt.annToMask(ann_gt).reshape(-1)
        mask_dt = cocoDt.annToMask(ann_dt).reshape(-1)

        # update the TP, FN and FP data
        tp_data.loc[img_id, cat_id] = np.sum(np.logical_and(mask_gt==1, mask_dt==1))
        fn_data.loc[img_id, cat_id] = np.sum(np.logical_and(mask_gt==1, mask_dt==0))
        fp_data.loc[img_id, cat_id] = np.sum(np.logical_and(mask_gt==0, mask_dt==1))

      else:
        bbox_gt = ann_gt['bbox']
        bbox_dt = ann_dt['bbox']

        # converting the bbox from xmin, ymin, width, height --> xmin, ymin, xmax, ymax
        bbox_gt = bbox_gt[:2] + [bbox_gt[0] + bbox_gt[2], bbox_gt[1] + bbox_gt[3]]
        bbox_dt = bbox_dt[:2] + [bbox_dt[0] + bbox_dt[2], bbox_dt[1] + bbox_dt[3]]

        # the coordinates of the intersection bounding box
        bbox_inter = [max(bbox_gt[0], bbox_dt[0]), max(bbox_gt[1], bbox_dt[1]), 
                      min(bbox_gt[2], bbox_dt[2]), min(bbox_gt[3], bbox_dt[3])]
        
        # now checking if the bbox_inter is actually the intersection of the two boxes or not
        if (bbox_inter[2] >= bbox_inter[0]) and (bbox_inter[3] >= bbox_inter[1]):
            
          # update the TP, FN and FP data
          tp_data.loc[img_id, cat_id] = (bbox_inter[2] - bbox_inter[0] + 1) * (bbox_inter[3] - bbox_inter[1] + 1)
          fn_data.loc[img_id, cat_id] = max(int((bbox_gt[2] - bbox_gt[0] + 1) * (bbox_gt[3] - bbox_gt[1] + 1) - tp_data.loc[img_id, cat_id]), 0)
          fp_data.loc[img_id, cat_id] = max(int((bbox_dt[2] - bbox_dt[0] + 1) * (bbox_dt[3] - bbox_dt[1] + 1) - tp_data.loc[img_id, cat_id]), 0)
        
        else:

          # now this means that both the bboxes are disjoint and hence no TP, but the area of the GT will be FN and the area of the DT will be FP
          # update the TP, FN and FP data
          tp_data.loc[img_id, cat_id] = 0
          fn_data.loc[img_id, cat_id] = int((bbox_gt[2] - bbox_gt[0] + 1) * (bbox_gt[3] - bbox_gt[1] + 1))
          fp_data.loc[img_id, cat_id] = int((bbox_dt[2] - bbox_dt[0] + 1) * (bbox_dt[3] - bbox_dt[1] + 1))
  
  # replace all the unfilled places with 0 as 0 was detected there
  tp_data.fillna(0, inplace=True)
  fn_data.fillna(0, inplace=True)
  fp_data.fillna(0, inplace=True)

  # calculate the total TP, FN and FP for each class
  tp_per_class = tp_data.sum()
  fn_per_class = fn_data.sum()
  fp_per_class = fp_data.sum()

  # per class IoU
  IoU_per_class = tp_per_class / (fp_per_class + tp_per_class + fn_per_class)
  
  # mean IoU
  mIoU = IoU_per_class.mean()

  # if per class iou needs to be returned
  if return_per_class_iou:
      return IoU_per_class
  
  return mIoU