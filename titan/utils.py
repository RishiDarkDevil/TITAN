# General
import numpy as np
import gc
import os
import json
from typing import List, Dict
from tqdm import tqdm

# Model
import torch

class NpEncoder(json.JSONEncoder): 
  """
  To help encode the unsupported datatypes to json serializable format
  """

  def default(self, obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return json.JSONEncoder.default(self, obj)

def save_annotations(
  info: Dict, 
  licenses: List, 
  images: List, 
  annotations: List, 
  categories: List,
  out_directory: str = 'Data-Generated/annotations',
  outfile_name: str = 'object-detection.json'
  ):
  """
  Saving annotation file in COCO format given the components
  """

  # Serializing json
  json_obj_det = json.dumps({
    'info': info,
    'licenses': licenses,
    'images': images,
    'annotations': annotations,
    'categories': categories
  }, indent=4, cls=NpEncoder)

  # Writing json
  with open(os.path.join(out_directory, outfile_name), "w") as outfile:
    outfile.write(json_obj_det)

  print(f'Saved Annotations... {outfile_name}')

  # Delete json from python env
  del json_obj_det

def save_captions(
  info: Dict, 
  licenses: List, 
  images: List, 
  captions: List, 
  out_directory: str = 'Data-Generated/captions',
  outfile_name: str = 'object-caption.json'
  ):
  """
  Saving caption file in COCO format given the components
  """

  # Serializing json
  json_obj_cap = json.dumps({
    'info': info,
    'licenses': licenses,
    'images': images,
    'annotations': captions,
  }, indent=4, cls=NpEncoder)

  # Writing json
  with open(os.path.join(out_directory, outfile_name), "w") as outfile:
    outfile.write(json_obj_cap)

  print(f'Saved Captions... {outfile_name}')

  # Delete json from python env
  del json_obj_cap

def merge_annotation_files(
  annotation_directory: str = 'Data-Generated/annotations',
  annotation_file_names: List[str] = None,
  outfile_name: str = 'annotations.json',
  remove_parts_after_merge: bool = False
  ):
  """
  Merge/Concatenates all the part-wise annotation files into a single annotation file
  annotation_directory: the folder path which contain all these part-wise annotation files
  annotation_file_names: If None, all the files in the `annotation_directory` will be merged.
    If passed then only these file names in the annotation_directory will be merged
  """
  if annotation_file_names is None:
    # Annotation File Names present in the annotations directory
    ann_file_names = os.listdir(annotation_directory)
  else:
    ann_file_names = annotation_file_names

  print('Starting Annotation Files Merge...')
  print('Number of Annotation Files to be merged:', len(ann_file_names))
  print('Annotation File Names:', ' '.join(ann_file_names))

  ann_files = list() # Contains the list of loaded annotation json files
  
  for ann_file_name in tqdm(ann_file_names): # Loads the annotation json files and appens to ann_files

    with open(os.path.join(annotation_directory, ann_file_name)) as json_file:
    
      ann_file = json.load(json_file)
      ann_files.append(ann_file)
  
  # Creating the single annotation file
  # Assuming all the annotation files have same info and licenses
  annotation_file = {
      'info': ann_files[0]['info'],
      'licenses': ann_files[0]['licenses'],
      'images': [image for ann_file in ann_files for image in ann_file['images']],
      'annotations': [ann for ann_file in ann_files for ann in ann_file['annotations']],
      'categories': [cat for ann_file in ann_files for cat in ann_file['categories']]
  }
  
  # Serializing json
  ann_json_file = json.dumps(annotation_file, indent=4)
  # Writing json
  with open(os.path.join(annotation_directory, outfile_name), "w") as outfile:
    outfile.write(ann_json_file)

  print()
  print(f'Saved Annotation file... {outfile_name}')

  if remove_parts_after_merge:
    print('Removing the annotation files that were merged...', end='')
    for ann_file_name in ann_file_names:
      os.remove(os.path.join(annotation_directory, ann_file_name))
    print('Done')
  print('A successful merge!')


def merge_caption_files(
  caption_directory: str = 'Data-Generated/captions',
  caption_file_names: List[str] = None,
  outfile_name: str = 'captions.json',
  remove_parts_after_merge: bool = False
  ):
  """
  Merge/Concatenates all the part-wise caption files into a single caption file
  caption_directory: the folder path which contain all these part-wise caption files
  caption_file_names: If None, all the files in the `caption_directory` will be merged.
    If passed then only these file names in the caption_directory will be merged
  """
  if caption_file_names is None:
    # Caption File Names present in the caption directory
    cap_file_names = os.listdir(caption_directory)
  else:
    cap_file_names = caption_file_names

  print('Starting Caption Files Merge...')
  print('Number of Caption Files found:', len(cap_file_names))
  print('Caption Files found:', ' '.join(cap_file_names))

  cap_files = list() # Contains the list of loaded caption json files
  
  for cap_file_name in tqdm(cap_file_names): # Loads the caption json files and appens to cap_files

    with open(os.path.join(caption_directory, cap_file_name)) as json_file:
    
      cap_file = json.load(json_file)
      cap_files.append(cap_file)
  
  # Creating the single caption file
  # Assuming all the caption files have same info and licenses
  caption_file = {
      'info': cap_files[0]['info'],
      'licenses': cap_files[0]['licenses'],
      'images': [image for cap_file in cap_files for image in cap_file['images']],
      'annotations': [ann for cap_file in cap_files for ann in cap_file['annotations']],
  }
  
  # Serializing json
  cap_json_file = json.dumps(caption_file, indent=4)

  # Writing json
  with open(os.path.join(caption_directory, outfile_name), "w") as outfile:
    outfile.write(cap_json_file)

  print()
  print(f'Saved Caption file... {outfile_name}')

  if remove_parts_after_merge:
    print('Removing the annotation files that were merged...', end='')
    for cap_file_name in cap_file_names:
      os.remove(os.path.join(caption_directory, cap_file_name))
    print('Done')
  print('A successful merge!')

def merge_img_cat_id(
    annotation_file_path: str = 'Data-Generated/annotations/annotations.json'
):
  """
  In the annotation file if there are images with multiple annotations for same category id.
  This merges all of them and overwrites the input annotation file.
  It also adds `score` parameter in the annotations if already not present.
  """
  # loading the annotation file
  with open(annotation_file_path) as json_file:
    annotation_file = json.load(json_file)

  # contains the image_id, cat_id and the annotation_file['annotations'] indices corresponding to that.
  img_cats_dict = dict()

  # fills the `img_cats_dict`
  for idx, ann in enumerate(annotation_file['annotations']):
    if (ann['image_id'], ann['category_id']) not in img_cats_dict:
      img_cats_dict[(ann['image_id'], ann['category_id'])] = [idx]
    else:
      img_cats_dict[(ann['image_id'], ann['category_id'])].append(idx)
  
  annotations_temp = list() # stores the merged annotations
  ann_id = 0 # keeps track of the annotation id

  # iterating over all the img_id and cat_id pairs and their respective indices in annotation_file['annotations']
  for img_cat_id, to_be_merged in img_cats_dict.items():

    # the image id and cat id
    img_id = img_cat_id[0]
    cat_id = img_cat_id[1]

    # stores the merged annotations for the same image_id and category_id
    segmentation = list()
    area = 0
    bbox = [float('inf'),float('inf'),0,0]
    score = 0.0 

    # iterating over all the indices in the original annotations that are to be merged
    for idx in to_be_merged:

      # pick the annotation to be merged and start merging process
      ann_temp = annotation_file['annotations'][idx]

      # separate contours but pertaining to the same category id in the same image are merged 
      segmentation.extend(ann_temp['segmentation'])

      # areas of these disjoint segments are added
      area += ann_temp['area']

      # sum scores if present in the dict
      score += ann_temp.get('score', 0.0)

      # making the box that is the largest size and contains all the boxes
      box = ann_temp['bbox']
      bbox[0] = min(bbox[0], box[0])
      bbox[1] = min(bbox[1], box[1])
      bbox[2] = max(bbox[2], box[0] + box[2])
      bbox[3] = max(bbox[3], box[1] + box[3])
    
    # converting the xmin, ymin, xmax, ymax --> xmin, ymin, width, height
    bbox = [int(bbox[0]), int(bbox[1]), int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])]

    # incrementing annotaiton id
    ann_id += 1

    # adding the merged annotation to the annotations_temp
    annotations_temp.append({
      'segmentation': segmentation,
      'area': area,
      'iscrowd': 0,
      'image_id': img_id,
      'bbox': bbox,
      'category_id': cat_id,
      'id': ann_id,
      'score':score / len(to_be_merged)
    })

  # putting the new annotations in place
  annotation_file['annotations'] = annotations_temp

  # Serializing json
  json_obj_det = json.dumps(annotation_file, indent=4, cls=NpEncoder)

  # Writing json
  with open(annotation_file_path, "w") as outfile:
    outfile.write(json_obj_det)

  print(f'Saved Annotations at... {annotation_file_path}')
  

def optimize_gpu():
  """
  Frees up GPU to help reduce memory leak
  Reset Already occupied Memory and Cache
  """
  torch.cuda.reset_max_memory_allocated()
  torch.cuda.reset_max_memory_cached()
  torch.cuda.empty_cache()

  # Garbage Collection
  gc.collect()


def find_max_class_id(
  annotation_directory: str = 'Data-Generated/annotations',
  annotation_file_names: List[str] = None
  ):
  """
  Finds the max class id useful 
  when trying to train DETR Object detection model
  annotation_directory: The directory containing the annotation files to be merged
  annotation_file_names: (optional) the names of the annotation files along with extension in the `annotation_directory` to be merged
  (If not passed considers all the files in the directory)

  """
  if annotation_file_names is None:
    # Annotation File Names present in the annotations directory
    ann_file_names = os.listdir(annotation_directory)
  else:
    ann_file_names = annotation_file_names
  
  max_class_id = -1
  
  for ann_file_name in tqdm(ann_file_names): # Loads the annotation json files and appens to ann_files

    with open(os.path.join(annotation_directory, ann_file_name)) as json_file:
    
      ann_file = json.load(json_file)
      categories = ann_file['categories']
      # get the max class id for the current annotation file
      max_class_id = max([x['id'] for x in categories] + [max_class_id])

  return max_class_id
