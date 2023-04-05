# General
import numpy as np
import os
import json
from typing import List, Dict

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
  annotations: List, 
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
  captiontion_directory: str = 'Data-Generated/captions',
  caption_file_names: List[str] = None,
  outfile_name: str = 'annotations.json',
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