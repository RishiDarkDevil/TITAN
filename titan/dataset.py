# Load Necessary Libraries
# General
import datetime
from tqdm import tqdm
import os
from typing import Tuple, List, Dict

# ObjectAnnotator
from .annotate_objects import ObjectAnnotator

# utilities
from .utils import save_annotations
from .utils import save_captions

# Defaults that are used if not specified to be something else
INFO = { # Info about the dataset
  "description": "Rishi Generated Data",
  "url": "https://github.com/RishiDarkDevil/Text-Based-Object-Discovery",
  "version": "1.0",
  "year": 2023,
  "contributor": "Rishi Dey Chowdhury (RishiDarkDevil)",
  "date_created": "2023"
}
LICENSES = [{ # Licenses associated with the dataset
    'url': 'https://huggingface.co/stabilityai/stable-diffusion-2/blob/main/LICENSE-MODEL',
    'id': 1,
    'name': 'CreativeML Open RAIL++-M License'
}]

class TITANDataset:
  """
  Handles all the synthetic detection dataset
  """

  def __init__(
    self,
    root_dir: str = 'Data-Generated',
    image_dir: str = 'images',
    annotation_dir: str = 'annotations',
    caption_dir: str = 'captions',
    info: Dict = INFO,
    licenses: List = LICENSES,
    **kwargs
    ):
    """
    root_dir: It contains the entire dataset
    image_dir: It is the name of the subfolder within `root_dir` which contains the iamges
    annotation_dir: It is the name of the subfolder within `root_dir` which contaitns the annotations
    caption_dir: It is the name of the subfolder within `root_dir` which contains the captions
    info: info dictionary of the dataset in COCO format
    licenses: list of dictionaries of the dataset in COCO format
    """
    self.root_dir = root_dir
    self.image_dir = os.path.join(root_dir, image_dir)
    self.annotation_dir = os.path.join(root_dir, annotation_dir)
    self.caption_dir = os.path.join(root_dir, caption_dir)

    # Create the folders which does not exists
    if not os.path.exists(self.root_dir):
      os.mkdir(self.root_dir)
    if not os.path.exists(self.image_dir):
      os.mkdir(self.image_dir)
    if not os.path.exists(self.annotation_dir):
      os.mkdir(self.annotation_dir)
    if not os.path.exists(self.caption_dir):
      os.mkdir(self.caption_dir)

    self.info = info
    self.licenses = licenses

    self.images = list() # Stores the generated image info
    self.annotations = list() # Stores the annotation info
    self.categories = list() # Stores the category info
    self.captions = list() # Stores the captions info
    self.cat2id = dict() # Stores the category to id mapping
    self.cat_id = 1 # Assigns id to categories as we go on adding categories which we discover
    self.image_id = 1 # Assigns generated image ids
    self.annotation_id = 1 # Assigns annotations annotation ids
    self.caption_id = 1 # Assigns captions caption ids
    self.save_idx = 1 # The index which stores how many times we saved the json file before
    self.object_annotator = ObjectAnnotator(**kwargs) # Annotates the WordHeatMaps

  def update(self, 
    image,
    image_name: str, 
    heatmap, 
    processed_prompt: Tuple[str, List[str], List[str]]
    ):
    """
    Updates all the COCO components
    image: generated PIL image
    heatmap: daam GlobalHeatMap
    processed_prompt: A Tuple of (sentence, tokenized and cleaned sentence, objects)
    """

    # The generated image
    output_image = image

    # The WordHeatMaps are to be obtained from here
    global_heat_map = heatmap

    # Picking up the ith original prompt, cleaned prompt and object prompt
    prompt, cleaned_prompt, object_prompt = processed_prompt

    # Stores the new objects found in this prompt
    new_words = list()
    
    # Updating Categories using cleaned prompt if required and assigning index
    for ind, word in enumerate(object_prompt):
      if word not in self.cat2id:
        new_words.append(word)
        self.cat2id[word] = self.cat_id
        self.categories.append({"supercategory": '', "id": self.cat_id, "name": word}) ### FIX SUPERCATEGORY
        self.cat_id += 1

    # Image details
    width, height = output_image.size
    image_det = {
        'license': 1,
        'file_name': f'{image_name}.png',
        'height': height,
        'width': width,
        'date_captured': datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        'id': self.image_id
    }
    self.images.append(image_det)

    # Captions details
    cap_det = {
        'id': self.caption_id,
        'image_id': self.image_id,
        'caption': prompt
    }
    self.captions.append(cap_det)

    # Generate Global Word Attribution HeatMap
    for word, obj in tqdm(zip(cleaned_prompt, object_prompt)):

      # word category id
      word_cat_id = self.cat2id[obj]
      
      # Compute heatmap for a non-stopword
      word_heatmap = global_heat_map.compute_word_heat_map(word).expand_as(output_image).numpy()

      # Annotate the Word Heatmap for current word
      anns = self.object_annotator.wordheatmap_to_annotations(
        word_heatmap, self.annotation_id, self.image_id, word_cat_id
        )

      # If no annotation detected for current word
      if anns is None:

        # Undoing the changes in case we skip the prompt
        # Observe if an exception happen then it can only happen in the Generate Global Word Attribution HeatMap Section
        # Assuming that Stable Diffusion with output atleast something for each prompt
        # So if an exception happens in the above mentioned section then by then we have appended some things which we undo below
        rmv_count = len(new_words)

        # Deleting the new words which are detected on this prompt in case we deal with exception
        for del_word in new_words:
          self.cat2id.pop(del_word, None)
        
        # Delete the new categories we added
        for _ in range(rmv_count):
          self.categories.pop()
        
        # Delete the image generated
        if len(self.images) > 0:
          self.images.pop()

        # Delete the caption for this image
        if len(self.captions) > 0:
          self.captions.pop()

        # Fix category id
        self.cat_id -= rmv_count
        
        return

      # Update the annotations
      self.annotations.extend(anns)

      # Incrementent annotation id
      self.annotation_id + len(anns)

    # Increment image id and caption id
    self.image_id += 1
    self.caption_id += 1

  def save(self, ann_outfile_name: str = None, cap_outfile_name: str = None):

    # save annotations
    save_annotations(
      self.info,
      self.licenses,
      self.images,
      self.annotations,
      self.categories,
      self.annotation_dir,
      f'object-detection-{self.save_idx}.json' if ann_outfile_name is None else f'{ann_outfile_name}'
      )

    # save captions
    save_captions(
      self.info,
      self.licenses,
      self.images,
      self.annotations,
      self.caption_dir,
      f'object-caption-{self.save_idx}.json' if cap_outfile_name is None else f'{cap_outfile_name}'
      )

  def clear(self):
    # Clearing out all the lists except cat2id to maintaining the unique category ids assigned to each new object
    self.images.clear()
    self.annotations.clear()
    self.categories.clear()
    self.captions.clear()