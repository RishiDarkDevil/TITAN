# Load Necessary Libraries
# General
import datetime
from tqdm import tqdm
import os
from typing import Tuple, List, Dict

# Image Handling
from PIL import Image

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
    self.image_name2id = dict() # Stores the image name to id mapping
  
  def load_image_name2id_dict(self):
    """
    Creates the image_name to id mapping given the `self.images`.
    """
    for img in self.images:
      self.image_name2id[img['file_name']] = img['id']
    print('Created image file name to id mapping... Done')
  
  def set_image_info(self, image_infos):
    """
    Sets the image infos in the `self.images`.
    - image_info: List of image info in the COCO format.
    """
    self.images = image_infos
    self.load_image_name2id_dict()
  
  def set_caption_info(self, cap_infos):
    """
    Sets the caption infos in the `self.captions`.
    - cap_infos: List of caption info in the COCO format
    """
    self.captions = cap_infos

  def create_image_infos(self):
    """
    Creates the image infos based on the images present in the `self.image_dir` folder.
    Useful when you are lazy to self-assign image info in COCO format to self.images
    """
    # get the image names
    image_names = os.listdir(self.image_dir)

    print('Getting Image Infos...', end='')

    # iterating over the image names
    for image_name in tqdm(image_names):

      # loading the image to get the size
      image = Image.open(os.path.join(self.image_dir, image_name)).convert('RGB')

      # Image details
      width, height = image.size
      image_det = {
          'license': license,
          'file_name': f'{image_name}',
          'height': height,
          'width': width,
          'date_captured': datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
          'id': self.image_id
      }

      # appending this info to the self.images
      self.images.append(image_det)
      
      # incrementing image id
      self.image_id += 1

    print('Done')

    # create the image name to image id mapping
    self.load_image_name2id_dict()
  
  def create_caption_infos(self, img_cap_dict):
    """
    Note: The image infos must be set before running this.
    Creates the caption infos based on the `img_cap_dict`.
    img_cap_dict: maps the image_name(with extension) to caption string
    """

    # Did you forget to run `set_image_info` or `create_image_infos`?
    assert len(self.image_name2id) > 0

    for image_name, caption_prompt in img_cap_dict.items():

      # load the corresponding image id for the `image_name`
      if image_name in self.image_name2id:
        curr_image_id = self.image_name2id[image_name]

      else:
        print(f'{image_name} is not present in the image info provided!')
        return
      
      # Captions details
      cap_det = {
          'id': self.caption_id,
          'image_id': curr_image_id,
          'caption': caption_prompt
      }

      # appending this info to the self.captions
      self.captions.append(cap_det)
      
      # incrementing caption id
      self.caption_id += 1

  def annotate_mask(self, 
    mask,
    image_name: str, 
    object_name: str,
    ):
    """
    Note: Before you run this function make sure you assign the image infos and caption infos using `set_image_info` and `set_caption_info`.
    If you are lazy to do that. Run `create_image_infos` and then `create_caption_infos`

    Updates all the COCO components - when you pass a mask corresponding to the `image_name`, it annotates the mask
    i.e. generates segmentation and bboxes and other fields.
    mask: [np.array] the segmentation mask (should be of the same size as the `image_name`) - preferably a binary image, heatmap also works
          The values of the mask should be between 0 to 1.
    image_name: the name with which this image is saved along with extension
    object_name: the object name for which the `mask` is given
    caption_prompt: the caption corresponding to the `image_name` if any
    """

    if mask is not None:
      # Check if the values of the mask are between 0 and 1.
      assert mask.max() <= 1.0

    # Did you forget to run `set_image_info` or `create_image_infos`?
    assert len(self.image_name2id) > 0

    # load the corresponding image id for the `image_name`
    if image_name in self.image_name2id:
      curr_image_id = self.image_name2id[image_name]
    else:
      print(f'{image_name} is not present in the image info provided!')
      return

    # If the object name is new then we add a new category
    if object_name not in self.cat2id:
      self.cat2id[object_name] = self.cat_id
      self.categories.append({"supercategory": '', "id": self.cat_id, "name": object_name}) ### FIX SUPERCATEGORY
      self.cat_id += 1

    if mask is not None:
      # word category id
      word_cat_id = self.cat2id[object_name]

      # Annotate the Word Heatmap for current word
      anns = self.object_annotator.wordheatmap_to_annotations(
        mask, self.annotation_id, curr_image_id, word_cat_id
        )

      # Update the annotations
      self.annotations.extend(anns)

      # Incrementent annotation id
      self.annotation_id += len(anns)

  def annotate(self, 
    image,
    image_name: str, 
    global_heat_map, 
    processed_prompt: Tuple[str, List[str], List[str]],
    license = 1,
    image_global_heat_map = None,
    **kwargs
    ):
    """
    Updates all the COCO components - Suited only when you are using DAAM or DAAM-I2I to annotate i.e. a fully
    generative pipeline and annotations using Attention Heatmaps of the generative model. The images are generated
    and annotated on the fly so we keep on updating the image info and caption info on the go as well.
    image: generated PIL image
    image_name: the name with which this image is saved along with extension
    global_heat_map: daam GlobalHeatMap
    processed_prompt: A Tuple of (sentence, tokenized and cleaned sentence, objects)
    image_global_heat_map: [Optional] daami2i GlobalHeatMap, if provided finds the DAAM-I2I guided heatmap (instead of DAAM word heatmap)
    """

    # The generated image
    output_image = image

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
        'license': license,
        'file_name': f'{image_name}',
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

      # Compute word heatmap for a non-stopword
      word_heatmap_temp = global_heat_map.compute_word_heat_map(word)
      
      if image_global_heat_map is None:
        # Compute word heatmap for a non-stopword
        word_heatmap = word_heatmap_temp.expand_as(output_image).numpy()
      
      else:
        # Compute daami2i word guided heatmap for a non-stopword
        word_heatmap = image_global_heat_map.compute_guided_heat_map(word_heatmap_temp.heatmap)
        word_heatmap = word_heatmap.expand_as(output_image).numpy()

      # Annotate the Word Heatmap for current word
      try:
        anns = self.object_annotator.wordheatmap_to_annotations(
          word_heatmap, self.annotation_id, self.image_id, word_cat_id, **kwargs
          )
      except Exception as e:
        print(f'The following error popped up while trying to annotate the word heatmap for {obj} in the image {image_name}:', e)
        anns = list()

      # # If no annotation detected for current word
      # if anns is None:

      #   # Undoing the changes in case we skip the prompt
      #   # Observe if an exception happen then it can only happen in the Generate Global Word Attribution HeatMap Section
      #   # Assuming that Stable Diffusion with output atleast something for each prompt
      #   # So if an exception happens in the above mentioned section then by then we have appended some things which we undo below
      #   rmv_count = len(new_words)

      #   # Deleting the new words which are detected on this prompt in case we deal with exception
      #   for del_word in new_words:
      #     self.cat2id.pop(del_word, None)
        
      #   # Delete the new categories we added
      #   for _ in range(rmv_count):
      #     self.categories.pop()
        
      #   # Delete the image generated
      #   if len(self.images) > 0:
      #     self.images.pop()

      #   # Delete the caption for this image
      #   if len(self.captions) > 0:
      #     self.captions.pop()

      #   # Fix category id
      #   self.cat_id -= rmv_count
        
      #   return

      # Update the annotations
      self.annotations.extend(anns)

      # Incrementent annotation id
      self.annotation_id += len(anns)

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
      f'object-detect-{self.save_idx}.json' if ann_outfile_name is None else f'{ann_outfile_name}'
      )

    # save captions
    save_captions(
      self.info,
      self.licenses,
      self.images,
      self.captions,
      self.caption_dir,
      f'object-caption-{self.save_idx}.json' if cap_outfile_name is None else f'{cap_outfile_name}'
      )

    self.save_idx += 1

  def clear(self):
    # Clearing out all the lists except cat2id to maintaining the unique category ids assigned to each new object
    self.images.clear()
    self.annotations.clear()
    self.categories.clear()
    self.captions.clear()