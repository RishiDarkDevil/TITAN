# General
from typing import Tuple
from IPython.display import clear_output
import functools
import os

# Plotting
import matplotlib.pyplot as plt

# Image Handling
from PIL import Image

# Data Handling
from pycocotools.coco import COCO

# Matrix Manipulation
import numpy as np

# Visualization
from ipywidgets import Output
from ipywidgets import Button, HBox, VBox, Output

class TITANViz(COCO):

  def __init__(
    self, 
    image_dir: str = 'Data-Generated/images',
    annotation_file: str = 'Data-Generated/annotations/annotations.json',
    caption_file: str = 'Data-Generated/captions/captions.json'
    ):

    # Call COCO
    super().__init__(annotation_file)

    self.annotation_file = annotation_file
    self.caption_dataset = COCO(caption_file)
    self.image_dir = image_dir


  # Helper Utility to display only annotations of a particular object class
  def _visualize_object_annotation(self, class_name, anns, ann_names, im, out, figsize = (20,10)):

    # Set plot figure size
    plt.rcParams['figure.figsize'] = (10,10)

    # select annotations with `class_name`
    anns1 = [ann for ann in anns if self.loadCats(ann['category_id'])[0]['name'] == class_name]
    ann_names1 = [name for name in ann_names if name == class_name]
    
    # display results
    with out:
      clear_output(True)
      # Plot and visualize results
      plt.axis('off')
      plt.imshow(np.asarray(im))
      self.showAnns(anns1, draw_bbox=True)
      for i, ann in enumerate(anns1):
        plt.text(anns1[i]['bbox'][0], anns1[i]['bbox'][1], ann_names1[i], style='italic', 
                bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 5})
      plt.suptitle(class_name)
      plt.show()

    # reset default plot size
    plt.rcParams['figure.figsize'] = (8,6)

  def visualize_annotation(self, image_name: str = '0_0.png', image_id: int = None, figsize: Tuple[int, int] = (20, 10)):
    """
    Visualize the Annotations of the dataset in COCO Format in an interactive way
    image_name: The image_name for which annotation needs to be displayed
    image_id: The image_id for which annotation needs to be displayed 
    (If passed image_name is ignored)
    """

    print('----- INTERACTIVE IMAGE ANNOTATION VISUALIZER -----')

    # Set default plotting dims
    plt.rcParams["figure.figsize"] = figsize

    if image_id == None:
      for image in self.dataset['images']:
        if image['file_name'] == image_name:
          image_id = image['id']

    # Load Image Details corresponding to the image_id
    img_info = self.loadImgs([image_id])[0]
    img_file_name = img_info['file_name']

    # Load Annotation Details corresponding to the image_id
    ann_ids = self.getAnnIds(imgIds=[image_id], iscrowd=None)
    anns = self.loadAnns(ann_ids)
    ann_names = [self.loadCats(ann['category_id'])[0]['name'] for ann in anns]

    # Load Caption Details corresponding to the image_id
    cap_ids = self.caption_dataset.getAnnIds(imgIds=[image_id])
    cap = self.caption_dataset.loadAnns(cap_ids)

    # Load Image corresponding to the image_id
    im = Image.open(os.path.join(self.image_dir, img_file_name))

    # output will be displayed here
    out = Output()

    # All the words to be displayed as buttons
    words = list(set(ann_names))

    # Object buttons
    items = [Button(description=w) for w in words]
    
    # Display Buttons in a nice grid with 10 columns
    rows = list()
    for i in range(len(items)//10+1):
      rows.append(HBox(items[(i*10):((i+1)*10)]))
    object_bttns = VBox(rows)
    display(object_bttns)

    # Display Annotation and Image
    with out:
      fig, ax = plt.subplots(1, 2)
      ax[0].axis('off')
      ax[1].axis('off')
      ax[0].imshow(np.asarray(im))
      ax[1].imshow(np.asarray(im))
      self.showAnns(anns, draw_bbox=True)
      for i, ann in enumerate(anns):
        ax[1].text(anns[i]['bbox'][0], anns[i]['bbox'][1], ann_names[i], style='italic', 
                bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 5})
      fig.suptitle(cap[0]['caption'])
      plt.show()

    # Add click functionality to buttons
    for item in items:
      item.on_click(functools.partial(self._visualize_object_annotation, item.description, anns, ann_names, im, out))
    display(out)
