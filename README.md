# TITAN: Large-Scale Visual ObjecT DIscovery Through Text attention using StAble DiffusioN

TITAN is a All-In-One destination if you are willing to create a synthetic segmentation/object annotated dataset if you have access to only prompts! The entire pipeline is very intuitive and you can have your dataset ready with less than 30 lines of code!

It relies on Stable Diffusion/Diffusion Models and Diffusion Attentive Attribution Map.

## Getting Started

First install [PyTorch](https://pytorch.org/) for your platform. You may also check the [Colab Tutorial](https://colab.research.google.com/drive/1CIHmTALtNLs4Pj7QrU2N9F6emJFTzBRB?usp=sharing).

### Installation

The following steps are required for setting up the `titan` package. The instructions are made kept in mind the Colab environment for ease of understanding. Feel free to adapt it to work on your local machine/cloud server.

```console
pip install git+https://github.com/RishiDarkDevil/TITAN.git
```

### Using TITAN as a Library

Import and use TITAN as follows:

```python
# For Stable Diffusion
from diffusers import StableDiffusionPipeline

# For Heatmap Generation
import daam

# For TITAN workflow
from titan import *
```

The several parts of a Data Generation Pipeline supported by TITAN:
- Detect Objects from List of Prompts.
  ```python

  # List of prompts
  prompts = [
  "A group of people stand in the back of trucks filled with cotton.",
  "A mother and three children collecting garbage from a blue and white garbage can on the street.",
  ]

  # Load PromptHandler from TITAN
  prompt_handler = PromptHandler()

  # Filter out the objects from the prompts to be used for annotations
  processed_prompts = prompt_handler.clean_prompt(prompts)

  print(processed_prompts)
  ```
- Object Detected Images using `processed_prompts`. We will use `stabilityai/stable-diffusion-2-base` as our Diffusion Model here.
  ```python
  
  # Diffusion Model Setup
  DIFFUSION_MODEL_PATH = 'stabilityai/stable-diffusion-2-base'
  DEVICE = 'cuda' # device
  NUM_IMAGES_PER_PROMPT = 1 # Number of images to be generated per prompt
  NUM_INFERENCE_STEPS = 50 # Number of inference steps to the Diffusion Model
  SAVE_AFTER_NUM_IMAGES = 1 # Number of images after which the annotation and caption files will be saved
  
  # Load Model
  model = StableDiffusionPipeline.from_pretrained(DIFFUSION_MODEL_PATH)
  model = model.to(DEVICE) # Set it to something else if needed, make sure DAAM supports that
  ```
  
  Now coming to Annotations. We will need a dataset which will store all the results from time to time on the disk and keep track of all the internal variables while generation.
  ```python
  
  # The TITAN Dataset
  titan_dataset = TITANDataset()
  
  
  # Generating and Annotating Generated Images
  try:

    # Iterating over the processed_prompts
    for i, processed_prompt in enumerate(processed_prompts):

      # Generating images for these processed prompts and annotating them
      for j in range(NUM_IMAGES_PER_PROMPT):

        # traversing the processed prompts
        prompt, _, _ = processed_prompt

        print()
        print(f'Prompt No.: {i+1}/{len(processed_prompts)}')
        print(f'Image No.: {j+1}/{NUM_IMAGES_PER_PROMPT}')
        print('Generating Image...')

        # generating images
        with daam.trace(model) as trc:
          output_image = model(prompt, num_inference_steps=NUM_INFERENCE_STEPS).images[0]
          global_heat_map = trc.compute_global_heat_map()
        
        # Saving Generated Image
        output_image.save(os.path.join(titan_dataset.image_dir, f'{i}_{j}.png'))
        print(f'Saved Generated Image... {i}_{j}.png')
        
        # Object Annotate Image
        print(f'Adding Annotation for {i}_{j}.png')
        titan_dataset.annotate(output_image, f'{i}_{j}.png', global_heat_map, processed_prompt)

        if len(titan_dataset.images) % SAVE_AFTER_NUM_IMAGES == 0:
          print()
          # Saving Annotations on Disk
          titan_dataset.save()
          # Freeing up Memory
          titan_dataset.clear()

    if len(titan_dataset.annotations):
      titan_dataset.save()
      titan_dataset.clear()

  except KeyboardInterrupt: # In case of KeyboardInterrupt save the annotations and captions
    titan_dataset.save()
    titan_dataset.clear()
  
  # merge annotation and caption files
  merge_annotation_files()
  merge_caption_files()
  ```
- Interactive Annotation Visualizer
  ```python
  
  # Load the Visualizer
  titan_visualizer = TITANViz()
  
  # Interactive Annotation Visualizer
  titan_visualizer.visualize_annotation(image_id = 1)
  ```
- More Features and Workflow Helpers to come soon!

### Citation

Add Link to this Repo for citation purpose.
