# COCO with YOLO

In this tutorial, we will walk through each step to configure a Deeplodocus project for object detection on the COCO dataset using our implementation of YOLOv3.

![COCODemo](https://github.com/Deeplodocus/deeplodocus/blob/master/docs/figures/tutorials/coco/ground-truths.png  "COCODemo")

**Prerequisite steps:**

1. [Download the COCO Detection Dataset](#1-Download)
2. [Install pycocotools](#2-install-pycocotools)

**Project setup:**

1. [Initialise a new project](#1-initialise-the-project)
2. [Data Configuration](#2-data-configuration)
3. [Model Configuration](#3-model-configuration)
4. [Loss & Metric Configuration](#4-loss-metric-configuration)
5. [Optimiser Configuration](#5-optimiser-configuration)
6. [Transformer Configuration](#6-transformer-configuration)
    - [Input Transformer](#51-input-transformer)
    - [Output Transformer](#52-output-transformer)
7. [Training](#7-training)

A copy of this project can be cloned from [here](https://github.com/Deeplodocus/COCO-with-YOLO) - but don't forget to follow the prerequisite steps below. 

## Prerequisite Steps

### 1. Download

First of all, let's download the appropriate data from the [COCO website](http://cocodataset.org/).

Specifically, we need the following items: 

- 2017 Train images [download [18GB]](http://images.cocodataset.org/zips/train2017.zip)
- 2017 Val images [download [1GB]](http://images.cocodataset.org/zips/val2017.zip)
- 2017 Train/Val annotations [download [241MB]](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)

When you have initialised your Deeplodocus project, extract each of these into the data folder. 

### 2. Install pycocotools

We also need the to install [pycocotools](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools), on which the CocoDataset torchvision module is dependent. 

pycocotool requires Cython, so we'll install that first, with:

```bash
pip3 install Cython
```

Then we can install pycocotools itself with:

```bash
pip3 install pycocotools
```

## 1. Initialise the Project

Initialise a new Deeplodocus project in your terminal with:

```bash
deeplodocus new-project COCO-with-YOLO
```

## 2. Data Configuration

### 2.1. Include the Dataset

Make sure the dataset is in the right place.
After initialising your project and extracting COCO, the data in your project should be structured like this: 

```text
data
├─  annotations
│	├─  instances_train2017.json
│	└─  instances_val2017.json
├─  train2017
│	├─  000000000000.jpg
│	├─  000000000001.jpg
│	└─  ...
└─  val2017
    ├─  000000000000.jpg
    ├─  000000000001.jpg
    └─  ...
```


Setting up the data configurations is one of most complicated steps in this project - but bare with us, we'll soon be feeding COCO efficiently into our data-hungry network. 
Open up the **config/data.yaml** file, and let's get started. 
    
### 2.2. Data loader    

At the top of the file you'll see the entry for **dataloader**, use this to set the batch size and the number of workers. 
If you have limited GPU memory, you may need to reduce your batch size.

```yaml
dataloader:
  batch_size: 32        # Possible batch sizes will depend on the available memory 
  num_workers: 4        # This will depend on your CPU, you probably have at least 4 cores
```
### 2.3. Enable the required pipelines

Next, us the **enabled** entry to enable different types of pipeline. 
As we only have training and validation data in this case, we need to enable just the trainer and validator as follows:

```yaml
enabled:
  train: True           # Enable the trainer
  validation: True      # Enable the validator
  test: False           # There is no test data
  predict: False        # There is no prediction data
```

### 2.4. Configure the dataset

Finally, we arrive at the **datasets** entry.
We define this with a list of two items, which configure the training and validation portions of the dataset respectively. 
We'll start with the first item - training data - which can be configured as follows: 
    
```yaml
datasets:    
  # Training portion
  - name: COCO Train 2017           # Human-readable name
    type: train                     # Dataset type (train/validation/test/predict)
    num_instances: Null             # Number of instances to use (Null = use all)
    entries:                        # Define the input and label entries
      # Input Entry
      - name: COCO Image            # Human-readable name
        type: input                 # Entry type (input/label/additional data)
        load_as: image              # Load data as image
        convert_to: float32         # Convert to float32 (after data transforms)
        move_axis: [2, 0, 1]        # Permute : (h x w x ch) to (ch x h x w)
        enable_cache: True          # Give other entries access to this entry
        # We define one source for this entry - CocoDetection from torchvision
        sources:                      
          - name: CocoDetection             
            module: torchvision.datasets    
            kwargs:                         
              root: data/train2017         
              annFile: data/annotations/instances_train2017.json
      # Label Entry
      - name: COCO Label            # Human-readable name
        type: label                 # Entry type (input/label/additional data)
        load_as: given              # Do not change data type on loading
        convert_to: float32         # Convert to float32 (after data transforms)
        move_axis: Null             # No need for move axis
        enable_cache: False         # No other entries need access to this data
        # Define one source for this entry - point to data from the input entry
        sources:
          - name: SourcePointer     # Point to an existing data source
            module: Null            # Import from default modules
            kwargs:
              entry_id: 0           # Take data from the first entry (defined above)
              source_id: 0          # Take from the first (and only) source
              instance_id: 1        # Take the second item - the label
```

Now, we can include the validation configurations, which will look very similar. 
There are only 4 differences: 

1. dataset **name**
2. dataset **type**
3. input entry source **root**
4. input entry source **annFile**

Validation dataset configurations: 

```yaml
  # Validation portion
  - name: COCO Val 2017
    type: validation
    num_instances: Null
    entries:
      # Input
      - name: COCO Image
        type: input
        load_as: image
        convert_to: float32
        move_axis: [2, 0, 1]
        enable_cache: True
        sources:
          - name: CocoDetection
            module: torchvision.datasets
            kwargs:
              root: data/val2017
              annFile: data/annotations/instances_val2017.json
      # Label
      - name: COCO Label
        type: label
        load_as: given
        convert_to: float32
        move_axis: Null
        enable_cache: False
        sources:
          - name: SourcePointer 
            module: Null
            kwargs:
              entry_id: 0           
              source_id: 0          
              instance_id: 1   
``` 

!!! note "Why are we using a SourcePointer?"
    When using torchvision datasets, the input and label entries are loaded together in a list. 
    This does not change how we configure the input source.
    However, for the label source, we use a SourcePointer to reference the second item from the first (and only) source of the first (input) entry. 

## 3. Model Configuration

Open and edit the **config/model.yaml** file to specify our object detector. 

```yaml
name: YOLO                              # Select YOLO
module: deeplodocus.app.models.yolo     # From the deeplodocus app
from_file: False                        # Don't try to load from file
file: Null                              # No need to specify a file to load from
input_size:                             # Specify the input size
  - [3, 448, 448]                     
kwargs:                                 # Keyword arguments for the model class
  num_classes: 91                       # Number of classes in COCO
  backbone:                             # Specify the backbone
    name: Darknet53                     # Select Darknet53 (Darknet19 is also available)
    module: deeplodocus.app.models.darknet      # Point to the darknet module
    kwargs:                                     # Keyword arguments for the backbone  
      num_channels: 3                           # Tell it to expect an input with 3 channels 
      include_top: False                        # Don't include the classifier
```

That's it! 
YOLO is configured and ready to go. 

You should now be able to load the model and print a summary, like this: 

![YOLOSummary](https://github.com/Deeplodocus/deeplodocus/blob/master/docs/figures/tutorials/coco/yolo-summary.png "YOLOSummary")

!!! note "Want to know more about YOLO?"
    For an in-depth understanding of the network architecture, we strongly recommend reading the YOLO papers:
    
    1. [You Only Look Once: Unified Real-Time Object Detection](https://arxiv.org/pdf/1506.02640.pdf)
    2. [YOLO9000: Better, Faster, Stronger](https://arxiv.org/pdf/1612.08242.pdf)
    3. [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
    
    Additionally, our source code for [YOLO](https://github.com/Deeplodocus/deeplodocus/blob/master/deeplodocus/app/models/yolo.py) and [Darknet](https://github.com/Deeplodocus/deeplodocus/blob/master/deeplodocus/app/models/darknet.py) can be found on GitHub. 
    
## 4. Loss & Metric Configuration

### 4.1. Losses

We have split the YOLO loss function into three components which relate to the objectness score, bounding box coordinates and object classifications respectively.
Thus, we need to configure each independently as follows:

```yaml
ObjectLoss:
  name: ObjectLoss                        # Name of the Python class to use
  module: deeplodocus.app.losses.yolo     # Import from deeplodocus app
  weight: 1
  kwargs:
    iou_threshold: 0.5
    noobj_weight: 0.5                     # Equivalent to λ_noobj
BoxLoss:
  name: BoxLoss                           # Name of the Python class to use
  module: deeplodocus.app.losses.yolo     # Import from deeplodocus app
  weight: 5                               # Equivalent to lambda_coord
  kwargs:
    iou_threshold: 0.5
ClassLoss:
  name: ClassLoss                         # Name of the Python class to use
  module: deeplodocus.app.losses.yolo     # Import from deeplodocus app
  weight: 1
  kwargs: {}
```

We have implemented these loss functions as explained by [the original YOLO paper](https://arxiv.org/pdf/1506.02640.pdf), and all source code is published [here](https://github.com/Deeplodocus/deeplodocus/blob/master/deeplodocus/app/losses/yolo.py).

### 4.2. Metrics

Currently, Deeplodocus does not include any of the traditional metrics for evaluating object detection.
Unless you wish to include you own metrics, make sure that the **config/metrics.yaml** file is empty. 

## 5. Optimiser Configuration

Have a look at the optimiser configurations specified in **config/optimizer.yaml**.
By default we have specified the Adam optimiser from [torch.optim](https://pytorch.org/docs/stable/optim.html).
The learning rate is specified by **lr**, and additional parameters can also be given.  

```yaml
name: "Adam"
module: "torch.optim"
kwargs:
  lr: 0.0001
  betas: [0.9, 0.999]
  weight_decay: 0
```

!!! note 
    Make sure the learning rate is not too high, otherwise training can become unstable. 
    If you are able to use a pre-trained backbone, a learning rate of 1e-3 should be just fine. 
    However, if you are training from scratch - like in this tutorial - a lower learning rate will be necessary in the beginning. 
    
## 6. Transformer Configuration

The final critical step is the configuration of two data transformers:

1. An input transformer to pre-process images and labels before they are given to the network.
2. An output transformer for post-processing and visualisation. 

Edit the **config/transform.yaml** file as follows: 

```yaml
train:
  name: Train Transform Manager
  inputs:
    - config/transformers/input.yaml      # Path to input transformer
  labels:
    - '*inputs:0'                         # Point to the first input transformer
  additional_data: Null
  outputs:
    - config/transformers/output.yaml     # Path to output transformer
validation:
  name: Validation Transform Manager
  inputs:
    - config/transformers/input.yaml      # Path to input transformer
  labels: 
    - '*inputs:0'                         # Point to the first input transformer
  additional_data: Null
  outputs: 
    - config/transformers/output.yaml     # Path to output transformer
test:
  name: Test Transform Manager
  inputs: Null
  labels: Null
  additional_data: Null
  outputs: Null
predict:
  name: Predict Transform Manager
  inputs: Null
  additional_data: Null
  outputs: Null
```

!!! note "Why does the label transformer point to the input transformer?"
    COCO images are different sizes, therefore each must be resized before being concatenated into a batch.
    To keep the bounding boxes labels relevant, we need to normalise them by the width and height of their associated image before it is resized. 
    Therefore, each label transformer should point to the input transformer, thus each label transform will be dependant on transform applied to its corresponding image.
 
### 5.1. Input Transformer

We now need to define the input transformer that defines the sequence of functions to apply. 
Open the **config/transformers/input.yaml** file and edit as follows: 

```yaml
method: sequential
name: Transformer for COCO input
mandatory_transforms_start:
  - format_labels:
      name: reformat_pointer
      module: deeplodocus.app.transforms.yolo.input
      kwargs:
        n_obj: 100
  - resize:
      name: resize
      module: deeplodocus.app.transforms.yolo.input
      kwargs:
        shape: [448, 448]
transforms: Null
mandatory_transforms_end: Null
```

The first stage is use to format the label into an array of size (100 x 5), then normalise the box coordinates by the corresponding image shape.
    
- An input (image) is given to the **[reformat_pointer](https://github.com/Deeplodocus/deeplodocus/blob/master/deeplodocus/app/transforms/yolo/input.py)** function, which returns:
    - the image (unchanged) and,
    - a **TransformData** object that stores the shape of the given image as well as a second transform function named **[reformat](https://github.com/Deeplodocus/deeplodocus/blob/master/deeplodocus/app/transforms/yolo/label.py)**.
- Since the label transformer points to the input transformer, the label will inputted to the function specified by this **TransformData** object, which:
    - formats the label into a numpy array and,
    - normalises the box coordinages w.r.t the given image shape. 
    
The second stage is responsible for resizing the input image to (448 x 448 x 3). 
    
- The image is inputted to the **[resize](https://github.com/Deeplodocus/deeplodocus/blob/master/deeplodocus/app/transforms/yolo/input.py)** function, which returns:
    - the image, resized to (448 x 448) and,
    - a **TransformData** object that points to an **[empty](https://github.com/Deeplodocus/deeplodocus/blob/master/deeplodocus/app/transforms/__init__.py)** transformer function.
- The label is given to the **[empty](https://github.com/Deeplodocus/deeplodocus/blob/master/deeplodocus/app/transforms/__init__.py)** transform.
    - This is just a place holder transform - the label is returned unchanged.
    
This process is illustrated below:

![TransformerPipeline](https://github.com/Deeplodocus/deeplodocus/blob/master/docs/figures/tutorials/coco/transformer.png "TransformerPipeline")

### 5.2. Output Transformer

To visualise the outputs of our YOLO model during training or validation, we can apply some post-processing transforms. 
To do this, we need to initialise an output transformer configuration file. 

Navigate to the **config/transformers** directory and use the command:

```bash
deeplodocus output-transformer output.yaml
```
This will create a new configuration file that you can open and edit to look like this: 

```yaml
# Define skip - for use in multiple transforms
# A skip of 20 will cause the transforms to only process every 20th batch.
# (We may not want to visualise every batch.)
skip: &skip
  20 

name: Output Transformer
transforms:
  ConcatenateDetections:
    name: Concatenate
    module: deeplodocus.app.transforms.yolo.output
    kwargs: 
      skip: *skip                                   # How many batches to skip
  NonMaximumSuppression:
    name: NonMaximumSuppression
    module: deeplodocus.app.transforms.yolo.output
    kwargs:
      iou_threshold: 0.5                            # IoU threshold for NMS
      obj_threshold: 0.5                            # Threshold for suppression by objectness score
      skip: *skip                                   # How many batches to skip
  Visualization:
    name: Visualize
    module: deeplodocus.app.transforms.yolo.output
    kwargs:
      rows: Null                                    # No. of rows when displaying images
      cols: Null                                    # No. of cols when displaying images
      scale: 1                                      # Re-scale the images before displaying 
      wait: 1                                       # How long to wait (ms) (0 = wait for a keypress)
      width: 2                                      # Line width for drawing boxes
      skip: *skip                                   # How many batches to skip
      lab_col: [32, 200, 32]                        # Color for drawing ground truth boxes (BGR)
      det_col: [32, 32, 200]                        # Color for drawing model detections (BGR)
```

The first transform, collects all of the YOLO detections from each of the three scales into a single array.
The second applies removes all detections with object scores below **obj_threshold** and applies non-maximum suppression according to the given IoU threshold. 
The third and final function draws the ground truth and the object detections inferred by the network onto each image and displays the result in a window. 

The source code for each of these transform functions can be found [here](https://github.com/Deeplodocus/deeplodocus/blob/master/deeplodocus/app/transforms/yolo/output.py).

## 7. Training

Now you're good to go! Just run the project main file, enter **load()**, **train()** and let it run.
Don't forget that you can also play with the optimiser and training configurations.  