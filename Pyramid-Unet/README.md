
# Neural Network Usage Guide

## 1. Project Overview

This repository contains the implementation of a neural network for **label-free cell nucleus recognition from optical bright-field images**. The network utilizes a U-Net architecture for image segmentation, and additional modules are included for probe template matching, binary image processing, and contour detection. The code is organized into different modules for easy training, inference, and visualization.

## 2. Repository Structure

The following is an overview of the files and their functionality in the repository:

```
├── net.py                  # Defines the neural network architecture (UNet)
├── data.py                 # Handles data preprocessing and loading
├── train.py                # Training script for the neural network
├── predict.py              # Script for model inference (prediction)
├── template_matching.py    # Determine the actual distance and detection area of the cell nucleus and probe
├── utils.py                # Uniform image size
└── params/                 # Folder to store model weights
```

## 3. Requirements

Before using the code, ensure that the following dependencies are installed:

- **Python 3.7+**
- **PyTorch 1.7+**
- **torchvision**
- **OpenCV 4.5+**
- **Pillow**
- **numpy**
- **pandas**
- **matplotlib**

To install the necessary Python libraries, run the following command:

```bash
pip install torch torchvision opencv-python pillow numpy pandas matplotlib
```

## 4. Dataset Preparation

The model expects input images in **JPG and PNG** format, which will be processed and segmented by the U-Net. Ensure that the dataset is structured appropriately for training and inference. For example:
Data set creation using labelme (open source, available for download on github), labeling the cell region of interest, using the dots to continuously circle the outline
```
dataset/
├── train/
│   ├── Images/          # Input images for training
│   ├── labels/          # Ground truth segmentation masks

```

## 5. Running the Code

### 5.1. Training the Model

To train the neural network, you can run the `train.py` script. This script loads the data, applies any necessary transformations, and trains the model using the U-Net architecture defined in `net.py`.

**Command to start training**:

```bash
python train.py --epochs 2000 --batch_size 12 --learning_rate 0.0001 --train_dir dataset/train --val_dir dataset/val
```

**Parameters**:
- `--epoch`: Number of training epochs (default: 2000).
- `--batch_size`: Batch size during training (default: 12).
- `--lr`: Learning rate for the optimizer (default: 0.0001).
- `--DateSet`: Path to the training dataset.
- `--train_image_nucleus`: View training results.

### 5.2. Model Inference

Once the model is trained, you can use the `predict.py` script to perform inference (prediction) on new images.

**Command for inference**:

```bash
python predict.py --input_image 'input.tif' --model_weights 'params/weights.pth' --output_dir 'result/'
```

**Parameters**:
- `--input_image`: Path to the input image.
- `--model_weights`: Path to the pre-trained model weights (from `params/` directory).
- `--output_dir`: Directory to save the predicted outputs.

### 5.3. Probe Template Matching

To perform template matching for detecting the probe, use the `template_matching.py` script. This script takes an image and a probe template, performs the template matching operation, and outputs the result.
Then calculate the actual distance from all detectable cells to the tip in the detectable range of all probes (50 microns ×50 microns rectangle) and save it in the table. Calculate the center coordinate of 
   the next detectable area and save it to the end of the table.
**Command for template matching**:

```bash
python template_matching.py --input_image 'result/nucleus_boundary/path_to_original_image.jpg' --probe_image 'Probe.tif' --output_path 'box_image/box_image_result.tif'
```

### 5.4. Utility Functions

`utils.py` contains helper functions used throughout the project for:
- Image resizing
- Data transformations
- Saving/loading model checkpoints

These utility functions are integrated into the main scripts (`train.py`, `predict.py`), and they ensure the model runs efficiently.

## 6. Outputs

### Inference Output:
- **Segmentation Mask**: The result of the U-Net segmentation will be saved as an image (e.g., in JPG or PNG format).
- **Contour Information**: If contour detection is performed, a CSV file with the detected contour centroids is saved.
- **Processed Images**: Processed images will be saved to the output directory specified during inference.

### Coordinate Files:
- **probe_coordinates.csv**: Contains the coordinates of the detected probe tip.
- **coordinates.csv**: Contains the coordinates of detected nuclei or other features in the image.

## 7. Common Issues and Troubleshooting

1. **CUDA Out of Memory**:
   - If you encounter an out-of-memory error on your GPU, try reducing the batch size during training or switch to CPU inference.

2. **Model Weights Not Found**:
   - Ensure that the pre-trained model weights are placed in the `params/` directory and correctly referenced in the scripts.

3. **File Path Issues**:
   - Double-check that all file paths (input images, model weights, etc.) are correctly set, especially when running scripts in different directories.

## 8. Extending the Model

If you wish to extend the model for other tasks or modify its architecture, you can do so by editing the `net.py` file. The current U-Net implementation can be adapted to work with other types of images or additional layers depending on your use case.


