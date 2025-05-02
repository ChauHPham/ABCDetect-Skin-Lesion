# SFU CMPT 419 Project -- ABCDetect
**ABCDetect**: Automated Boundary Classification and Detection for Skin Lesions

ABCDetect is a tool for skin lesion segmentation and classification using deep learning. It helps identify potential melanoma boundaries from dermoscopic images and performs automated ABCD rule analysis to assist in diagnosis.

## Important Links

| [Timesheet](https://1sfu-my.sharepoint.com/:x:/g/personal/hamarneh_sfu_ca/EUcDkxJpdj5BhtqdgrlkBs8Bq7pB34Q2yUVFNF01W31fOQ?e=AL5Zei) | [Slack channel](https://cmpt419spring2025.slack.com/archives/C086CRM81JN) | [Project report](https://www.overleaf.com/4797734754qgyqrkymdzng#014097) |
|-----------|---------------|-------------------------|

## Background

The ABCD rule is a clinical algorithm used by dermatologists to evaluate suspicious skin lesions and detect potential melanomas. Each letter represents a feature to be analyzed:

- **A**: Asymmetry - Lesions are evaluated for differences in color, structure, and contour across perpendicular axes.
- **B**: Border irregularity - The lesion's border is assessed for abrupt or indistinct pigment patterns.
- **C**: Color variation - The presence of specific colors (e.g., white, red, brown, blue-gray, black) is analyzed.
- **D**: Dermoscopic structures - Key structural features such as pigment networks, dots, and globules are identified.

ABCDetect automates this analysis by first segmenting the lesion and then quantifying each ABCD feature to calculate a Total Dermoscopic Score (TDS), which can help classify the lesion as benign, suspicious, or likely malignant.

## Video/demo/GIF
Video Link: https://drive.google.com/file/d/1KuKjS7i70OPOy5zqHXcjzu_l7hwtBvab/view
![SPOILER_image](https://github.com/user-attachments/assets/a0cd7254-5813-4509-b7d6-4bac8a8f6685)
![dfbgwrgwe](https://github.com/user-attachments/assets/856839b9-8381-45b5-9a5b-91c4c154a3ff)
![sbfhmt](https://github.com/user-attachments/assets/5e05ca5c-8411-4f86-95d4-199efa1183b4)
![ytdmsrm](https://github.com/user-attachments/assets/9986efb6-53f6-4767-baa4-966b83712948)
![sbsdeberb](https://github.com/user-attachments/assets/6929b4c4-99c3-451e-9729-e18fcb19bafe)
![htrjnrj4st](https://github.com/user-attachments/assets/a6695c7f-8fdd-4ad8-8b3f-c50ce2927846)
![tyl,tukty](https://github.com/user-attachments/assets/0ec08bb8-731e-44e2-bdaf-ebdef024249c)
![fmfgnmd](https://github.com/user-attachments/assets/c0a21ccc-0bf6-49f1-b40a-c86c2b3ccf60)
![bvhfjngn](https://github.com/user-attachments/assets/204e1de3-ef44-4540-a03a-a4a0a1b048de)
![mghfmxgfm](https://github.com/user-attachments/assets/14519241-404c-4841-8359-167df2b1ae7f)
![ethnestb](https://github.com/user-attachments/assets/ae16362a-5dfd-4c11-bd50-9682a67e1829)
![tmtdmty](https://github.com/user-attachments/assets/36eec9f9-ecee-463a-8995-0b947a82e199)
![yrhtrherh](https://github.com/user-attachments/assets/0ffa34ad-ca8f-4a0d-8a5e-dbd954d65501)



## Table of Contents
1. [Demo](#demo)

2. [Installation](#installation)

3. [Reproducing this project](#repro)

4. [Using Pre-trained Model](#pretrained)

5. [Guidance](#guide)


<a name="demo"></a>

# Performing ABCD detection on a single image
To analyze a single skin lesion image and visualize the ABCD analysis:
**NOTE: You must first download or train a model, see *[Using Pre-trained Model](#pretrained)*.**

```bash
python -m abcdetect analyze --image /path/to/your/image.jpg --show-graph
```

## 2. Training a model and performing a demonstration of the ABCD detection
Run a complete demonstration of the project with dataset download, model training, and evaluation:

```bash
python -m abcdetect demo --show-graph
```
## 2. Training a model only
For segmentation only without ABCD analysis:

```bash
python -m abcdetect segment --image /path/to/your/image.jpg --show-graph
```

The `--show-graph` option displays visualization of each step in the ABCD analysis, including:
- Asymmetry axis detection and measurement
- Border irregularity assessment
- Color distribution analysis
- Dermoscopic structure identification

### What to find where

```bash
repository
├── abcdetect/                   ## Source code package with segmentation and classification modules
│   ├── __main__.py              ## Entry point with CLI implementation
│   ├── download_dataset.py      ## Dataset download functionality
│   ├── segmentation.py          ## Segmentation model training and inference
│   ├── classification/
│   │   ├── asymmetry.py         ## Asymmetry detection algorithms
│   │   ├── border.py            ## Border irregularity assessment
│   │   ├── colour.py            ## Color variation detection
│   │   ├── dermoscopy.py        ## Dermoscopic structure analysis
│   │   └── ...
│   └── ...
├── docs/                        ## Documentation of the project and the libraries used  
├── README.md                    ## You are here
├── requirements.txt             ## Dependencies of the project
```

<a name="installation"></a>

## 2. Installation

ABCDetect requires Python 3.12 or later and PyTorch. Follow these steps to set up the environment:

```bash
# Clone the repository
git clone https://github.com/sfu-cmpt419/2025_1_project_06.git
cd 2025_1_project_06

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

You will also need to install PyTorch with the appropriate version for your system:

```bash
# For CUDA support (if you have a compatible NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# For CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

You can also visit the [PyTorch installation page](https://pytorch.org/get-started/locally/#start-locally) to get the exact command for your system configuration.

<a name="repro"></a>
## 3. Reproduction

To reproduce the results in our report, follow these steps:

```bash
# 1. Download the HAM10K dataset
python -m abcdetect download

# 2. Train the segmentation model (adjust batch size based on your GPU memory)
python -m abcdetect train --batch-size 48 --device cuda

# 3. Evaluate the model on the test set (optional)
python -m abcdetect demo --show-graph

# 4. To analyze a specific image using the trained model with visualization
python -m abcdetect analyze --image path/to/test_image.jpg --show-graph
```

The downloaded dataset will be stored in the `datasets` directory by default.
Model checkpoints and evaluation results will be saved in the `output` directory.
You can change these locations using the `--datasets-dir` and `--output-dir` options.

<a name="pretrained"></a>
## 4. Using a Pre-trained Model

If you prefer to skip the training process, you can use our pre-trained model available on Hugging Face:

1. Visit [our Hugging Face repository](https://huggingface.co/ColwynAIWiz/LesionSegmentation/tree/main)
   > **NOTE: You will need a Hugging Face account to download the model.**
2. Download the latest `.pt` model file (the one with the most recent timestamp)
3. Use the model for ABCD analysis/segment/demo:

```bash
# Specify the model path directly with --model
python -m abcdetect analyze --model path/to/downloaded/model.pt --image path/to/your/image.jpg --show-graph

# OR place the model in your output directory and it will be used automatically
python -m abcdetect analyze --image path/to/your/image.jpg --show-graph
```

<a name="guide"></a>
## 5. Guidance

### CLI Usage Guide

```
python -m abcdetect [mode] [options]
```

Available modes:
- `download`: Download the HAM10K dataset only
- `train`: Train the segmentation model
- `segment`: Segment a specific image using a trained model
- `analyze`: Perform ABCD analysis on a specific image
- `demo`: Run the full pipeline (download, train, evaluate)

Common options:
- `--datasets-dir`, `-d`: Directory to store datasets (default: ./datasets)
- `--output-dir`, `-o`: Directory for output files (default: ./output)
- `--device`, `--dev`: Device to use (auto, cuda, cpu)
- `--show-graph`, `-v`: Show visualizations during execution

Mode-specific options:
- Download mode: `--force`, `-f`: Force re-download even if data exists
- Train mode: `--batch-size`, `-k`: Batch size for training (default: 32)
- Train mode: `--num-workers`, `-n`: Number of data loader workers
- Segment/Analyze mode: `--image`, `-i`: Path to the image to process
- Segment/Analyze mode: `--model`, `-m`: Path to the model file (optional)

Example commands:
```bash
# Download dataset
python -m abcdetect download --force

# Train with specific settings
python -m abcdetect train --batch-size 64 --num-workers 4 --device cuda

# Segment an image
python -m abcdetect segment --image test_images/lesion.jpg

# Analyze an image with ABCD rule and visualization
python -m abcdetect analyze --image test_images/lesion.jpg --show-graph

# Run full demo with custom directories
python -m abcdetect demo --datasets-dir ./my_datasets --output-dir ./results
```
