# SFU CMPT 419 Project -- ABCDetect
**ABCDetect**: Automated Boundary Classification and Detection for Skin Lesions

ABCDetect is a tool for skin lesion segmentation and classification using deep learning. It helps identify potential melanoma boundaries from dermoscopic images.

## Important Links

| [Timesheet](https://1sfu-my.sharepoint.com/:x:/g/personal/hamarneh_sfu_ca/EUcDkxJpdj5BhtqdgrlkBs8Bq7pB34Q2yUVFNF01W31fOQ?e=AL5Zei) | [Slack channel](https://cmpt419spring2025.slack.com/archives/C086CRM81JN) | [Project report](https://www.overleaf.com/4797734754qgyqrkymdzng#014097) |
|-----------|---------------|-------------------------|


## Video/demo/GIF
[Add your project demo video or GIF here - 1:40 to 2 minutes maximum]


## Table of Contents
1. [Demo](#demo)

2. [Installation](#installation)

3. [Reproducing this project](#repro)

4. [Guidance](#guide)


<a name="demo"></a>
## 1. Example demo

Run a complete demonstration of the project with dataset download, model training, and evaluation:

```bash
python -m abcdetect demo
```

To segment a single skin lesion image using a pre-trained model:

```bash
python -m abcdetect segment --image /path/to/your/image.jpg
```

### What to find where

```bash
repository
├── abcdetect/                   ## Source code package with segmentation and classification modules
│   ├── __main__.py              ## Entry point with CLI implementation
│   ├── download_dataset.py      ## Dataset download functionality
│   ├── segmentation.py          ## Segmentation model training and inference
│   └── ...                      ## Additional modules
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

Install PyTorch with the appropriate version for your system:

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
python -m abcdetect train --batch-size 32 --device cuda

# 3. Evaluate the model on the test set
python -m abcdetect demo

# 4. To segment a specific image using the trained model
python -m abcdetect segment --image path/to/test_image.jpg
```

The downloaded dataset will be stored in the `datasets` directory by default.
Model checkpoints and evaluation results will be saved in the `output` directory.
You can change these locations using the `--datasets-dir` and `--output-dir` options.

<a name="guide"></a>
## 4. Guidance

### CLI Usage Guide

```
python -m abcdetect [mode] [options]
```

Available modes:
- `download`: Download the HAM10K dataset only
- `train`: Train the segmentation model
- `segment`: Segment a specific image using a trained model
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
- Segment mode: `--image`, `-i`: Path to the image to segment
- Segment mode: `--model`, `-m`: Path to the model file (optional)

Example commands:
```bash
# Download dataset
python -m abcdetect download --force

# Train with specific settings
python -m abcdetect train --batch-size 64 --num-workers 4 --device cuda

# Segment an image
python -m abcdetect segment --image test_images/lesion.jpg

# Run full demo with custom directories
python -m abcdetect demo --datasets-dir ./my_datasets --output-dir ./results
```

- Use [git](https://git-scm.com/book/en/v2)
    - Do NOT use history re-editing (rebase)
    - Commit messages should be informative:
        - No: 'this should fix it', 'bump' commit messages
        - Yes: 'Resolve invalid API call in updating X'
    - Do NOT include IDE folders (.idea), or hidden files. Update your .gitignore where needed.
    - Do NOT use the repository to upload data
- Use [VSCode](https://code.visualstudio.com/) or a similarly powerful IDE
- Use [Copilot for free](https://dev.to/twizelissa/how-to-enable-github-copilot-for-free-as-student-4kal)
- Sign up for [GitHub Education](https://education.github.com/) 
