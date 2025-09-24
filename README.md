# HCCM for RoboSense Challenge 2025 - Track 4

[![Paper](https://img.shields.io/badge/arXiv-2508.21539-b31b1b.svg)](https://arxiv.org/pdf/2508.21539)
[![Competition](https://img.shields.io/badge/RoboSense-Track%20%234-blue)](https://robosense2025.github.io/track4)
[![GitHub](https://img.shields.io/badge/Official--Code-rhao--hur/HCCM-green)](https://github.com/rhao-hur/HCCM)

---

This is the code implementation of our `HCCM` model for the [RoboSense Challenge 2025 - Track #4: Cross-Modal Drone Navigation](https://robosense2025.github.io/track4).

For a detailed description of the `HCCM` algorithm, please refer to our ACM MM 2025 paper: **"HCCM: Hierarchical Cross-Granularity Contrastive and Matching Learning for Natural Language-Guided Drones"**.

---

## Environment Setup

We highly recommend using `conda` to create and manage an isolated Python environment:

```bash
# Create and activate the conda environment
conda create -n hccm python=3.9.20 -y
conda activate hccm
```

Next, please install the required dependencies in the following order:

```bash
# 1. Install PyTorch (CUDA 11.8)
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118

# 2. Install basic dependencies
pip install ftfy==6.3.0 regex==2024.9.11 fsspec==2024.10.0
pip install opencv-python==4.10.0.84

# 3. Install OpenMMLab libraries
pip install -U openmim==0.3.9 
mim install mmengine==0.10.5
pip install "numpy==1.26.4" scipy
mim install "mmcv==2.1.0"

# 4. Install Transformers & MMPretrain
pip install transformers==4.12.5 tokenizers==0.10.3
mim install "mmpretrain==1.2.0"

# 5. Install other libraries
pip install pycocotools==2.0.8
pip install timm==0.6.13
```

### Directory Structure

Please ensure your project root follows the structure below. Files marked with `(Competition File)` are specific to the RoboSense challenge.

```
.
├── datasets/
│   └── GeoText1652_Dataset/
│       ├── images/                         # Extracted GeoText1652 images
│       ├── Track_4_Phase_I_Images/         # Competition Phase I Images (Competition File)
│       ├── Track_4_Phase_II_Images/        # Competition Phase II Images (Competition File)
│       ├── test_queries_0615_Phase_I.txt   # Competition Phase I Queries (Competition File)
│       ├── PhaseII-queries.txt             # Competition Phase II Queries (Competition File)
│       └── ... (Other GeoText1652 files)
└── RoboSense25_HCCM_Code/
    ├── configs/
    ├── pretrain/
    │   ├── 16m_base_model_state_step_199999_(xvlm2mmcv).pth
    │   └── bert-base-uncased/
    ├── src/
    ├── tools/
    ├── submit.py                           # Competition submission script (Competition File)
    └── README.md
```

## Data and Model Preparation

### 1. Dataset

First, create and navigate to the `datasets` directory:
```bash
mkdir datasets && cd datasets
```

Download the dataset from the [GeoText1652 GitHub repository](https://github.com/MultimodalGeo/GeoText-1652). We recommend using `huggingface-cli`:
```bash
huggingface-cli download --repo-type dataset --resume-download truemanv5666/GeoText1652_Dataset --local-dir GeoText1652_Dataset
```

After downloading, extract all `.tar.gz` files within the `images` directory:
```bash
cd GeoText1652_Dataset/images
find . -type f -name "*.tar.gz" -print0 | xargs -0 -I {} bash -c 'tar -xzf "{}" -C "$(dirname "{}")" && rm "{}"'
cd ../..
```
**Note:** Competition-specific data (e.g., `Track_4_Phase_II_Images` and `PhaseII-queries.txt`) should be downloaded from the official competition website and placed in the correct locations as shown in the directory structure above.

### 2. Pretrained Models

#### X-VLM Model

Download the `16m_base_model_state_step_199999.th` model from [Google Drive](https://drive.google.com/file/d/1iXgITaSbQ1oGPPvGaV0Hlae4QiJG5gx0/view?usp=sharing) and place it in the `pretrain/` directory.

Then, run the script to convert it to an MMCV-compatible format:
```bash
python process_xvlm2mmcv.py \
--input_path pretrain/16m_base_model_state_step_199999.th \
--output_path "pretrain/16m_base_model_state_step_199999_(xvlm2mmcv).pth"
```

#### BERT Model

Download the `bert-base-uncased` model to the `pretrain/bert-base-uncased` directory using `huggingface-cli`:
```bash
huggingface-cli download --resume-download google-bert/bert-base-uncased --local-dir pretrain/bert-base-uncased
```

## Model Training and Testing

### Model Training

We provide scripts to train the model on the GeoText1652 dataset. The default configuration loads data from the `../datasets` path.

- **Single-GPU Training:**
  ```bash
  CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/exp/xvlm_1xb24_hccm_geotext1652.py
  ```

- **Multi-GPU Training (Requires disabling validation):**
  To train on multiple GPUs, please disable periodic validation by modifying `train_cfg` in `configs/exp/xvlm_1xb24_hccm_geotext1652.py`:
  ```python
  # Set val_interval to a value greater than max_epochs
  train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=6, val_interval=10) 
  ```  Then, run the multi-GPU training script (e.g., for 4 GPUs):
  ```bash
  bash tools/dist_train.sh configs/exp/xvlm_1xb24_hccm_geotext1652.py 4
  ```

### Model Testing

We provide an HCCM model checkpoint trained on GeoText1652, which you can download from [here](https://drive.google.com/file/d/1p468glkjTqxuE7YhzdXnC1Kx4xEJQEM3/view?usp=sharing) (`epoch_6.pth`).

Due to GPU memory constraints, we recommend performing the bi-directional evaluation (Text-to-Image and Image-to-Text) in two separate steps.

1.  **Test Image-to-Text (I2T):**
    Modify `configs/exp/xvlm_1xb24_hccm_geotext1652.py` and set `i2t=True` in `test_cfg`.
    ```bash
    CUDA_VISIBLE_DEVICES=0 python tools/test.py \
    configs/exp/xvlm_1xb24_hccm_geotext1652.py \
    epoch_6.pth 
    ```

2.  **Test Text-to-Image (T2I):**
    Modify the config file again, setting `i2t=False` in `test_cfg`.
    ```bash
    CUDA_VISIBLE_DEVICES=0 python tools/test.py \
    configs/exp/xvlm_1xb24_hccm_geotext1652.py \
    epoch_6.pth 
    ```

## Generating the Competition Submission File

After training your model or preparing our provided checkpoint, you can run the `submit.py` script to generate the final submission file for **Phase II of the RoboSense Challenge - Track 4**.

### 1. Configure the Submission Script (`submit.py`)

Before running, open `submit.py` and check the `Settings` class at the top. While most settings can be left as default, please pay close attention to the following:

*   **`data_root`**:
    *   **Description**: The root directory where your datasets are stored.
    *   **Default**: `../datasets`
    *   **Action**: If you have followed the [Directory Structure](#directory-structure) guide, **you do not need to change this**. If you have stored the datasets in a custom location, please update this path accordingly.
    ```python
    # This defaults to the 'datasets' folder in the parent directory.
    # No changes are needed if you followed the guide.
    data_root = "../datasets"
    ```

*   **`checkpoint_file`**:
    *   **Description**: Path to the HCCM model checkpoint file.
    *   **Default**: `"pretrain/epoch_6.pth"`
    *   **Action**: Ensure that you have downloaded the model checkpoint from the provided link and placed it at this exact path.
    ```python
    # Ensure the pretrained checkpoint is downloaded to this path.
    checkpoint_file = "pretrain/epoch_6.pth"
    ```

*   **`batch_size`**:
    *   **Description**: The batch size used during inference.
    *   **Default**: `64`
    *   **Action**: If you encounter an out-of-memory (OOM) error, try reducing this value (e.g., to `32` or `16`). This will increase the total inference time.
    ```python
    # Lower this value if you have limited GPU memory.
    batch_size = 64
    ```

**Tip**: It is recommended to keep other parameters (such as `config_file` and re-ranking `top_k` values) at their default settings to reproduce our competition results.

### 2. Run the Script

Once the configuration is confirmed, execute the following command in your terminal:

```bash
CUDA_VISIBLE_DEVICES=0 python submit.py
```

This will generate the final submission file, `topk256_diver.txt`, in the current directory.


## Acknowledgements

Our work is built upon many excellent open-source projects. We would like to express our sincere gratitude to the developers of the following projects:

- [X-VLM](https://github.com/zengyan-97/X-VLM): A pre-trained model for multi-granularity vision-language alignment.
- [GeoText-1652](https://github.com/MultimodalGeo/GeoText-1652): A benchmark dataset for natural language-guided drones, focusing on spatial relationship matching.
- [MMPretrain](https://github.com/open-mmlab/mmpretrain): An open-source pre-training toolbox from OpenMMLab that provides a rich collection of backbones and pre-trained models.
- [MMCV](https://github.com/open-mmlab/mmcv): A foundational computer vision library from OpenMMLab that provides powerful support for research.
- [MMEngine](https://github.com/open-mmlab/mmengine): A foundational library for training deep learning models from OpenMMLab, which unifies training workflows and interfaces.
