# MORPH-GEN

**MORPH-GEN: Morphology-Guided Representation Alignment for Enhanced Single-Domain Generalization in Histopathological Cancer Classification**

---

## 📂 Dataset

Details coming soon...

---

## 🛠️ Installation

Set up a virtual environment and install dependencies from `requirements.txt` using either Python's built-in `venv` or Conda.

---

### 🔹 Option 1: Using `venv` (Python Standard Virtual Environment)

#### Step 1: Clone the Repository

```bash
git clone https://github.com/hikmatkhan/MORPH-GEN.git
cd MORPH-GEN
```

#### Step 2: Create and Activate the Virtual Environment  
(You can change `morphgen_env` to any name you prefer.)

For **Windows**:

```bash
python -m venv morphgen_env
morphgen_env\Scripts\activate
```

For **macOS/Linux**:

```bash
python3 -m venv morphgen_env
source morphgen_env/bin/activate
```

#### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```
---
### 🔸 Option 2: Using Conda

#### Step 1: Clone the Repository

```bash
git clone https://github.com/hikmatkhan/MORPH-GEN.git
cd MORPH-GEN
```

#### Step 2: Create and Activate the Conda Environment  
(You can change `morphgen_env` to any name you prefer.)

```bash
conda create --name morphgen_env python=3.9
conda activate morphgen_env
```

#### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---



## 🧠 Training:
1. Training on CAMELYON Dataset. (Ensure that you have download and preprocess the dataset)
2. To train the model on a specific hospital data, run the following command with the specific hospital index (must be 0, 1, 2, 3, or 4).
  ```bash
python train.py --config_path PATH_TO_CONFIG --hospital HOSPITAL_ID --data_dir PATH_TO_DATA
```
3. Logs will be saved to: training_log.txt
4. Best model checkpoint will be saved to: ckpts/resnet50/0.pth (if hospital 0)

## 🧪 Testing:
### Testing on CAMELYON Dataset
This section explains how to evaluate trained models on the [CAMELYON17](https://camelyon17.grand-challenge.org/) dataset using the provided evaluation script.

Use the following command to run evaluation on all models:
```bash
python test.py --ckpts_path checkpoints_directory_path --data_dir complete_camelyon_data
```
This script will:

- Load each model checkpoint (e.g., 0.pth, 1.pth, etc.)

- Evaluate the model on validation or all splits for hospitals 0 to 4

- Print accuracy and confusion matrix per hospital

###  Testing on BCSS Dataset
This section explains how to evaluate trained models on the BCSS dataset using the provided evaluation script.

Use the following command to run evaluation on all models:
```bash
python test_bcss.py --ckpts_path ./ckpts --image_folder ./preprocessed_BCSS_data_path

```
This script will:

- Load each model checkpoint (e.g., 0.pth, 1.pth, etc.)

- Evaluate the model on BCSS dataset

- Print accuracy and confusion matrix

### Testing on OCELOT Dataset
This script evaluates trained models on the [OCELOT dataset](https://zenodo.org/record/10621217) across multiple organs (bladder, endometrium, head-and-neck, kidney, prostate, stomach).

```bash
python test_ocelot.py --ckpts_path ckpts --image_folder OCELOT_data
```
This will automatically loop through all six organ types:

- bladder

- endometrium

- head-and-neck

- kidney

- prostate

- stomach
---

## 🔊 Robustness Evaluation with Image Corruptions (CAMELYON17)
This script (`test_noise.py`) evaluates the robustness of trained models on the **CAMELYON17 dataset** under various types of noise corruptions applied during inference.

The noise types and corruption process are adapted using the `corrupt()` function from the [ImageNet-C benchmark](https://github.com/hendrycks/robustness) to simulate real-world degradation such as blur, compression, and sensor noise.

### 🧪 Objective

To assess how well models generalize under different visual corruptions during inference time.  
This is important for evaluating **robustness in clinical deployment**, where imaging artifacts may exist.

### 💾 Noise Types Applied

The following types of corruptions are applied to each input image before passing them to the model:

- `shot_noise`
- `gaussian_noise`
- `impulse_noise`
- `defocus_blur`
- `jpeg_compression`
- `motion_blur`
- `snow`
- `elastic_transform`

### 🔧 Severity Levels

The severity of each noise is controlled using the `severity` parameter in the `corrupt()` function.

In the paper, we evaluated model robustness at **two corruption levels**:

- `severity=1` (low corruption)
- `severity=2` (moderate corruption)

You can change this in the code (File: test_noise.py):

```python
x_np_corrupted = np.array([
    corrupt(img, severity=2, corruption_name=c_name) for img in x_np
])
```
To test at severity level 1, simply change severity=2 to severity=1

This script (test_noise.py) will:

- Load each model checkpoint (e.g., 0.pth, 1.pth, ...)

- Loop through each predefined noise type

- Apply noise to every input batch using corrupt(...)

- Evaluate performance (accuracy + confusion matrix) on hospitals 0 to 4

▶️ Running the Script

```bash
python test_noise.py --ckpts_path ckpts --data_dir dataset
```
---

## 🧠 Visualizing Model Interpretability using Saliency Maps

This guide explains how to generate **Integrated Gradients-based saliency maps** for tumor classification using a ResNet50 model trained on the CAMELYON17 dataset. The script (`saliency_maps.py`) produces visual explanations that highlight the **positive (tumor evidence)** and **negative (non-tumor evidence)** regions in pathology images.

---

### 📂 File: `saliency_maps.py`

This script uses [Captum](https://captum.ai/)’s **Integrated Gradients** method to visualize how different parts of an input image contribute to the model's prediction.

---

### 🧪 Purpose

- To understand what regions in a pathology image the model focuses on when predicting tumor presence.
- Provides **green-highlighted areas** for positive contributions (tumor evidence) and **red-highlighted areas** for negative contributions (non-tumor evidence).

###🧾 How It Works
1. Loads a trained ResNet50 model and its checkpoint.

2. Reads all .png images from a specified folder.

3. For each image:

      -- Resizes and normalizes it for inference.

      -- Applies Integrated Gradients using Captum.

      -- Generates and overlays saliency maps on a grayscale background.

Saves the original image, the positive map (green), and the negative map (red) side-by-side.
---

## Complete code is coming soon!
