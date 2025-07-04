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

### 🔸 Option 2: Using Conda

#### Step 1: Clone the Repository

```bash
git clone https://github.com/hikmatkhan/MORPH-GEN.git
cd MORPH-GEN
```

#### Step 2: Create and Activate the Conda Environment  
(You can change `morphgen_env` to any name you prefer.)

```bash
conda create --name morphgen_env python=3.8
conda activate morphgen_env
```

#### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---



## Training:
1. Training on CAMELYON Dataset. (Ensure that you have download and preprocess the dataset)
2. To train the model on a specific hospital data, run the following command with the specific hospital index (must be 0, 1, 2, 3, or 4).
  ```bash
python train.py --config_path PATH_TO_CONFIG --hospital HOSPITAL_ID --data_dir PATH_TO_DATA
```
3. Logs will be saved to: training_log.txt
4. Best model checkpoint will be saved to: ckpts/resnet50/0.pth (if hospital 0)
## Complete code is coming soon!
