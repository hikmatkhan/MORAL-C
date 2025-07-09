## 🧩 OCELOT Dataset Preprocessing (Adapted from WILDS Package)

This preprocessing workflow is adapted from the [WILDS repository](https://github.com/p-lambda/wilds), specifically their [CAMELYON17 dataset preparation scripts](https://github.com/p-lambda/wilds/tree/main/dataset_preprocessing/camelyon17). The process has been extended to support the OCELOT dataset, which includes annotated cancer pathology images across multiple organs.

---

### 📦 Requirements

Ensure the following dependencies are installed **before** running the Python scripts:

- **OpenCV**: For image processing  
  Installation guide: [OpenCV Docs](https://docs.opencv.org/4.x/df/d65/tutorial_table_of_content_introduction.html)

- **OpenSlide**: For handling `.svs` whole slide image files  
  Installation guide: [OpenSlide-Python](https://github.com/openslide/openslide-python)

After installing these system libraries, install the Python dependencies listed in `requirements.txt`.

---

### 📥 Dataset Download & Preparation

Due to the fact that OCELOT provides annotations at a resolution lower than 40x, extra steps are required to align them with high-resolution WSIs.

#### 0️⃣ Download Dataset & Annotations

- Download the OCELOT dataset from the official website:  
  https://ocelot2023.grand-challenge.org/datasets/  
  (Tested with version **1.0.1**)

#### 1️⃣ Download Original WSIs from TCGA

- Extract patient and image identifiers from `metadata.json` (provided with OCELOT).
- Use this information to create a GDC manifest file (`gdc_manifest.txt` is already provided).
- Download WSIs using the **GDC Data Transfer Tool** ([Docs here](https://docs.gdc.cancer.gov/Data_Transfer_Tool/Users_Guide/Getting_Started/)):

```bash
gdc-client download -m gdc_manifest.txt
```
After downloading, move all .svs files into a directory named wsis inside your <SLIDE_ROOT> folder.

#### 🖼️ Extract Initial OCELOT Patches
Run the script below to extract patches from TCGA WSIs that correspond to the OCELOT annotations:
```bash
python extract_initial_patches.py
```
Make sure to update the paths inside the __main__ block of the script to point to:

- metadata.json (provided)
- wsis/ folder (contains downloaded TCGA images)

### 🧪 Patch Processing (Same as BCSS or CAMELYON17)
Once patches are extracted, continue as follows:
#### 1️⃣ Generate All Potential Patch Coordinates
```bash
python generate_all_patch_coords.py --slide_root <SLIDE_ROOT> --output_root <OUTPUT_ROOT>
```
This will generate a .csv file with all possible patch locations.

✅ You can skip this step if you're using the pre-generated metadata.csv.

#### 2️⃣ Create Balanced Metadata
```bash
python generate_final_metadata.py --output_root <OUTPUT_ROOT>
```
Selects a class-balanced subset of patches for training and evaluation.

✅ This step is also optional if using the provided metadata.csv.

### 3️⃣ Extract Final Patch Images
```bash
python extract_final_patches_to_disk.py --slide_root <SLIDE_ROOT> --output_root <OUTPUT_ROOT>
```
This extracts final image crops from WSIs using the filtered metadata.
