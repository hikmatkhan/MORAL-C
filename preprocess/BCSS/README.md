## 🧩 BCSS Data Processing (Adapted from WILDS Package)

This preprocessing pipeline has been adapted from the [WILDS repository](https://github.com/p-lambda/wilds), specifically from their [CAMELYON17 preprocessing code](https://github.com/p-lambda/wilds/tree/main/dataset_preprocessing/camelyon17). It has been modified to work with the [BCSS dataset](https://github.com/PathologyDataScience/BCSS) to extract class-balanced patches from whole slide images (WSIs).

---

### 📦 Requirements

- Make sure OpenCV is installed **before** installing the Python wrapper listed in `requirements.txt`.
- Official OpenCV installation instructions: [OpenCV Docs](https://docs.opencv.org/4.x/df/d65/tutorial_table_of_content_introduction.html)

---

### ⚙️ Preprocessing Instructions

#### 0️⃣ Download and Prepare Dataset

1. Download the BCSS dataset from [here](https://github.com/PathologyDataScience/BCSS).
2. Only extract the following folders:
   - `masks`
   - `rgbs_colorNormalized`
3. Rename `rgbs_colorNormalized` to `images`.

Your `<SLIDE_ROOT>` directory should now contain:


<pre>  
   SLIDE_ROOT/ 
      ├── masks/
      └── images/
</pre>

---

#### 1️⃣ Generate Patch Coordinates

Run the script to generate potential patch coordinates:

```bash
python generate_all_patch_coords.py --slide_root <SLIDE_ROOT> --output_root <OUTPUT_ROOT>
```
- <SLIDE_ROOT>: Path to the dataset with images/ and masks/

- <OUTPUT_ROOT>: Directory where patch coordinates and outputs will be stored

- Note: If you have metadata.csv already, you can skip this step.

#### 2️⃣ Select Balanced Patches
Create a class-balanced selection of patches using:
```bash
python generate_final_metadata.py --output_root <OUTPUT_ROOT>
```
Note: This step can also be skipped if you're using a provided metadata.csv.

#### 3️⃣ Extract Patches to Disk
Extract image patches and save them locally using the filtered metadata:
```bash
python extract_final_patches_to_disk.py --slide_root <SLIDE_ROOT> --output_root <OUTPUT_ROOT>
```
This will crop the patches from the whole slide images and save them to disk.

After running all steps, your <OUTPUT_ROOT> directory will contain:

<pre>  
   OUTPUT_ROOT/ 
      ├── metadata.csv 
      └── patches/ 
         ├── 0001.png 
         ├── 0002.png 
         └── ... 
</pre>

These patches can now be used for training and evaluation in your classification or segmentation pipelines.



## BCSS processing using code from WILDS package
The code here has been taken from the [WILDS package](https://github.com/p-lambda/wilds/tree/main). In particular from [this subdirectory](https://github.com/p-lambda/wilds/tree/main/dataset_preprocessing/camelyon17).


### Requirements
OpenCV must be installed before you install its corresponding Python wrapper listed in the requirements.txt file.

See installation instructions for OpenCV [here](https://docs.opencv.org/4.x/df/d65/tutorial_table_of_content_introduction.html).

### Instructions

0. Download the data from https://github.com/PathologyDataScience/BCSS in your data directory (`--slide_root` argument you'll pass to files). You only need to download the directories called `masks` and `rgbs_colorNormalized`. Rename `rgbs_colorNormalized` to `images`. Your `<SLIDE_ROOT>` directory should contain 2 subdirectories called `masks` and `images`.


1. Run `python generate_all_patch_coords.py --slide_root <SLIDE_ROOT> --output_root <OUTPUT_ROOT>` to generate a .csv of all potential patches. `<OUTPUT_ROOT>` is wherever you would like the patches to eventually be written. You can also skip this step and the next one if you simply use the `metadata.csv` provided here.


2. Then run `python generate_final_metadata.py --output_root <OUTPUT_ROOT>` to select a class-balanced set of patches. You can also skip this step and the previous one if you simply use the `metadata.csv` provided here.


3. Finally, run `python extract_final_patches_to_disk.py --slide_root <SLIDE_ROOT> --output_root <OUTPUT_ROOT>` to extract the chosen patches from the WSIs and write them to disk.
