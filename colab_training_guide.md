# YOLOX Custom Training Guide for Google Colab

This guide provides a step-by-step walkthrough for training your custom YOLOX model on Google Colab and exporting it to ONNX format.

## 1. Setting up the Colab Environment

**IMPORTANT: Before proceeding, ensure your Google Colab runtime is set to GPU.**
To do this:
1.  Go to `Runtime` in the top menu.
2.  Select `Change runtime type`.
3.  Under `Hardware accelerator`, choose `GPU`.
4.  Click `Save`.

Training deep learning models like YOLOX on a CPU is extremely slow and not recommended. The code is optimized for NVIDIA GPUs.

Once you have selected the GPU runtime, execute the following commands in a code cell:

```bash
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone your forked YOLOX repository
!git clone https://github.com/fadhilrobbani/YOLOX.git
%cd YOLOX

# Install dependencies
!pip install onnx onnxsim
!pip install -r requirements.txt
!pip install -e .
!pip install cython; pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

## 2. Preparing Your Dataset

The YOLOX `VOCDetection` class expects a specific folder structure for PASCAL VOC datasets. Since your Roboflow dataset is organized with `train`, `test`, and `valid` subfolders containing images and annotations, we need to move these files into the expected YOLOX structure.

### Step 2.1: Upload and Unzip Your Dataset

First, upload your zipped dataset (e.g., `your_dataset.zip`) to your Colab session storage (usually `/content/`). Then, unzip it. **Ensure that after unzipping, your `train`, `test`, and `valid` folders are directly under `/content/datasets/`**.

```bash
# Example: if your zip file is named 'my_dataset.zip'
!unzip /content/my_dataset.zip -d /content/datasets
```

### Step 2.2: Create the YOLOX-Expected VOC Structure

Now, create the necessary PASCAL VOC directory structure *within your cloned YOLOX project*. This is where YOLOX will look for your data.

```bash
# Create the main VOCdevkit and VOC2007 folders within the YOLOX project
!mkdir -p ./datasets/VOCdevkit/VOC2007/ImageSets/Main
!mkdir -p ./datasets/VOCdevkit/VOC2007/Annotations
!mkdir -p ./datasets/VOCdevkit/VOC2007/JPEGImages
```

### Step 2.3: Move Your Dataset Files into the YOLOX Structure

Now, move your actual image (`.jpg`) and annotation (`.xml`) files from where you unzipped them (`/content/datasets/train`, `/content/datasets/test`, `content/datasets/valid`) into the newly created `JPEGImages` and `Annotations` folders.

```bash
# Move all .jpg files from your uploaded 'train', 'test', 'valid' folders
# into the YOLOX-expected JPEGImages directory.
!mv /content/datasets/train/*.jpg ./datasets/VOCdevkit/VOC2007/JPEGImages/
!mv /content/datasets/test/*.jpg ./datasets/VOCdevkit/VOC2007/JPEGImages/
!mv /content/datasets/valid/*.jpg ./datasets/VOCdevkit/VOC2007/JPEGImages/

# Move all .xml files from your uploaded 'train', 'test', 'valid' folders
# into the YOLOX-expected Annotations directory.
!mv /content/datasets/train/*.xml ./datasets/VOCdevkit/VOC2007/Annotations/
!mv /content/datasets/test/*.xml ./datasets/VOCdevkit/VOC2007/Annotations/
!mv /content/datasets/valid/*.xml ./datasets/VOCdevkit/VOC2007/Annotations/
```

### Step 2.4: Generate ImageSet Files

Finally, create the `trainval.txt` and `test.txt` files that list the image IDs. These files are crucial for YOLOX to identify which images belong to which dataset split.

**IMPORTANT:** The commands below use a robust `for` loop with `basename -s .jpg` to ensure that only the correct filename (without the `.jpg` extension) is written to the text files, even if the filenames contain multiple dots. Each line is prefixed with `!` for execution in Colab.

```bash
# Generate trainval.txt (list of all image filenames without extension)
# This iterates through all JPG files, extracts their base filename without the .jpg extension, and writes to trainval.txt.
! ( \
for img_path in ./datasets/VOCdevkit/VOC2007/JPEGImages/*.jpg; do \
  filename=$(basename "$img_path" .jpg); \
  echo "$filename"; \
done \
) > ./datasets/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt

# Generate test.txt (can be the same as trainval.txt if you're not using a separate test set)
# This iterates through all JPG files, extracts their base filename without the .jpg extension, and writes to test.txt.
! ( \
for img_path in ./datasets/VOCdevkit/VOC2007/JPEGImages/*.jpg; do \
  filename=$(basename "$img_path" .jpg); \
  echo "$filename"; \
done \
) > ./datasets/VOCdevkit/VOC2007/ImageSets/Main/test.txt
```

Your dataset is now correctly structured for YOLOX.

## 3. Running the Training

Now you are ready to start the training process. Execute the following command in a code cell.

```bash
# Start training
!python3 tools/train.py -f exps/example/custom_voc/yolox_voc_m.py -d 1 -b 8 --fp16 -c weights/yolox_m.pth
```

## 4. Exporting to ONNX

After the training is complete, you can export the trained model to ONNX format. The trained weights will be saved in the `YOLOX_outputs` directory.

```bash
# Export to ONNX
!python3 tools/export_onnx.py --output-name cup_yolox.onnx -f exps/example/custom_voc/yolox_voc_m.py -c "/content/drive/MyDrive/Colab Notebooks/trained_models/cup-detection-yolox/yolox_voc_m/best_ckpt.pth"
```

This will create a `cup_yolox.onnx` file in your `YOLOX` directory, which you can then download and use for inference.