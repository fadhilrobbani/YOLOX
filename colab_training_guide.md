# YOLOX Custom Training Guide for Google Colab

This guide provides a step-by-step walkthrough for training your custom YOLOX model on Google Colab and exporting it to ONNX format.

## 1. Setting up the Colab Environment

First, open a new Colab notebook and set the runtime to use a GPU (Runtime -> Change runtime type -> GPU). Then, execute the following commands in a code cell:

```bash
# Clone your forked YOLOX repository
!git clone https://github.com/fadhilrobbani/YOLOX.git
%cd YOLOX

# Install dependencies (with updated requirements)
!pip install -r requirements.txt
!pip install -e .
!pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

## 2. Uploading Your Dataset

You will need to upload your PASCAL VOC formatted dataset to your Colab environment. You can do this by:

*   **Uploading from your local machine**: Use the file browser in the left-hand sidebar to upload your dataset.
*   **Mounting Google Drive**: Mount your Google Drive and place your dataset there.

Once uploaded, create a symbolic link to the dataset directory:

```bash
# Create a symbolic link to your dataset
# Make sure to replace /path/to/your/VOCdevkit with the actual path
!ln -s /path/to/your/VOCdevkit ./datasets/VOCdevkit
```

## 3. Running the Training

Now you are ready to start the training process. Execute the following command in a code cell. This command uses the custom experiment file we created.

```bash
# Start training
!python3 tools/train.py -f exps/example/custom_voc/yolox_voc_m.py -d 1 -b 8 --fp16 -c weights/yolox_m.pth
```

## 4. Exporting to ONNX

After the training is complete, you can export the trained model to ONNX format. The trained weights will be saved in the `YOLOX_outputs` directory.

```bash
# Export to ONNX
!python3 tools/export_onnx.py --output-name cup_yolox.onnx -f exps/example/custom_voc/yolox_voc_m.py -c YOLOX_outputs/yolox_voc_m/best_ckpt.pth
```

This will create a `cup_yolox.onnx` file in your `YOLOX` directory, which you can then download and use for inference.