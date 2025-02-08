# 3DFaciesModel

*3DFaciesModel* introduces a novel method to reconstruct a 3D facies model of an outcrop using a convolutional neural network (CNN) model. This method is based on <a href="" target="_blank">Sato et al. (2025)</a>.

![](https://github.com/sugar-ryusei/3DFaciesModel/blob/main/figure/facies_models.png)

This approach utilizes a 3D point cloud of an outcrop constructed by drone photogrammetry, which is effective even for large-scale or inaccessible outcrops.

Developed by Ryusei Sato, <a href="https://researchmap.jp/k_kikuchi1020" target="_blank">Kazuki Kikuchi</a>, <a href=https://turbidite.secret.jp/>Hajime Naruse</a> from Kyoto University, Japan.


## Usage

*3DFaciesModel* includes sample outcrop data to allow testing of the program.

The data includes the outcrop of a mass-transport deposit exposed in the Upper Cretaceous–Paleocene Akkeshi Formation along the Esashito coast of Hokkaido Island, northern Japan.

To reproduce the 3D facies model, run the Python files in `working_directory` folder in this order.

### 1. Translate 3D Point Cloud to 2D Images
The point cloud of the outcrop is segmented and translated into a set of 2-D images that capture the color and roughness properties of the outcrop surface.
This process uses Open3D (version 0.8.0.0) for handling 3D point cloud data.

Run the following command to generate 2D images:

    python dataset_generation.py

### 2. Train the CNN model
A U-Net-based CNN with residual connections is used for semantic segmentation of the outcrop images.
The model is implemented using Python version 3.9 with TensorFlow version 2.8.2 and Keras version 2.8.0.

To build and train the CNN model, execute:

    python train.py

### 3. Construct the 3D Facies Model
Predict facies using the trained CNN model:

    python predict.py

Construct the 3D facies model from predicted 2D labels:

    python transcribe.py

Visualize the 3D facies model using Open3D:

    python visualize.py
