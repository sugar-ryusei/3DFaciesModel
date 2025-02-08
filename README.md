# 3DFaciesModel

*3DFaciesModel* introduces a novel method to construct a 3D facies model of an outcrop using a convolutional neural network (CNN) model. This method is based on <a href="" target="_blank">Sato et al. (2025)</a>.

![](https://github.com/sugar-ryusei/3DFaciesModel/blob/main/figure/facies_models.png)

This approach utilizes a 3D point cloud of an outcrop constructed by drone photogrammetry, which is effective even for large-scale or inaccessible outcrops.

Created by Ryusei Sato, <a href="https://researchmap.jp/k_kikuchi1020" target="_blank">Kazuki Kikuchi</a>, <a href=https://turbidite.secret.jp/>Hajime Naruse</a> from Kyoto University, Japan.


## Usage

*3DFaciesModel* provides a part of the outcrop data to allow testing of the programs.
The data includes the outcrop of a mass-transport deposit exposed in the Upper Cretaceous–Paleocene Akkeshi Formation along the Esashito coast of Hokkaido Island, northern Japan.
To reproduce the 3D facies model, run the Python files in working_directory in this order.

### Translation from 3D Point Cloud to 2D Images
The point cloud exhibiting the outcrop is segmented and translated into a set of 2-D images containing the color and roughness properties of the outcrop surface.
Open3D version 0.8.0.0, a Python package, is utilized to process 3-D point cloud data.
To obtain the 2D images:

    python dataset_generation.py

### Training of CNN model
A U-net-type architecture with residual connections is adopted for the CNN model to conduct semantic segmentation of the outcrop images.
The neural network model was built by Python version 3.9 with TensorFlow version 2.8.2 and Keras version 2.8.0.
To build and train the CNN model:

    python train.py

### Construction of 3D Facies Model
To automatically predict the outcrop facies using the trained CNN model:

    python predict.py

To construct the 3D facies model from predicted 2D labels:

    transcribe.py

To visualize the 3D facies model using Open3D:

    visualize.py
