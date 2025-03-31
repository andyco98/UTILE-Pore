# *UTILE-Pore* - Deep learning-enabled 3D analysis of porous materials in polymer electrolyte membrane technologies 

![](https://github.com/andyco98/UTILE-Pore/blob/main/images/method_comp_scale.jpg)


We present  an automated workflow using deep learning for the analysis of porous-like structures from tomographs related to PEM technologies by: 1. preparing annotated dataset and training models in order to conduct semantic seg- mentation of pores and 2. automating the extraction of pore properties for further distribution analysis.


The publication [UTILE-Pore - Deep learning-enabled 3D analysis of porous materials in polymer electrolyte membrane technologies]() will be available soon!


## Description
This project focuses on the deep learning-based automatic analysis of Vanadium Redox Flow Batteries (VRFB) Synchrotron X-ray tomographies. 
This repository contains the Python implementation of the UTILE-Redox software for automatic volume analysis, feature extraction, and visualization of the results.

The models we present in this work are trained on a specific use-case scenario of interest in VRFB bubble tomographies. Nevertheless, it is possible to fine-tune, re-train or employ another model suitable for your individual case if your data has a strong visual deviation from the presented data here, which was recorded and shown as follows:

![](https://github.com/andyco98/UTILE-Pore/blob/main/images/workflowpore1.jpg)

## Model's benchmark
In our study, we trained several models to compare their prediction performance on unseen data. We trained specifically four different models on the same dataset composed by :

- 3D U-Net with a ResNeXt 101 backbone 
- 3D V-Net
- 3D Swin U-Net

We obtained the following performance results for the binary case of GDLs and CLs:

| Model                             | Precision [%]  | Recall [%] | F1-Score [%] | Porosity difference [%] |
|-----------------------------------|----------------|------------|--------------|-------------------------|
| 3D U-Net with ResNeXt101 backbone | 91             | 89         | 90           | 1.7                     |
| 3D V-Net                          | 89             | 91         | 91           | 3.9                     |
| Swin U-Net                        | 83             | 88         | 85           | 2.4                     |
| Otsu Thresholding                 | -              | -          | -            | 5                       |

And we obtained the following performance results for the ternary case of GDLs with MPL:

| Model                             | Precision [%]  | Recall [%] | F1-Score [%] | Porosity difference [%] |
|-----------------------------------|----------------|------------|--------------|-------------------------|
| 3D U-Net with ResNeXt101 backbone | 83             | 92         | 88           | 9.8                     |
| 3D V-Net                          | 88             | 90         | 89           | 8.4                     |
| Swin U-Net                        | 85             | 90         | 88           | 3.3                     |

A visual inspection was carried out to find the best-performing model:

![](https://github.com/andyco98/UTILE-Pore/blob/main/images/pore_comp.png)


## Extracted features
### Binary segmented tomographies quantifications
![](https://github.com/andyco98/UTILE-Pore/blob/main/images/binary_res.jpg)

### Ternary segmented tomographies of GDLs with MPL quantifications
![](https://github.com/andyco98/UTILE-Pore/blob/main/images/mpl_analysis.jpg)

### Installation
- Create an Anaconda environment with Python 3.9
- Run this line: 
    - pip install notebook numpy==1.23.2 scikit-image tensorflow==2.8.0 protobuf==3.20 porespy pypardiso opencv-python vtk 
- Follow the instructions in the Jupyter Notbook file in UTILE-Pore analysis