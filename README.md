# Whole Body Positron Emission Tomography Attenuation Correction Map Synthesizing using 3D Deep Networks

This repository contains the code to replicate the results reported in [the publication (currently under review)](https://www.researchsquare.com/article/rs-46953/v1). 
It also includes additional topologies reported in the -Phd. thesis- (TO BE RELEASED).

The trained models used to produce the reported results are provided [here](https://mega.nz/file/q0t1mR5K#Omg0UfYREqVqnZo_7SWSgWY7W8uwhiv6gZlwdVnZnXo). They should be decompressed in the root of the repository.


# Table of Contents
1. [Overview](#Overview)
2. [Code Description](#Code_Description)

# Overview

The correction of attenuation effects in Positron Emission Tomography (PET) imaging is fundamental to obtain a correct radiotracer distribution. However direct measurement of this attenuation map is not error-free and normally results in additional ionization radiation dose to the patient.

This model obtains the whole body attenuation map using a 3D U-Net generative adversarial network. The network is trained to learn the mapping from non attenuation corrected 18 F-fluorodeoxyglucose PET images to a synthetic Computerized Tomography (sCT) and also to label the input voxel tissue. The sCT image is further refined using an adversarial training scheme to recover higher frequency details and lost structures using context information. 

This work is trained and tested on public available datasets, containing several PET images from different scanners with different radiotracer administration and reconstruction modalities. 


### Results

The 3D-Unet and GAN topologies were tested on 133 samples from 8 distinct datasets. The resulting mean absolute error of the network is 103 ± 18 HU and a peak signal to noise ratio of 18.6 ± 1.5 dB. 

<img src="./images/result_mosaique_AMC-009.png" width="800">

The image shows the coronal and sagital central cuts of the *AMC-009* sample of the *NSCLC Radiogenomics* test dataset. Each colum represents the Non-Attenuation Correct PET (NAC-PET) input, the sCT of the 3D U-Net, the sCT from the GAN refined 3D U-Unet and the objective CT or ground truth, respectively.
The bone tissue is presented using a 3D projection in the following image:

<img src="./images/result_mosaique_AMC-009_3D.png" width="600">

It can be seen that the proposed method recovers good quality attenuation maps from unseen data from different scanners,
patients and lesions, showing that the technique can be used on multiple sources. Also, the method performs with accuracy comparable to other methods showing their fitting for PET image correction. 






### 3D U-Net and GAN topologies

The network topology is composed of a 3D U-Net generator and a convolutional critic (or discriminator). An additional segmentation branch is used to regularize
the training. Nevertheless, the adversarial loss gradient flow is limited to the last part of the network. 

<img src="./images/gen_network_topology.png" width="800">

Top branch is used for label segmentation (auxiliary task) and bottom branch for artificial CT generation. The bottom output is the synthetic CT refined by
the GAN layers and built from the 3D U-Net outputs. All convolutional operations use a filter of size 3×3×3 except the output layer which use a 1×1×1 filter. The network posses 5 resolution levels, each of them composed of two convolutional layers with filter shape of 3×3×3 and Rectified Linear Unit (ReLU) activation. Instead of convolutional resampling the resolution changes are performed using trilinear up- or down-sampling. After each convolutional layer, voxel normalization along feature maps is applied. Also, at each convolutional layer, a scaling factor based on He’s is applied to the filter kernel.


The discriminator or critic is a 3D convolutional network:

<img src="./images/critic_network_topology.png" width="600">

The critic or discriminator network is a fully convolutional network with ReLU activation in all layers, only the last layer has no activation. The input of this network is a two channel volume composed of the NAC-PET volume and the real or sCT image. The output of the network is a value proportional to quality value of the  generated image. The network is conformed by 4 resolution levels with two convolutional layers per level. Each convolution has a filter size of 3 × 3 × 3 and ReLU  activation. No batch or pixel normalization is applied. The last two layers of the critic are a flatten operation followed by a single dense layer with linear output.



### Progressive Growing GAN

The progressive growing GAN (ProGAN) is based on a 3D-Unet. It's topology is similar to the GAN. The main difference is the training procedure. The ProGAN is trained from low resolutions and it is incrementally expanded until reaching the desired resolution. The generator topology can be described as:

<img src="./images/prog_gen_network_topology.png" width="800">

and the discriminator or critic:

<img src="./images/prog_critic_network_topology.png" width="800">

Each coloured section is a bock that grows over the previous structure. The neural network (NN) starts with the structure shown in red, once this resolution level is trained this connection is eliminated and the next resolution level is connected (the intermediate resolution structure is shown in green). This process continues until the final resolution structure is added which shields the desired resolution. The blocks marked with *Aux.* are only used during the training of its specific resolution and then discarded.




# Code_Description

### Requirements

The training of the model requires:
* CPU RAM >= 16 GB
* GPU RAM >= 6 GB
* CUDA CAPABILITY > 3.6

The training times will largely depend on the GPU and CPU speed. A solid state storage unit is advised but not required.
To run the test notebooks a GPU is not required, but will speed up the process. Testing requires at least 8GB CPU RAM.
A docker image is provided to run the code, [here](https://hub.docker.com/r/rawthil/iar-pet), and then executing the jupyter notebook server with:

<p>docker run -u $UID:$UID --gpus all -v /PATH/TO/REPOSITORY:/tf/iAR-PET -p 5000:8888 -it ar-pet jupyter notebook --ip=0.0.0.0 --allow-root</p>

and opening on the browser: [http://localhost:5000/tree/](http://localhost:5000/tree/)


Alternatively, the code can be run in an environment with python 3.6.8, jupyter notebook support and the following libraries:

* dataclasses==0.7
* dicom==0.9.9.post1
* graphviz==0.14
* h5py==2.10.0
* ipykernel==5.1.2
* ipython==7.8.0
* joblib==0.14.1
* jupyter==1.0.0
* jupyter-client==5.3.3
* jupyter-core==4.5.0
* Keras-Applications==1.0.8
* Keras-Preprocessing==1.1.0
* matplotlib==3.1.1
* MedPy==0.4.0
* notebook==6.0.1
* numpy==1.17.2
* pandas==0.25.3
* pickleshare==0.7.5
* pydicom==1.3.0
* pydot==1.4.1
* scikit-image==0.16.2
* scipy==1.4.1
* seaborn==0.9.0
* SimpleITK==1.2.4
* six==1.11.0
* tensorboard==2.0.0
* tensorflow-estimator==2.0.0
* tensorflow-gpu==2.0.0





### Datasets downloading

The work is based on a public dataset from [The Cancer Image Archive (TCIA)](https://nbia.cancerimagingarchive.net). Within *./datasets/DICOM/* you will find a ".csv" file and a ".tcia" for each data source:

* HNSCC_TrainValidation
* CPTAC-LSCC
* CPTAC-LUAD
* CPTAC-PDA
* CPTAC-UCEC
* NSCLC_Radiogenomics
* TCGA-HNSC
* TCGA-LUAD
* TCGA-THCA

The data can be downloaded opening the ".tcia" file using the [NBIADataRetriever](https://wiki.cancerimagingarchive.net/display/NBIA/Cancer+Imaging+Archive+User's+Guide#CancerImagingArchiveUser'sGuide-DownloadingtheNBIADataRetriever). The data must be downloaded into the *./datasets/DICOM/* folder with the "Descriptive Directory Name" option. The ".tcia" file was tested using the NBIADataRetriever v3.6 build "2020-04-03_15-44".  The ".csv" contains the same information as the ".tcia" file. 

After a successful download the directory should look like this:

* *./datasets/DICOM/HNSCC/HNSCC-01-0004/...*
* *./datasets/DICOM/HNSCC/HNSCC-01-0008/...*
* *...*
* *./datasets/DICOM/TCGA-THCA/TCGA-DE-A4MA/...*

### TFDataset file generation (Train and Validation datasets)

In the *./datasets/DICOM/* folder the *Generate_train_and_validation_dataset* notebook is found. It must be executed to generate the training files. This process is very long, it takes approx. 5 hs to run on a 4x3.6Ghz CPU. It will produce a large output, approx. 120 GB if using the largest resolution of 256x256x512 voxels. The process support other datasets or larger selections of the HNSCC dataset (the dataset is updated from time to time). It is advised to inspect the output to find any defective sample if using samples from outside the provided list.
The output is a series of ".tfrecord" files, for different data resolution:

* *./datasets/Validation_Dataset_256x256x512.tfrecord*
* *./datasets/Train_Dataset_256x256x512.tfrecord*
* *./datasets/Validation_Dataset_128x128x256.tfrecord*
* *./datasets/Train_Dataset_128x128x256.tfrecord*
* *...*

## Training

The training of the different models is done using three specific notebooks:

1. *./Train_3D_Unet_Model.ipynb*: Basic 3D U-Net topology (Sup. Pre-Train No GAN.Loss.) and pre-training of GAN based generators.
2. *./Train_GAN_Model.ipynb*: Trains GAN based models, such as (see the paper for reference): 
    * Baseline
    * Sup. Pre-Train
    * Sup. No Pre-Train
    * Sup. Pre-Train RestrictedGrad.
3. *./Train_ProGAN_Model.ipynb*: Trains the progressive growing 3D pix2pix model described in the thesis.

The notebooks are prepared to use multiple GPUs as the memory requirements of the model is large. However it should work with single GPUs set-ups (if the GPU memory is larger than 6 GB).


## Testing

To compute the testing metrics the *./Test_Dataset.ipynb* notebook should be executed. The notebook tests a selected model using a selected test dataset. The model and dataset selection is done inside the notebook.
After execution it will produce the following outputs under *./test_outputs/SELECTED_MODEL/SELECTED_DATASET/*:


* Input, synthetic CT and objective CT images of a random sample (NAC_PET_input.png, sCT.png and objective_CT.png). If the model also computes segmentation of tissues it will create the segmentation image (Tissue_Clases.png)
* A matrix containing the sCTs of each processed sample (CT_SYNTH_images.npy) and the segmentation image (SEGMENTED_images.npy), if available.
* A Pandas dataframe (meassurements_dataframe.pkl) containing the following metrics:
    * SSIN:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Structural Similarity Index on the whole Field of View (FoV).
    * MSE_Whole:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Mean Squared Error on the whole FoV.
    * MSE_RoI:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Mean Squared Error on the occupied voxels.
    * PSNR_Whole:&nbsp;Peak Signal to Noise Ratio on the whole FoV.
    * PSNR_RoI:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Peak Signal to Noise Ratio on the occupied voxels.
    * MAE:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Mean Absolute Error on the whole FoV.
    * MAE_HU:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Mean Absolute Error on the whole FoV using Hounsfield Units.
    * NMSE_Whole:&nbsp;Normalized Mean Squared Error on the whole FoV.
    * NMSE_RoI:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Normalized Mean Squared Error on the occupied voxels.
    * NCC_Whole:&nbsp;&nbsp;&nbsp;&nbsp;Normalized Cross Correlation on the whole FoV.
    * NCC_RoI:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Normalized Cross Correlation on the occupied voxels.
* Each processed sample in DICOM format, for inspection purposes only.










