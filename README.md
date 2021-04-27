# SiamMask Implementation

This repository is an application of the SiamMask object tracking and segmentation algorithm presented in the *Fast Online Object Tracking and Segmentation: A Unifying Approach* paper. We trained and validated the model against different test datasets, and applied it to recycling data.

## Code Execution

The model has two stages of training:

1. Rough tracking.
2. Segmentation mask refinement.

In each stage, we trained the model on the supplied datasets and selected the best model for each. For rough tracking, the best model was the model that had the most amount of frames where the predicted bounding box overlapped with the ground-truth bounding box. The segmentation mask refinement model was selected based on the *intersection-over-union* (IOU) value. We applied this model to the DAVIS test dataset and the recycling data.

### Setup

For general instructions on how to run the code, please go to https://github.com/foolwood/SiamMask. For this project, we have modified the code to run on BU's Shared Computing Cluster (SCC). The instructions below are specific to the SCC.

You need a machine with at least 1 GPU. We trained with 2 V100 GPUs. Testing is possible without a GPU, but it will not achieve real time performance.

On the SCC, load anaconda with 

```module load miniconda```

Initialize the conda enviornment.
```
conda create -n siammask python=3.6
source activate siammask
pip install -r requirements.txt
bash make.sh
export PYTHONPATH=$PWD:$PYTHONPATH
```

### Training 

For both first and second stage training, we use three datasets: Youtube-VOS, ImageNet-VID, and COCO datasets. Youtube-VOS is small. COCO and ImageNet-VID datasets are around 50 GB each. Each of these datasets contain many small files. It is IMPORTANT that you unzip these files in a LOCAL disk and then convert to .tar file for future use. On the SCC, use the scratch disk for this purpose.

Instructions for downloading each of the datasets can be found in the following three folders:

```
data/vid
data/coco
data/ytb_vos
```

modify ```experiments/siammask_base/config.json``` to point to the copy of the data  you have locally (on the scratch disk).

You can use ```copy_data.sh``` to copy data from a file share. It is important to group files into one tar file when copying to and from a file share.

For the first stage of training, download a pre-trainined res-net:

```
cd experiments
wget http://www.robots.ox.ac.uk/~qwang/resnet.model
ls | grep siam | xargs -I {} cp resnet.model {}
```

#### First Stage Training
Then start the first stage of training:
```
experiments/siammask_base/
bash run.sh
```

For us it took 10 hours to train 20 epochs on 2 V100 GPUs. However, it may not be mecessary to train the model for more than ~10 epochs. Please see our validation plot for reference.

-Tensor board here-

Download validation data using ```data/get_test_data.sh```. We used VOT2018 data for validation of first stage model.

Modify and use the bash script ```experiments/siammask_base/test_all_mine.sh``` to validate the first stage models. This bash script generates statistics on how many frames of the validation videos where the predicted boudning box does not overlap with the ground truth bounding box. Choose the model with the lowest number of frames lost.

#### Second Stage

For the second stage of training, we use the Youtube-VOS and COCO datasets. Edit the ```config.json``` file in ```experiements/siammask_sharp/config.json``` to point to the path to the datasets in your local disk. 

Copy your best checkpoint from the first stage model to the ```experiments/siammask_sharp``` directory. The model in the second stage will be initialized with the weights from the first stage model. Run the following to train the second stage model. In our case, the accuracy of the model did not improve significantly on validation data after the firset few epochs of training. So 10 epochs of training should be sufficient.

```
cd experiments/siammask_sharp
bash run.sh checkpoint_e10.pth
```

This stage of training only trains the mask branch of the siammask network, so you should only see the mask loss decrease over training iterations.

(Tensorboard here)

After training, download the DAVIS 2016 or 2017 data. We used the DAVIS2017 training data for validation of the second stage model and the DAVIS2017 validation data for testing of the second stage model.

Use the script here: ```experiments/siammask_sharp/test_all_mine_1.sh``` to generate the IOU metrics for the second stage model checkpoints (for each segmentation mask threshold setting). Pick the model and threshold with the largest average IOU.

You can use the same bash script on the chosen model to generate results on the test data. Our results are included below and in the results directory.

(Put images into results directory).

## Results

### DAVIS Results
Average IOU: *0.643*
Standard Deviation: *0.182*

The standard deviation seems relatively high. In some of the test cases the model performed very well, but in others it performed poorly.

(Note that there is a different measure of segmentation accuracy called F-merasure. This measures the similarity of two contours. We do not report the F-measure results because we do not train the neural network to optimize for this metric. Therefore, the F-measure/contour accuracy is not as relevant to our project. Generally, you should expect our model to perform quite pourly with respect to contour accuracy due to loss of contour detail in segmentation mask.

Below are the results across the 30 deifferent test videos of DAVIS 2017 test set:

![image](https://user-images.githubusercontent.com/17884767/116179448-5e9cfb80-a6e5-11eb-8ddd-f82f0a24d469.png)

### Recycling data
For the recycling data, there was no ground-truth to make a quantitative measurement against. Instead we looked at the segmentation results qualitatively. The model seems to perform decently, but the image blurring from the supplied video seem to negatively affect the tracking. Also the object cluttering in the video.

To select the object to track, we used the supplied GUI object selector from the SiamMask repository.

## Citations

Q. Wang, L. Zhang, L. Bertinetto, W. Hu and P. H. S. Torr, "Fast Online Object Tracking and Segmentation: A Unifying Approach," 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 1328-1338, doi: 10.1109/CVPR.2019.00142.

Q. Wang, L. Zhang, L. Bertinetto (2019) SiamMask [Source code]. https://github.com/foolwood/SiamMask 
