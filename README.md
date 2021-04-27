# SiamMask Implementation

This repository is an application of the SiamMask object tracking algorithm presented in the *Fast Online Object Tracking and Segmentation: A Unifying Approach* paper. We trained and validated the model against different test datasets, and applied it to recycling data.

## Code Execution
** *NEED HELP ON THIS ONE* **

The model has two stages of training:

1. Rough tracking
2. Segmentation mask refinement

In each stage, we trained the model on the supplied data sets and selected the best model for each. For rough tracking, the best model was the model that had the most amount of frames where the predicted bounding box overlapped with the ground-truth bounding box. The segmentation mask refinement model was selected based on the *intersection-over-union* (IOU) value. We applied this model to the DAVIS test dataset and the recycling data.

## Results

### DAVIS Results
Average IOU: 0.643
Standard Deviation: 0.182

The standard deviation seems relatively high. In some of the test cases the model performed very well, but in others it performed poorly.

Below are the results of the different test sets:

![image](https://user-images.githubusercontent.com/17884767/116179448-5e9cfb80-a6e5-11eb-8ddd-f82f0a24d469.png)

### Recycling data
For the recycling data, there was no ground-truth to make a quantitative measurement against. Instead we looked at the segmentation results qualitatively. The model seems to perform decently, but the image blurring from the supplied video seem to negatively affect the tracking.

To select the object to track, we used the supplied GUI object selector from the SiamMask repository.

## Citations

Q. Wang, L. Zhang, L. Bertinetto, W. Hu and P. H. S. Torr, "Fast Online Object Tracking and Segmentation: A Unifying Approach," 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 1328-1338, doi: 10.1109/CVPR.2019.00142.

Q. Wang, L. Zhang, L. Bertinetto (2019) SiamMask [Source code]. https://github.com/foolwood/SiamMask 
