# Solution of Team IMR for MICCAI 2023 LNQ Challenge

***

**Transfer Mapping for Clinically Relevant Lymph Nodes in The Mediastinal Area of CT data**  
*Hanxiao Zhang, Minghui Zhang, Xin You, Zhebing Lin, Yi Zhang, Liu Liu, Xinghua Cheng, Yun Gu and Guang-Zhong Yang*

Built upon [nnUNet](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1/), this repository provides the solution of Team IMR for [MICCAI 2023 LNQ Challenge](https://lnq2023.grand-challenge.org/lnq2023/). This repository contains the training method and inference code for diseased and mediastinal lymph node segmentation using contrast-enhanced CT scans.

***




## Additional data
In that [LNQ training data](https://lnq2023.grand-challenge.org/data/) are partially annotated (i.e. one node out of five) and it is suboptimal to use straight off the shelf for fully supervised pixel-wise segmentation, this solution leverages two additional data with full mediastinal lymph node annotations in the training process. Both additional data are publicly available. Totally 102 CT volumes (87 CTs + 15 CTs) with full annotations were employed.

**TCIA data with refined annotations**  
[Original TCIA data](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=19726546#19726546fcb14b04d2494090ab696ba899c8d70c/) consists of 90 contrast-enhanced CTs of the mediastinum with lymph node position labels marked by radiologists at the National Institutes of Health ([Roth et al. 2014](https://link.springer.com/chapter/10.1007/978-3-319-10404-1_65)). Based on these 90 CTs, [Self et al. 2015](https://link.springer.com/chapter/10.1007/978-3-319-24571-3_7) independently produced lymph node mask annotations examined by a board-certified radiologist. However, these annotations are often sparse and not available with a complete set of mediastinal lymph nodes (e.g., some small but visible lymph nodes were left unsegmented). [Bouget et al. 2023](https://github.com/dbouget/ct_mediastinal_structures_segmentation) took these available annotations as a starting point and manually refined segmentations for all mediastinal lymph nodes in 89 CTs (mediastinal case 43 was removed by [Bouget et al. 2023](https://github.com/dbouget/ct_mediastinal_structures_segmentation) due to its incomplete CT volume). We additionally excluded two CTs (mediastinal case 06 and case 80) because of the absence of full annotations after strict examination, thus leaving eligible a final set of 87 CTs.

**St. Olavs Hospital data**  
[Original St. Olavs Hospital data](https://datadryad.org/stash/dataset/doi:10.5061/dryad.mj76c) consists of 17 CTs, among which 15 contrast-enhanced CTs were first assigned their manual annotations of lymph nodes and fifteen different anatomical structures in the mediastinal area ([Bouget et al. 2019](https://link.springer.com/article/10.1007/s11548-019-01948-8)). These lymph node annotations of the 15 CTs were further refined and proofed by an expert thoracic radiologist, as a benchmark dataset of [Bouget et al. 2023](https://github.com/dbouget/ct_mediastinal_structures_segmentation).


## Environments and Requirements
Install nnUNet as below and meet the requirements of nnUNet. For more details please refer to [https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1/)  
```
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
git checkout nnunetv1
pip install -e .
```


## Instructions for training
### 1.1 Airway-guided pre-processing  
- Resampling (0.8mm x 0.8mm x 0.8mm)
- Lung segmentation
- Initial VOI boundary extraction based on lung mask
- Cropping VOI for lung airway segmentation
- Removing airway maps in lung mask regions
- Secondary VOI boundary extraction based on bronchus
- Cropping input VOI based on two VOI bounding boxes 

### 1.2 Learning with full annotation data for fully supervised segmentation of mediastinal lymph nodes  
- Developing a nnUNet model (3d_fullres configuration) based on two additional data with full annotations (102 cases)
- Training nnUNet based on one fold of the 5-fold-cross-validation for 1000 epochs 
### 1.3 Volume-based post-processing
- Resampling and restoring to original CT volume
- Removing individual components whose volume are less than the volume of a sphere with a radius of 5 mm
### 1.4 Pseudo label transfer for LNQ training data and label fusion
- Generating pseudo labels for LNQ training data
- Combining the LNQ pseudo labels and original labels to generate the new labels
### 1.5 Model finetuning using total cohort
- The total cohort includes the full-annotation data and LNQ data with pseudo label transfer
- Finetune the former model parameters using this total cohort for 300 epochs
