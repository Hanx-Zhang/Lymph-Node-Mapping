![image](https://github.com/Hanx-Zhang/Lymph-Node-Mapping/assets/47120259/f1ddf9f6-9028-4d34-9980-ae5d91a030b3)# Solution of Team IMR for MICCAI 2023 LNQ Challenge

***

**Transfer Mapping for Clinically Relevant Lymph Nodes in The Mediastinal Area of CT data**  
*Hanxiao Zhang, Minghui Zhang, Xin You, Zhebing Lin, Yi Zhang, Liu Liu, Xinghua Cheng, Yun Gu and Guang-Zhong Yang*

Built upon [nnUNet](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1/), this repository provides the solution of Team IMR for [MICCAI 2023 LNQ Challenge](https://lnq2023.grand-challenge.org/lnq2023/). This repository contains the training method and inference code for diseased and mediastinal lymph node segmentation using contrast-enhanced CT scans.

***

## Environments and Requirements


## Additonal data
In that [LNQ training data](https://lnq2023.grand-challenge.org/data/) are partially annotated (i.e. one node out of five), this solution leverages two additional data with full mediastinal lynph node annotations in training process. Both additional data are publicly available.  

**TICA data with refined annotations**  
[Original TICA data](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=19726546#19726546fcb14b04d2494090ab696ba899c8d70c/) consists of 90 contrast-enhanced CTs of the mediastinum with lymph node position labels marked by radiologists at the National Institutes of Health (NIH). Case 43

**St. Olavs University Hospital Data** 


## Instructions for running
