# <div align=center> [SegRap2023](https://segrap2023.grand-challenge.org/)</div>

## 1. Tutorial for SegRap2023 Challenge
[PyMIC_link]:https://github.com/HiLab-git/PyMIC

This repository provides tutorial code for Segmentation of Organs-at-Risk and Gross Tumor Volume of NPC for Radiotherapy Planning (SegRap2023) Challenge. Our code is based on [PyMIC](PyMIC_link), a pytorch-based toolkit for medical image computing with deep learning, that is lightweight and easy to use. 

```
@article{wang2023pymic,
  title={PyMIC: A deep learning toolkit for annotation-efficient medical image segmentation},
  author={Wang, Guotai and Luo, Xiangde and Gu, Ran and Yang, Shuojue and Qu, Yijie and Zhai, Shuwei and Zhao, Qianfei and Li, Kang and Zhang, Shaoting},
  journal={Computer Methods and Programs in Biomedicine},
  volume={231},
  pages={107398},
  year={2023},
  publisher={Elsevier}
}
```

### Requirements
This code depends on [Pytorch](https://pytorch.org), [PyMIC][PyMIC_link].
To install PyMIC and GeodisTK, run:
```
pip install PYMIC
``` 

### Dataset and Preprocessing
Download the dataset from [SegRap2023](https://segrap2023.grand-challenge.org/) and put the dataset in the `data_dir/raw_data`.

For data preprocessing, run:
```bash
python Tutorial/preprocessing.py
```
This will crop the images with the maximal nonzero bounding box, and the cropped results are normalized based on the intensity properties of all training images. By setting the `args.task` to `OARs` and `GTVs`, we can get the preprocessed images and labels for two tasks that are saved in `data_dir/Task001_OARs_preprocess` and `data_dir/Task002_GTVs_preprocess`, respectively.


### Segmentation model based on PyMIC
#### Training
1. Run the following command to create csv files for training, validation, and testing. The csv files will be saved to `config/data_OARs` and `config/data_GTVs`.
    ```bash
    python Tutorial/write_csv_files.py
    ```
    
2. Run the following command for training and validation. The segmentation model will be saved in `model/unet3d_OARs` and `model/unet3d_GTVs`, respectively.
    ```bash
    pymic_train Tutorial/config/unet3d_OARs.cfg
    pymic_train Tutorial/config/unet3d_GTVs.cfg
    ```

3. After training, run the following command, we can get the performance on the testing set, and the predictions of testing data will be saved in `result/unet3d_OARs` and `result/unet3d_GTVs`.
    ```bash
    pymic_test Tutorial/config/unet3d_OARs.cfg
    pymic_test Tutorial/config/unet3d_GTVs.cfg
    ```
Note that you can modify the settings in .cfg file to get better segmentation results, such as RandomCrop_output_size, loss_class_weight, etc.


4. Run the following command to obtain the final predictions, which are saved in `result/unet3d_OARs_post` and `result/unet3d_GTVs_post`.
    ```bash
    python Tutorial/postprocessing.py
    ```

### Segmentation model based on nnUNet
#### postprocessing
Following `Tutorial/nnunet_baseline.ipynb`, you can obtain the final predictions.


## 2. Evaluation for SegRap2023 Challenge
Run following command to get the quantitative evaluation results.
```bash
python Eval/SegRap_Task001_DSC_NSD_Eval.py
python Eval/SegRap_Task002_DSC_NSD_Eval.py
```
