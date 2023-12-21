# <div align=center> [SegRap2023](https://segrap2023.grand-challenge.org/)</div>

## 1. Totorial for Algorithm Docker Image ([official guideline of grand challenge](https://grand-challenge.org/documentation/test-and-deploy-your-container/))
### 1.1 Important input and output
- For task1, the input dir is `/input/images/head-neck-ct/` (non-contrast-ct images) and `/input/images/head-neck-contrast-enhanced-ct/` (contrast-ct images). The output dir is `/output/images/head-neck-segmentation/`. Note that the final prediction has to be a 4D .mha file, which array shape is [45, *image_shape]. An example code and output is shown as `Docker_tutorial/stacked_results_to_4d_mha.py` and `Docker_tutorial/oars_output_example.mha`.

- For task2, the dir is `/input/images/head-neck-ct/` (non-contrast-ct images) and `/input/images/head-neck-contrast-enhanced-ct/` (contrast-ct images). The output dir is `/output/images/gross-tumor-volume-segmentation/`. Note that the final prediction has to be a 4D .mha file, which array shape is [2, *image_shape]. An example code and output is shown as `Docker_tutorial/stacked_results_to_4d_mha.py` and `Docker_tutorial/gtvs_output_example.mha`.

### 1.2 Algorithm examples based on nnUNet
We provide two algorithm examples based on nnUNet, which is only the baseline for two tasks. If your method is based on nnUNet, you can follow the example to generate predictions and run `sh export.sh` to generate an Algorithm Container Image in tar.gz format. The details about loading input, generating predictions, and saving output can be seen in the `process.py`. 

In addition, you can download the example data and model weight from [GoogleDrive](https://drive.google.com/file/d/17hJz9hQ1sajsW0aEgmiydvL9bVchqipr/view?usp=sharing) and [BaiduNetDisk](https://pan.baidu.com/s/1lwGENM9R7z3791FxQoy7fQ?pwd=2023) to the folder `images` and  `weight`, respectively. Before submitting, you can test the docker image on your local machine by running `sh test.sh` or `sudo sh test.sh`, we show an [example output](https://github.com/HiLab-git/SegRap2023/blob/main/Docker_tutorial/outputs.png) on our ubuntu22.04 (one 3090 GPU).

#### 1.2.1 How to test the container locally?
1. Parepare your images and weight as following format
    - SegRap2023_task1_OARs_nnUNet_Example
        - images
            - images
                - head-neck-contrast-enhanced-ct
                    - segrap_0001.mha
                - head-neck-ct
                    - segrap_0001.mha
        - weight
            - fold_0
                - model_final_checkpoint.model
                - model_final_checkpoint.model.pkl
                - plans.pkl

2. How about the output?
    - You can check out if there are predictions in the output folder `/output/images/head-neck-segmentation` or `/output/images/gross-tumor-volume-segmentation` that are corresponded to the input images. Run the following command will show the files in the output folder.
        ```bash
        docker run --rm \
            -v segrap2023_segmentationcontainer-output-$VOLUME_SUFFIX:/output/ \
            python:3.10-slim ls -al /output/images/head-neck-segmentation
        ```
    - The `test` folder is just an empty folder which hasn't been used in the docker image, so you can ignore or remove it.

    - You can ignore the error `No such file or directory: '/output/results.json'` when you run the docker locally.
      
### 1.3 Algorithm examples based on others.
If your method is not based on nnUNet, you can modify the function of `predict()` in `process.py` and other corresponding parts for inference. It's easy to read and modify, but please ensure the format of the output file (a 4D .mha, the right mapping between the index of 4D file and OARs or GTVs.). We provided an example (`Docker_tutorial/stacked_results_to_4d_mha.py`) to stack individual oars/gtvs predictions of a patient into a required 4d .mha files.

### 1.4 Q&A.
If you meet any questions when submitting your docker images, you can email us (`luoxd1996@gmail.com` or `fujia98914@gmail.com`), or post [the issue](https://github.com/HiLab-git/SegRap2023/issues) or discuss it in [the forum](https://grand-challenge.org/forums/forum/segmentation-of-organs-at-risk-and-gross-tumor-volume-of-npc-699/) at any time.

## 2. How to submit the algorithm?
1. If you have not created your algorithm, you can go to https://segrap2023.grand-challenge.org/evaluation/challenge/algorithms/create/ to create an algorithm with 30GB memory.

2. Upload your Algorithm Container Image, then wait for the container to be active.

3. Go to the [SegRap2023 submit website](https://segrap2023.grand-challenge.org/evaluation/challenge/submissions/create/), choose the task and submit your Algorithm Image.

4. After submitting, you can wait for the update of [Leaderboards](https://segrap2023.grand-challenge.org/evaluation/challenge/leaderboard/).


## 3. Tutorial for SegRap2023 Challenge

This repository provides tutorial code for Segmentation of Organs-at-Risk and Gross Tumor Volume of NPC for Radiotherapy Planning (SegRap2023) Challenge. Our code is based on [PyMIC](https://github.com/HiLab-git/PyMIC), a pytorch-based toolkit for medical image computing with deep learning, that is lightweight and easy to use. 

### Requirements
This code depends on [Pytorch](https://pytorch.org), [PyMIC](https://github.com/HiLab-git/PyMIC).
To install PyMIC and GeodisTK, run:
```
pip install PYMIC
``` 


### Segmentation model based on PyMIC


#### Dataset and Preprocessing
- Download the dataset from [SegRap2023](https://segrap2023.grand-challenge.org/) and put the dataset in the `data_dir/raw_data`.

- For data preprocessing, run:
    ```bash
    python Tutorial/preprocessing.py
    ```
    This will crop the images with the maximal nonzero bounding box, and the cropped results are normalized based on the intensity properties of all training images. By setting the `args.task` to `OARs` and `GTVs`, we can get the preprocessed images and labels for two tasks that are saved in `data_dir/Task001_OARs_preprocess` and `data_dir/Task002_GTVs_preprocess`, respectively.

#### Training
- Run the following command to create csv files for training, validation, and testing. The csv files will be saved to `config/data_OARs` and `config/data_GTVs`.
    ```bash
    python Tutorial/write_csv_files.py
    ```
    
- Run the following command for training and validation. The segmentation model will be saved in `model/unet3d_OARs` and `model/unet3d_GTVs`, respectively.
    ```bash
    pymic_train Tutorial/config/unet3d_OARs.cfg
    pymic_train Tutorial/config/unet3d_GTVs.cfg
    ```
    Note that you can modify the settings in .cfg file to get better segmentation results, such as RandomCrop_output_size, loss_class_weight, etc.

#### Testing
- After training, run the following command, we can get the performance on the testing set, and the predictions of testing data will be saved in `result/unet3d_OARs` and `result/unet3d_GTVs`.
    ```bash
    pymic_test Tutorial/config/unet3d_OARs.cfg
    pymic_test Tutorial/config/unet3d_GTVs.cfg
    ```

#### Postprocessing
- Run the following command to obtain the final predictions, which are saved in `result/unet3d_OARs_post` and `result/unet3d_GTVs_post`.
    ```bash
    python Tutorial/postprocessing.py
    ```

### Segmentation model based on nnUNet
#### Postprocessing
- Following `Tutorial/nnunet_baseline.ipynb`, you can obtain the final predictions based on the outputs from [nnUNet](https://github.com/MIC-DKFZ/nnUNet). In addition, you also can use `Tutorial/preprocessing.py` for preprocessing firstly (details as mentioned in the above tutorial) and then train networks using `Tutorial/nnunet_baseline.ipynb`.


## 4. Evaluation for SegRap2023 Challenge
Run following command to get the quantitative evaluation results.
```bash
python Eval/SegRap_Task001_DSC_NSD_Eval.py
python Eval/SegRap_Task002_DSC_NSD_Eval.py
```
### Reference

```
@article{luo2023segrap2023,
  title={SegRap2023: A Benchmark of Organs-at-Risk and Gross Tumor Volume Segmentation for Radiotherapy Planning of Nasopharyngeal Carcinoma},
  author={Luo, Xiangde and Fu, Jia and Zhong, Yunxin and Liu, Shuolin and Han, Bing and Astaraki, Mehdi and Bendazzoli, Simone and Toma-Dasu, Iuliana and Ye, Yiwen and Chen, Ziyang and others},
  journal={arXiv preprint arXiv:2312.09576},
  year={2023}
}

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
