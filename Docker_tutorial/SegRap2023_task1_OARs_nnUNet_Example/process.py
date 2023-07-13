import torch
from post_processing import convert_one_hot_label_to_multi_organs
from inference_code import predict_from_folder_segrap2023
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)
from evalutils import SegmentationAlgorithm
import numpy as np
import SimpleITK
import os
import sys
o_path = os.getcwd()
print(o_path)
sys.path.append(o_path)


class Customalgorithm():  # SegmentationAlgorithm is not inherited in this class anymore
    def __init__(self):
        """
        Do not modify the `self.input_dir` and `self.output_dir`. 
        (Check https://grand-challenge.org/algorithms/interfaces/)
        """
        self.input_dir = "/input/"
        self.output_dir = "/output/images/head-neck-segmentation/"
        
        """
        Store the validation/test data and predictions into the `self.nii_path` and `self.result_path`, respectively.
        Put your model and pkl files into the `self.weight`.
        """
        self.nii_path = '/opt/app/nnUNet_raw_data_base/nnUNet_raw_data/Task001_SegRap2023/imagesTs'
        self.result_path = '/opt/app/nnUNet_raw_data_base/nnUNet_raw_data/Task001_SegRap2023/result'
        self.nii_seg_file = 'SegRap2023_001.nii.gz'
        self.weight = "./weight/"
        if not os.path.exists(self.nii_path):
            os.makedirs(self.nii_path, exist_ok=True)
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path, exist_ok=True)
        pass

    def convert_mha_to_nii(self, mha_input_path, nii_out_path):  # nnUNet specific
        img = SimpleITK.ReadImage(mha_input_path)
        print(img.GetSize())
        SimpleITK.WriteImage(img, nii_out_path, True)

    def convert_nii_to_mha(self, nii_input_path, mha_out_path):  # nnUNet specific
        img = SimpleITK.ReadImage(nii_input_path)
        SimpleITK.WriteImage(img, mha_out_path, True)

    def check_gpu(self):
        """
        Check if GPU is available. Note that the Grand Challenge only has one available GPU.
        """
        print('Checking GPU availability')
        is_available = torch.cuda.is_available()
        print('Available: ' + str(is_available))
        print(f'Device count: {torch.cuda.device_count()}')
        if is_available:
            print(f'Current device: {torch.cuda.current_device()}')
            print('Device name: ' + torch.cuda.get_device_name(0))
            print('Device memory: ' +
                  str(torch.cuda.get_device_properties(0).total_memory))

    def load_inputs(self):      # use two modalities input data
        """
        Read input data (two modalities) from `self.input_dir` (/input/). 
        Please do not modify the path for CT and contrast-CT images.
        """
        ct_mha = os.listdir(os.path.join(self.input_dir, 'images/head-neck-ct/'))[0]
        ctc_mha = os.listdir(os.path.join(self.input_dir, 'images/head-neck-contrast-enhanced-ct/'))[0]
        uuid = os.path.splitext(ct_mha)[0]

        """
        if your model was based on nnUNet baseline and used two modalities as inputs, 
        please convert the input data into '_0000.nii.gz' and '_0001.nii.gz' using following code.
        """
        self.convert_mha_to_nii(os.path.join(self.input_dir, 'images/head-neck-ct/', ct_mha),
                                os.path.join(self.nii_path, 'SegRap2023_001_0000.nii.gz'))
        self.convert_mha_to_nii(os.path.join(self.input_dir, 'images/head-neck-contrast-enhanced-ct/', ctc_mha),
                                os.path.join(self.nii_path, 'SegRap2023_001_0001.nii.gz'))
        
        # Check the validation/test data exist.
        print(os.listdir('/opt/app/nnUNet_raw_data_base/nnUNet_raw_data/Task001_SegRap2023/imagesTs'))
        
        return uuid
    

    # def load_inputs(self):            # only use non-contrast-CT images as input
    #     """
    #     Read input data (non-contrast-CT images) from `self.input_dir` (/input/). 
    #     Please do not modify the path for non-contrast-CT images.
    #     """
    #     ct = os.listdir(os.path.join(self.input_dir, 'images/head-neck-ct/'))[0]
    #     uuid = os.path.splitext(ct)[0]

    #     """
    #     if your model was based on nnUNet baseline and only used non-contrast-CT images as inputs, 
    #     please convert the input data into '_0000.nii.gz' using following code.
    #     """
    #     self.convert_mha_to_nii(os.path.join(self.input_dir, 'images/head-neck-ct/', ct),
    #                             os.path.join(self.nii_path, 'SegRap2023_001_0000.nii.gz'))
    #     # Check the validation/test data exist.
    #     print(os.listdir('/opt/app/nnUNet_raw_data_base/nnUNet_raw_data/Task001_SegRap2023/imagesTs'))
        
    #     return uuid
    

    # def load_inputs(self):            # only use contrast-CT images as input
    #     """
    #     Read input data (single contrast-CT images) from `self.input_dir` (/input/). 
    #     Please do not modify the path for contrast-CT images.
    #     """
    #     ct = os.listdir(os.path.join(self.input_dir, 'images/head-neck-contrast-enhanced-ct/'))[0]
    #     uuid = os.path.splitext(ct)[0]

    #     """
    #     if your model was based on nnUNet baseline and only used contrast-CT images as inputs, 
    #     please convert the input data into '_0000.nii.gz' using following code.
    #     """
    #     self.convert_mha_to_nii(os.path.join(self.input_dir, 'images/head-neck-contrast-enhanced-ct/', ct),
    #                             os.path.join(self.nii_path, 'SegRap2023_001_0000.nii.gz'))

    #     # Check the validation/test data exist.
    #     print(os.listdir('/opt/app/nnUNet_raw_data_base/nnUNet_raw_data/Task001_SegRap2023/imagesTs'))
        
    #     return uuid


    def write_outputs(self, uuid):
        """
        If you used one-hot label (54 classes) for training, please convert the 54 classes prediction to 45 oars prediction using function `convert_one_hot_label_to_multi_organs`.
        Otherwise, stack your 45 predictions for oars in the first channel, the corresponding mapping between the channel index and the organ names is:
        {0: 'Brain',
        1: 'BrainStem',
        2: 'Chiasm',
        3: 'TemporalLobe_L',
        4: 'TemporalLobe_R',
        5: 'Hippocampus_L',
        6: 'Hippocampus_R',
        7: 'Eye_L',
        8: 'Eye_R',
        9: 'Lens_L',
        10: 'Lens_R',
        11: 'OpticNerve_L',
        12: 'OpticNerve_R',
        13: 'MiddleEar_L',
        14: 'MiddleEar_R',
        15: 'IAC_L',
        16: 'IAC_R',
        17: 'TympanicCavity_L',
        18: 'TympanicCavity_R',
        19: 'VestibulSemi_L',
        20: 'VestibulSemi_R',
        21: 'Cochlea_L',
        22: 'Cochlea_R',
        23: 'ETbone_L',
        24: 'ETbone_R',
        25: 'Pituitary',
        26: 'OralCavity',
        27: 'Mandible_L',
        28: 'Mandible_R',
        29: 'Submandibular_L',
        30: 'Submandibular_R',
        31: 'Parotid_L',
        32: 'Parotid_R',
        33: 'Mastoid_L',
        34: 'Mastoid_R',
        35: 'TMjoint_L',
        36: 'TMjoint_R',
        37: 'SpinalCord',
        38: 'Esophagus',
        39: 'Larynx',
        40: 'Larynx_Glottic',
        41: 'Larynx_Supraglot',
        42: 'PharynxConst',
        43: 'Thyroid',
        44: 'Trachea'}
        Please ensure the 0 channel is the prediction of Brain, the 1 channel is the prediction of BrainStem, ......, the 44 channel is the prediction of Trachea.
        and also ensure the shape of final prediction array is [45, *image_shape].
        The predictions should be saved in the `self.output_dir` (/output/). Please do not modify the path and the suffix (.mha) for saving the prediction.
        """
        os.makedirs(os.path.dirname(self.output_dir), exist_ok=True)
        convert_one_hot_label_to_multi_organs(os.path.join(
            self.result_path, self.nii_seg_file), os.path.join(self.output_dir, uuid + ".mha"))
        print('Output written to: ', os.path.join(self.output_dir, uuid + ".mha"))
        
    def predict(self):
        """
        load the model and checkpoint, and generate the predictions. You can replace this part with your own model.
        """
        predict_from_folder_segrap2023(self.weight, self.nii_path, self.result_path, 0, 0, 1)
        print("nnUNet segmentation done!")
        if not os.path.exists(os.path.join(self.result_path, self.nii_seg_file)):
            print('waiting for nnUNet segmentation to be created')

        while not os.path.exists(os.path.join(self.result_path, self.nii_seg_file)):
            import time
            print('.', end='')
            time.sleep(5)
        # print(cproc)  # since nnUNet_predict call is split into prediction and postprocess, a pre-mature exit code is received but segmentation file not yet written. This hack ensures that all spawned subprocesses are finished before being printed.
        print('Prediction finished !')

    def post_process(self):
        self.check_gpu()
        print('Start processing')
        uuid = self.load_inputs()
        print('Start prediction')
        self.predict()
        print('Start output writing')
        self.write_outputs(uuid)

    def process(self):
        """
        Read inputs from /input, process with your algorithm and write to /output
        """
        print(self.weight, self.nii_path, self.result_path)
        self.post_process()


if __name__ == "__main__":
    Customalgorithm().process()
