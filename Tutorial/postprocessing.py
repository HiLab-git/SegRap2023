import os
import glob
import numpy as np
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *
from utils import *
from scipy import ndimage

segrap_subset_task001 = {
    'Brain': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "BrainStem": 2,
    "Chiasm": 3,
    "TemporalLobe_L": [4, 6],
    "TemporalLobe_R": [5, 7],
    "Hippocampus_L": [8, 6],
    "Hippocampus_R": [9, 7],
    'Eye_L': [10, 12],
    'Eye_R': [11, 13],
    "Lens_L": 12,
    "Lens_R": 13,
    "OpticNerve_L": 14,
    "OpticNerve_R": 15,
    "MiddleEar_L": [18, 16, 20, 24, 28, 30],
    "MiddleEar_R": [19, 17, 21, 25, 29, 31],
    "IAC_L": 18,
    "IAC_R": 19,
    "TympanicCavity_L": [22, 20],
    "TympanicCavity_R": [23, 21],
    "VestibulSemi_L": [26, 24],
    "VestibulSemi_R": [27, 25],
    "Cochlea_L": 28,
    "Cochlea_R": 29,
    "ETbone_L": [32, 30],
    "ETbone_R": [33, 31],
    "Pituitary": 34,
    "OralCavity": 35,
    "Mandible_L": 36,
    "Mandible_R": 37,
    "Submandibular_L": 38,
    "Submandibular_R": 39,
    "Parotid_L": 40,
    "Parotid_R": 41,
    "Mastoid_L": 42,
    "Mastoid_R": 43,
    "TMjoint_L": 44,
    "TMjoint_R": 45,
    "SpinalCord": 46,
    "Esophagus": 47,
    "Larynx": [48, 49, 50, 51],
    "Larynx_Glottic": 49,
    "Larynx_Supraglot": 50,
    "PharynxConst": [51, 52],
    "Thyroid": 53,
    "Trachea": 54}


segrap_subset_task002 = {
    "GTVp": 1,
    "GTVnd": 2}

def merge_multi_class_to_one(input_arr, classes_index=None):
    new_arr = np.zeros_like(input_arr)
    for cls_ind in classes_index:
        new_arr[input_arr == cls_ind] = 1
    return new_arr


def convert_one_hot_label_to_multi_organs(ont_hot_label_arr, save_fold, spacing=None, origin=None, direction=None):
    for organ in segrap_subset_task001.keys():        
        if type(segrap_subset_task001[organ]) is list:
            new_arr = merge_multi_class_to_one(ont_hot_label_arr, segrap_subset_task001[organ])
        else:
            new_arr = np.zeros_like(ont_hot_label_arr)
            new_arr[ont_hot_label_arr == segrap_subset_task001[organ]] = 1
        save_nii(new_arr, "{}/{}.nii.gz".format(save_fold, organ), spacing=spacing, origin=origin, direction=direction)
    return "Conversion Finished"


def convert_one_hot_label_to_multi_lesions(ont_hot_label_arr, save_fold, spacing=None, origin=None, direction=None):
    for lesion in segrap_subset_task002.keys():
        new_arr = np.zeros_like(ont_hot_label_arr)
        new_arr[ont_hot_label_arr == segrap_subset_task002[lesion]] = 1
        save_nii(new_arr, "{}/{}.nii.gz".format(save_fold, lesion), spacing=spacing, origin=origin, direction=direction)
    return "Conversion Finished"


def get_raw_data(seg, params, name):
    """get preprocessing parameters"""    
    raw_spacing, target_spacing, origin, direction, raw_shape, resample_shape, bbox = params[name]

    """cropped --> resampled"""    
    seg_full = np.zeros(resample_shape, dtype=seg.dtype)
    seg_full[bbox[0][0]: bbox[0][1], bbox[1][0]: bbox[1][1], bbox[2][0]: bbox[2][1]] = seg
    
    """resampled --> raw"""
    scale = np.array(target_spacing) / np.array(raw_spacing)
    seg_raw = ndimage.zoom(seg_full, scale, order=0)
    
    assert list(seg_raw.shape) == raw_shape[1:]
    spacing = (raw_spacing[2], raw_spacing[1], raw_spacing[0])
    
    return seg_raw, spacing, origin, direction


if __name__ == "__main__":
    seg_dir = 'result/unet3d_OARs'
    seg_dir_post = 'result/unet3d_OARs_post'

    json_file =  'data_dir/SegRap2023_dataset.json'
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    for patient in glob.glob(seg_dir + '/*.nii.gz'):
        patient_name = patient.split('/')[-1].replace('.nii.gz', '')
        seg = nii2array(patient)
        seg_itk = sitk.ReadImage(patient)

        seg_raw, spacing, origin, direction = get_raw_data(seg, data, patient_name)   

        new_path = '{}/{}'.format(seg_dir_post, patient)
        maybe_mkdir_p(new_path)
        convert_one_hot_label_to_multi_organs(seg_raw, new_path, spacing=spacing, origin=origin, direction=direction)
        