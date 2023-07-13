import SimpleITK as sitk
import numpy as np
import shutil
import glob
import os
import sys
o_path = os.getcwd()
sys.path.append(o_path)


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


def nii2array(path):
    mask_itk_ref = sitk.ReadImage(path)
    mask_arr_ref = sitk.GetArrayFromImage(mask_itk_ref)
    return mask_arr_ref


def merge_multi_class_to_one(input_arr, classes_index=None):
    new_arr = np.zeros_like(input_arr)
    for cls_ind in classes_index:
        new_arr[input_arr == cls_ind] = 1
    return new_arr


def convert_one_hot_label_to_multi_organs(ont_hot_label_path, save_path):
    patient_results = []
    spacing = None
    for organ in segrap_subset_task001.keys():
        ont_hot_label_arr = nii2array(ont_hot_label_path)
        ont_hot_label_itk = sitk.ReadImage(ont_hot_label_path)
        spacing = ont_hot_label_itk.GetSpacing()

        if type(segrap_subset_task001[organ]) is list:
            new_arr = merge_multi_class_to_one(
                ont_hot_label_arr, segrap_subset_task001[organ])
        else:
            new_arr = np.zeros_like(ont_hot_label_arr)
            new_arr[ont_hot_label_arr == segrap_subset_task001[organ]] = 1
        patient_results.append(new_arr)

    oars = []
    for t in patient_results:
        oars.append(sitk.GetImageFromArray(t, False))
    output_itk = sitk.JoinSeries(oars)
    new_spacing = (spacing[0], spacing[1], spacing[2], 1)
    output_itk.SetSpacing(new_spacing)
    print(output_itk.GetSize())
    sitk.WriteImage(output_itk, save_path, True)
    print("Conversion Finished !")


def convert_one_hot_label_to_multi_lesions(ont_hot_label_path, save_fold):
    patient_results = []
    spacing = None
    for lesion in segrap_subset_task002.keys():
        ont_hot_label_arr = nii2array(ont_hot_label_path)
        ont_hot_label_itk = sitk.ReadImage(ont_hot_label_path)
        spacing = ont_hot_label_itk.GetSpacing()
        new_arr = np.zeros_like(ont_hot_label_arr)
        new_arr[ont_hot_label_arr == segrap_subset_task002[lesion]] = 1
        patient_results.append(new_arr)
    new_itk = sitk.GetImageFromArray(
        np.array(patient_results).transpose(1, 2, 3, 0))
    new_itk.SetSpacing(spacing)
    sitk.WriteImage(new_itk, "{}.nii.gz".format(save_fold))
    return "Conversion Finished"


# if __name__ == "__main__":
#     for patient in glob.glob("test/*"):
#         new_path = "test/{}".format(
#             patient.split("/")[-1].replace("_cropped.nii.gz", ""))
#         if os.path.exists(new_path):
#             pass
#             convert_one_hot_label_to_multi_organs(patient, new_path)
#         else:
#             os.mkdir(new_path)
#             convert_one_hot_label_to_multi_organs(patient, new_path)
#     print("Convert all predictions to single organ files")
