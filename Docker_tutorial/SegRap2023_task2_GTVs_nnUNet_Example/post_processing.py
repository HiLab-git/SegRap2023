import SimpleITK as sitk
import numpy as np
import shutil
import glob
import os
import sys
o_path = os.getcwd()
sys.path.append(o_path)


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


def convert_one_hot_label_to_multi_lesions(ont_hot_label_path, save_path):
    patient_results = []
    spacing = None
    for lesion in segrap_subset_task002.keys():
        ont_hot_label_arr = nii2array(ont_hot_label_path)
        ont_hot_label_itk = sitk.ReadImage(ont_hot_label_path)
        spacing = ont_hot_label_itk.GetSpacing()
        new_arr = np.zeros_like(ont_hot_label_arr)
        new_arr[ont_hot_label_arr == segrap_subset_task002[lesion]] = 1
        patient_results.append(new_arr)
    oars = []
    for t in patient_results:
        oars.append(sitk.GetImageFromArray(t, False))
    output_itk = sitk.JoinSeries(oars)
    new_spacing = (spacing[0], spacing[1], spacing[2], 1)
    output_itk.SetSpacing(new_spacing)
    print(output_itk.GetSize())
    sitk.WriteImage(output_itk, save_path, True)
    print("Conversion Finished")


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
