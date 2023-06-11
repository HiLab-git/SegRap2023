import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import SimpleITK as sitk
from two_evaluation_metrics import dsc, nsd


submission_path = '/data_8t/radiology_images/processed/SegRap2023/SegRap2023_Validation_Set(with labels)'
gt_path = '/data_8t/radiology_images/processed/SegRap2023/SegRap2023_Validation_Set(with labels)'
save_path = '/data_8t/radiology_images/processed/SegRap2023/SegRap2023_Validation_Set(with labels)'
patientnames = os.listdir(submission_path)
task02_submission_result = OrderedDict()

task02_submission_result['Name'] = list()

task02_label_tolerance = OrderedDict({"GTVp": 1, "GTVnd": 1})


for lesion in task02_label_tolerance.keys():
    task02_submission_result['{}_DSC'.format(lesion)] = list()
for lesion in task02_label_tolerance.keys():
    task02_submission_result['{}_NSD'.format(lesion)] = list()


def compute_each_organ_lesion_performace(result, reference, voxel_spacing, tolerance_mm):
    if np.sum(reference) == 0 and np.sum(result) == 0:
        DSC = 1
        NSD = 1
    elif np.sum(reference) == 0 and np.sum(result) > 0:
        DSC = 0
        NSD = 0
    else:
        DSC = dsc(result, reference)
        NSD = nsd(result, reference, voxel_spacing, tolerance_mm)
    return round(DSC, 4), round(NSD, 4)


def nii2arr(path):
    itk_data = sitk.ReadImage(path)
    arr_data = sitk.GetArrayFromImage(itk_data)
    spacing = itk_data.GetSpacing()[::-1]
    return arr_data, spacing


for patient in os.listdir(submission_path):
    print(patient)
    task02_submission_result["Name"].append(patient)

    for lesion in sorted(task02_label_tolerance.keys()):
        result_lesion, spacing = nii2arr(
            "{}/{}/{}.nii.gz".format(submission_path, patient, lesion))
        reference_lesion, spacing = nii2arr(
            "{}/{}/{}.nii.gz".format(gt_path, patient, lesion))
        DSC_lesion, NSD_lesion = compute_each_organ_lesion_performace(
            result_lesion > 0, reference_lesion > 0, spacing, task02_label_tolerance[lesion])
        task02_submission_result['{}_DSC'.format(lesion)].append(DSC_lesion)
        task02_submission_result['{}_NSD'.format(lesion)].append(NSD_lesion)

task02_df = pd.DataFrame(task02_submission_result)
task02_df.to_csv(os.path.join(
    save_path, 'DSC_NSD_Task02_Admin_nnUNetV2.csv'), index=False)
