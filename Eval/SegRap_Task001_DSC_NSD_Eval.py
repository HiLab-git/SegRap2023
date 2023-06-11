import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import SimpleITK as sitk
from two_evaluation_metrics import dsc, nsd


submission_path = '/data_8t/radiology_images/processed/SegRap2023/nnUNetV2_infersVal/task001'
gt_path = '/data_8t/radiology_images/processed/SegRap2023/SegRap2023_Validation_Set_Labels_Cropped'
save_path = '/data_8t/radiology_images/processed/SegRap2023/nnUNetV2_infersVal'
patientnames = os.listdir(submission_path)
task01_submission_result = OrderedDict()

task01_submission_result['Name'] = list()

task01_label_tolerance = OrderedDict({
    "Brain": 1,
    "BrainStem": 1,
    "Chiasm": 1,
    "Cochlea_L": 1,
    "Cochlea_R": 1,
    "Esophagus": 1,
    "ETbone_L": 1,
    "ETbone_R": 1,
    "Eye_L": 1,
    "Eye_R": 1,
    "Hippocampus_L": 1,
    "Hippocampus_R": 1,
    "IAC_L": 1,
    "IAC_R": 1,
    "Larynx": 2,
    "Larynx_Glottic": 1,
    "Larynx_Supraglot": 1,
    "Lens_L": 1,
    "Lens_R": 1,
    "Mandible_L": 1,
    "Mandible_R": 1,
    "Mastoid_L": 1,
    "Mastoid_R": 1,
    "MiddleEar_L": 1,
    "MiddleEar_R": 1,
    "OpticNerve_L": 1,
    "OpticNerve_R": 1,
    "OralCavity": 3,
    "Parotid_L": 1,
    "Parotid_R": 1,
    "PharynxConst": 1,
    "Pituitary": 1,
    "SpinalCord": 1,
    "Submandibular_L": 1,
    "Submandibular_R": 1,
    "TemporalLobe_L": 1,
    "TemporalLobe_R": 1,
    "Thyroid": 1,
    "Trachea": 1,
    "TympanicCavity_L": 1,
    "TMjoint_L": 1,
    "TMjoint_R": 1,
    "TympanicCavity_R": 1,
    "VestibulSemi_L": 1,
    "VestibulSemi_R": 1
})

for organ in task01_label_tolerance.keys():
    task01_submission_result['{}_DSC'.format(organ)] = list()
for organ in task01_label_tolerance.keys():
    task01_submission_result['{}_NSD'.format(organ)] = list()


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
    task01_submission_result["Name"].append(patient)

    for organ in sorted(task01_label_tolerance.keys()):
        result_organ, spacing = nii2arr(
            "{}/{}/{}.nii.gz".format(submission_path, patient, organ))
        reference_organ, spacing = nii2arr(
            "{}/{}/{}.nii.gz".format(gt_path, patient, organ))
        DSC_organ, NSD_organ = compute_each_organ_lesion_performace(
            result_organ > 0, reference_organ > 0, spacing, task01_label_tolerance[organ])
        task01_submission_result['{}_DSC'.format(organ)].append(DSC_organ)
        task01_submission_result['{}_NSD'.format(organ)].append(NSD_organ)


task01_df = pd.DataFrame(task01_submission_result)
task01_df.to_csv(os.path.join(
    save_path, 'DSC_NSD_Task001_Admin_nnUNetV2.csv'), index=False)
