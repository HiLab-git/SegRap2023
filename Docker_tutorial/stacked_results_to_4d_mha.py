import os
import numpy as np
import SimpleITK as sitk

# download and unzip "Docker_tutorial/gtvs_output_example.zip" and "Docker_tutorial/oars_output_example.zip" to run this example.
# # The mapping between the structures (OARs or GTVs) and the index of the 4D file (mha).
oars_mapping_dict = {0: 'Brain',
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

gtvs_mapping_dict = {0: 'GTVp',
                     1: 'GTVnd'}


def convert_individual_organs_to_4d_mha(input_dir="/path/of/45organs/folder", output_path="/path/of/4dmha/"):
    patient_results = []
    spacing = None
    for index in oars_mapping_dict.keys():
        organ_itk = sitk.ReadImage(os.path.join(
            input_dir, "{}.nii.gz".format(oars_mapping_dict[index])))
        organ_arr = sitk.GetArrayFromImage(organ_itk)
        spacing = organ_itk.GetSpacing()
        patient_results.append(organ_arr)
    
    # the following part is very important, please save your results as follows.
    stacked_oars = []
    for each_organ in patient_results:
        # isVector must be set to False!!!
        stacked_oars.append(sitk.GetImageFromArray(each_organ, isVector=False))
    output_itk = sitk.JoinSeries(stacked_oars)
    new_spacing = (spacing[0], spacing[1], spacing[2], 1)
    output_itk.SetSpacing(new_spacing)
    print(output_itk.GetSize())
    # The last parameter must be True
    sitk.WriteImage(output_itk, output_path, True)
    print("Conversion Finished")


def convert_individual_gtvs_to_4d_mha(input_dir="/path/of/45organs/folder", output_path="/path/of/4dmha/"):
    patient_results = []
    spacing = None
    for index in gtvs_mapping_dict.keys():
        gtv_itk = sitk.ReadImage(os.path.join(
            input_dir, "{}.nii.gz".format(gtvs_mapping_dict[index])))
        gtv_arr = sitk.GetArrayFromImage(gtv_itk)
        spacing = gtv_itk.GetSpacing()
        patient_results.append(gtv_arr)
    # the following part is very important, please save your results as follows.
    stacked_gtvs = []
    for each_gtv in patient_results:
        # isVector must be set to False!!!
        stacked_gtvs.append(sitk.GetImageFromArray(each_gtv, isVector=False))
    output_itk = sitk.JoinSeries(stacked_gtvs)
    new_spacing = (spacing[0], spacing[1], spacing[2], 1)
    output_itk.SetSpacing(new_spacing)
    print(output_itk.GetSize())
    # The last parameter must be True
    sitk.WriteImage(output_itk, output_path, True)
    print("Conversion Finished")


convert_individual_gtvs_to_4d_mha(
    "./gtvs_output_example/", "gtvs_output_example.mha")
convert_individual_organs_to_4d_mha(
    "./oars_output_example/", "oars_output_example.mha")
