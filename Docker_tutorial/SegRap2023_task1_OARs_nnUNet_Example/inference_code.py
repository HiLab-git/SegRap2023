import os
import sys
o_path = os.getcwd()
sys.path.append(o_path)
import shutil
from multiprocessing import Pool

import numpy as np
import torch
from copy import deepcopy
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.inference.predict import preprocess_multithreaded
from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax
from nnunet.postprocessing.connected_components import load_postprocessing, load_remove_save
from nnunet.training.model_restore import load_model_and_checkpoint_files
from torch import cuda
from torch.nn import functional as F


def predict_cases_segrap2023(model, list_of_lists, output_filenames, folds, save_npz=False, num_threads_preprocessing=6,
                             num_threads_nifti_save=2, segs_from_prev_stage=None, do_tta=False,
                             overwrite_existing=False, step_size=0.5, checkpoint_name="model_final_checkpoint",
                             disable_postprocessing: bool = False):
    assert len(list_of_lists) == len(output_filenames)
    if segs_from_prev_stage is not None:
        assert len(segs_from_prev_stage) == len(output_filenames)

    pool = Pool(num_threads_nifti_save)
    results = []

    cleaned_output_files = []
    for o in output_filenames:
        dr, f = os.path.split(o)
        if len(dr) > 0:
            maybe_mkdir_p(dr)
        if not f.endswith(".nii.gz"):
            f, _ = os.path.splitext(f)
            f = f + ".nii.gz"
        cleaned_output_files.append(join(dr, f))

    if not overwrite_existing:
        print("number of cases:", len(list_of_lists))
        # if save_npz=True then we should also check for missing npz files
        not_done_idx = [i for i, j in enumerate(cleaned_output_files) if
                        (not isfile(j)) or (save_npz and not isfile(j[:-7] + '.npz'))]

        cleaned_output_files = [cleaned_output_files[i] for i in not_done_idx]
        print(list_of_lists)
        list_of_lists = [list_of_lists[i] for i in not_done_idx]
        if segs_from_prev_stage is not None:
            segs_from_prev_stage = [segs_from_prev_stage[i]
                                    for i in not_done_idx]

        print("number of cases that still need to be predicted:",
              len(cleaned_output_files))

    print("emptying cuda cache")
    torch.cuda.empty_cache()

    print("loading parameters for folds,", folds)
    trainer, params = load_model_and_checkpoint_files(model, folds, mixed_precision=True,
                                                      checkpoint_name=checkpoint_name)

    print("starting preprocessing generator")
    preprocessing = preprocess_multithreaded(trainer, list_of_lists, cleaned_output_files, num_threads_preprocessing,
                                             segs_from_prev_stage)
    print("starting prediction...")
    all_output_files = []
    with torch.no_grad():
        for preprocessed in preprocessing:
            output_filename, (d, dct) = preprocessed
            all_output_files.append(all_output_files)
            if isinstance(d, str):
                data = np.load(d)
                os.remove(d)
                d = data

            # we need to be able to del it if things fail (just in case)
            softmax = None

            try:
                print("predicting", output_filename)
                print(f"attempting all_in_gpu {True}")
                trainer.load_checkpoint_ram(params[0], False)
                softmax = trainer.predict_preprocessed_data_return_seg_and_softmax(
                    d, do_mirroring=do_tta, mirror_axes=trainer.data_aug_params[
                        'mirror_axes'], use_sliding_window=True,
                    step_size=step_size, use_gaussian=True, all_in_gpu=True,
                    mixed_precision=True)[1]
                for p in params[1:]:
                    trainer.load_checkpoint_ram(p, False)
                    softmax += trainer.predict_preprocessed_data_return_seg_and_softmax(
                        d, do_mirroring=do_tta, mirror_axes=trainer.data_aug_params[
                            'mirror_axes'], use_sliding_window=True,
                        step_size=step_size, use_gaussian=True, all_in_gpu=True,
                        mixed_precision=True)[1]
            except RuntimeError:  # out of gpu memory
                del softmax
                cuda.empty_cache()
                print(
                    f"\nGPU AGGREGATION FAILED FOR CASE {os.path.basename(output_filename)} DUE TO OUT OF MEMORY, falling back to all_in_gpu False\n")
                trainer.load_checkpoint_ram(params[0], False)
                softmax = trainer.predict_preprocessed_data_return_seg_and_softmax(
                    d, do_mirroring=do_tta, mirror_axes=trainer.data_aug_params[
                        'mirror_axes'], use_sliding_window=True,
                    step_size=step_size, use_gaussian=True, all_in_gpu=False,
                    mixed_precision=True)[1]

                for p in params[1:]:
                    trainer.load_checkpoint_ram(p, False)
                    softmax += trainer.predict_preprocessed_data_return_seg_and_softmax(
                        d, do_mirroring=do_tta, mirror_axes=trainer.data_aug_params[
                            'mirror_axes'], use_sliding_window=True,
                        step_size=step_size, use_gaussian=True, all_in_gpu=False,
                        mixed_precision=True)[1]
            cuda.empty_cache()

            if len(params) > 1:
                softmax /= len(params)

            transpose_forward = trainer.plans.get('transpose_forward')
            if transpose_forward is not None:
                transpose_backward = trainer.plans.get('transpose_backward')
                # softmax = softmax.transpose([0] + [i + 1 for i in transpose_backward])

            # resampling linearly on GPU
            torch.cuda.empty_cache()
            target_shape = dct.get('size_after_cropping')
            target_shape = [target_shape[i] for i in transpose_forward]
            if not isinstance(softmax, torch.Tensor):
                softmax = torch.from_numpy(softmax)
            try:
                with torch.no_grad():
                    softmax_resampled = torch.zeros((softmax.shape[0], *target_shape), dtype=torch.half,
                                                    device='cuda:0')
                    if not softmax.device == torch.device('cuda:0'):
                        softmax_gpu = softmax.to(torch.device('cuda:0'))
                    else:
                        softmax_gpu = softmax
                    for c in range(len(softmax)):
                        softmax_resampled[c] = \
                            F.interpolate(
                                softmax_gpu[c][None, None], size=target_shape, mode='trilinear')[0, 0]
                    del softmax, softmax_gpu
                    softmax_resampled = softmax_resampled.cpu().numpy()
            except RuntimeError:
                # gpu failed, try CPU
                print(
                    f"\nGPU RESAMPLING FAILED FOR CASE {os.path.basename(output_filename)} DUE TO OUT OF MEMORY, falling back to CPU\n")

                if not softmax.device == torch.device('cpu'):
                    softmax_cpu = softmax.to(torch.device('cpu')).float()
                else:
                    softmax_cpu = softmax

                torch.cuda.empty_cache()
                with torch.no_grad():
                    softmax_resampled = torch.zeros(
                        (softmax.shape[0], *target_shape))
                    # depending on where we crash this has already been converted or not
                    if not isinstance(softmax, torch.Tensor):
                        softmax = torch.from_numpy(softmax)
                    for c in range(len(softmax)):
                        softmax_resampled[c] = \
                            F.interpolate(
                                softmax_cpu[c][None, None], size=target_shape, mode='trilinear')[0, 0]
                    del softmax, softmax_cpu
                    softmax_resampled = softmax_resampled.half().numpy()
            torch.cuda.empty_cache()
            #####################################
            softmax_resampled = softmax_resampled.transpose(
                [0] + [i + 1 for i in transpose_backward])

            if save_npz:
                npz_file = output_filename[:-7] + ".npz"
            else:
                npz_file = None

            if hasattr(trainer, 'regions_class_order'):
                region_class_order = trainer.regions_class_order
            else:
                region_class_order = None

            """There is a problem with python process communication that prevents us from communicating objects 
            larger than 2 GB between processes (basically when the length of the pickle string that will be sent is 
            communicated by the multiprocessing.Pipe object then the placeholder (I think) does not allow for long 
            enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually 
            patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will 
            then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either 
            filename or np.ndarray and will handle this automatically"""
            bytes_per_voxel = 4
            print(
                f'softmax shape {softmax_resampled.shape}, softmax dtype {softmax_resampled.dtype}')
            if True:
                # if all_in_gpu then the return value is half (float16)
                bytes_per_voxel = 2
            # * 0.85 just to be save
            if np.prod(softmax_resampled.shape) > (2e9 / bytes_per_voxel * 0.85):
                print(
                    "This output is too large for python process-process communication. Saving output temporarily to disk")
                np.save(output_filename[:-7] + ".npy", softmax_resampled)
                softmax_resampled = output_filename[:-7] + ".npy"

            results.append(pool.starmap_async(save_segmentation_nifti_from_softmax,
                                              ((softmax_resampled, output_filename, dct, 1, region_class_order,
                                                None, None,
                                                npz_file, None, False, 1),)
                                              ))

    print("inference done. Now waiting for the segmentation export to finish...")
    _ = [i.get() for i in results]
    # now apply postprocessing
    # first load the postprocessing properties if they are present. Else raise a well visible warning
    if not disable_postprocessing:
        results = []
        pp_file = join(model, "postprocessing.json")
        if isfile(pp_file):
            print("postprocessing...")
            shutil.copy(pp_file, os.path.abspath(
                os.path.dirname(output_filenames[0])))
            # for_which_classes stores for which of the classes everything but the largest connected component needs to be
            # removed
            for_which_classes, min_valid_obj_size = load_postprocessing(
                pp_file)
            results.append(pool.starmap_async(load_remove_save,
                                              zip(output_filenames, output_filenames,
                                                  [for_which_classes] *
                                                  len(output_filenames),
                                                  [min_valid_obj_size] * len(output_filenames))))
            _ = [i.get() for i in results]
        else:
            print("WARNING! Cannot run postprocessing because the postprocessing file is missing. Make sure to run "
                  "consolidate_folds in the output folder of the model first!\nThe folder you need to run this in is "
                  "%s" % model)

    pool.close()
    pool.join()


def check_input_folder_and_return_caseIDs(input_folder, expected_num_modalities):
    print("This model expects %d input modalities for each image" %
          expected_num_modalities)
    files = subfiles(input_folder, suffix=".nii.gz", join=False, sort=True)

    maybe_case_ids = np.unique([i[:-12] for i in files])

    remaining = deepcopy(files)
    missing = []

    assert len(
        files) > 0, "input folder did not contain any images (expected to find .nii.gz file endings)"

    # now check if all required files are present and that no unexpected files are remaining
    for c in maybe_case_ids:
        for n in range(expected_num_modalities):
            expected_output_file = c + "_%04.0d.nii.gz" % n
            if not isfile(join(input_folder, expected_output_file)):
                missing.append(expected_output_file)
            else:
                remaining.remove(expected_output_file)

    print("Found %d unique case ids, here are some examples:" % len(maybe_case_ids),
          np.random.choice(maybe_case_ids, min(len(maybe_case_ids), 10)))
    print("If they don't look right, make sure to double check your filenames. They must end with _0000.nii.gz etc")

    if len(remaining) > 0:
        print("found %d unexpected remaining files in the folder. Here are some examples:" % len(remaining),
              np.random.choice(remaining, min(len(remaining), 10)))

    if len(missing) > 0:
        print("Some files are missing:")
        print(missing)
        raise RuntimeError("missing files in input_folder")

    return maybe_case_ids


def predict_from_folder_segrap2023(model: str, input_folder: str, output_folder: str, folds: 0, part_id:0, num_parts:1):
    """
        here we use the standard naming scheme to generate list_of_lists and output_files needed by predict_cases

    :param model:
    :param input_folder:
    :param output_folder:
    :param folds:
    :param save_npz:
    :param num_threads_preprocessing:
    :param num_threads_nifti_save:
    :param lowres_segmentations:
    :param part_id:
    :param num_parts:
    :param tta:
    :param mixed_precision:
    :param overwrite_existing: if not None then it will be overwritten with whatever is in there. None is default (no overwrite)
    :return:
    """
    maybe_mkdir_p(output_folder)
    # shutil.copy(join(model, 'plans.pkl'), output_folder)

    # assert isfile(join(model, "plans.pkl")
    #               ), "Folder with saved model weights must contain a plans.pkl file"
    expected_num_modalities = load_pickle(
        join(model+"/fold_{}".format(folds), "plans.pkl"))['num_modalities']

    # check input folder integrity
    case_ids = check_input_folder_and_return_caseIDs(
        input_folder, expected_num_modalities)

    output_files = [join(output_folder, i + ".nii.gz") for i in case_ids]
    all_files = subfiles(input_folder, suffix=".nii.gz", join=False, sort=True)
    list_of_lists = [[join(input_folder, i) for i in all_files if i[:len(j)].startswith(j) and
                      len(i) == (len(j) + 12)] for j in case_ids]
    
    return predict_cases_segrap2023(model, list_of_lists[part_id::num_parts], output_files[part_id::num_parts], folds=0)
   

# predict_from_folder_segrap2023("weight/", "images/", "test/", 0, 0, 1)
