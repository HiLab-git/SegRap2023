import argparse
import numpy as np
from skimage import measure
import SimpleITK as sitk
from collections import OrderedDict

from batchgenerators.augmentations.utils import resize_segmentation
from skimage.transform import resize
from scipy.ndimage import map_coordinates
from batchgenerators.utilities.file_and_folder_operations import *
from collections import OrderedDict
from pymic.io.image_read_write import *
from utils import *

def largestConnectComponent(binaryimg):
    label_image, num = measure.label(binaryimg, background=0, return_num=True)
    areas = [r.area for r in measure.regionprops(label_image)]
    areas.sort()
    if num > 1:
        for region in measure.regionprops(label_image):
            if (region.area < areas[-1]):
                # print(region.area)
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1], coordinates[2]] = 0
    label_image = label_image.astype(np.int8)
    label_image[np.where(label_image > 0)] = 1

    return label_image

def create_nonzero_mask(data, thresh=-500):
    mask = np.zeros_like(data)
    mask[data > thresh] = 1
    nonzero_mask = largestConnectComponent(mask)
    return nonzero_mask


def get_bbox_from_mask(mask, outside_value=0):
    mask_voxel_coords = np.where(mask != outside_value)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]


def crop_to_bbox(image, bbox):
    assert len(image.shape) == 3, "only supports 3d images"
    resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
    return image[resizer]


def crop_to_nonzero(data, seg=None, nonzero_label=-1):
    """

    :param data:
    :param seg:
    :param nonzero_label: this will be written into the segmentation map
    :return:
    """
    nonzero_mask = create_nonzero_mask(data)
    bbox = get_bbox_from_mask(nonzero_mask, 0)

    data = crop_to_bbox(data, bbox)

    return data, bbox

def get_do_separate_z(spacing, anisotropy_threshold=4):
    do_separate_z = (np.max(spacing) / np.min(spacing)) > anisotropy_threshold
    return do_separate_z


def get_lowres_axis(new_spacing):
    axis = np.where(max(new_spacing) / np.array(new_spacing) == 1)[0]  # find which axis is anisotropic
    return axis


def resample_patient(data, data_contrast, seg, original_spacing, target_spacing, order_data=3, order_seg=0, force_separate_z=False,
                     order_z_data=0, order_z_seg=0, separate_z_anisotropy_threshold=4):
    """
    :param data:
    :param seg:
    :param original_spacing:
    :param target_spacing:
    :param order_data:
    :param order_seg:
    :param force_separate_z: if None then we dynamically decide how to resample along z, if True/False then always
    /never resample along z separately
    :param order_z_seg: only applies if do_separate_z is True
    :param order_z_data: only applies if do_separate_z is True
    :param separate_z_anisotropy_threshold: if max_spacing > separate_z_anisotropy_threshold * min_spacing (per axis)
    then resample along lowres axis with order_z_data/order_z_seg instead of order_data/order_seg

    :return:
    """
    assert not (((data is None) or (data_contrast is not None)) and (seg is None))
    if data is not None:
        assert len(data.shape) == 4, "data must be c x y z"
    if data_contrast is not None:
        assert len(data_contrast.shape) == 4, "seg must be c x y z"
    if seg is not None:
        assert len(seg.shape) == 4, "seg must be c x y z"
    

    if data is not None:
        shape = np.array(data[0].shape)
    elif data_contrast is not None:
        shape = np.array(data_contrast[0].shape)
    else:
        shape = np.array(seg[0].shape)
    
    new_shape = np.round(((np.array(original_spacing) / np.array(target_spacing)).astype(float) * shape)).astype(int)
    
    if force_separate_z is not None:
        do_separate_z = force_separate_z
        if force_separate_z:
            axis = get_lowres_axis(original_spacing)
        else:
            axis = None
    else:
        if get_do_separate_z(original_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(original_spacing)
        elif get_do_separate_z(target_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(target_spacing)
        else:
            do_separate_z = False
            axis = None

    if axis is not None:
        if len(axis) == 3:
            # every axis has the spacing, this should never happen, why is this code here?
            do_separate_z = False
        elif len(axis) == 2:
            # this happens for spacings like (0.24, 1.25, 1.25) for example. In that case we do not want to resample
            # separately in the out of plane axis
            do_separate_z = False
        else:
            pass

    if data is not None:
        data_reshaped = resample_data_or_seg(data, new_shape, False, axis, order_data, do_separate_z, order_z=order_z_data)
    else:
        data_reshaped = None

    if data_contrast is not None:
        data_contrast_reshaped = resample_data_or_seg(data_contrast, new_shape, False, axis, order_data, do_separate_z, order_z=order_z_data)
    else:
        data_contrast_reshaped = None

    if seg is not None:
        seg_reshaped = resample_data_or_seg(seg, new_shape, True, axis, order_seg, do_separate_z, order_z=order_z_seg)
    else:
        seg_reshaped = None
    
    return data_reshaped.squeeze(), data_contrast_reshaped.squeeze(), seg_reshaped.squeeze()


def resample_data_or_seg(data, new_shape, is_seg, axis=None, order=3, do_separate_z=False, order_z=0):
    """
    separate_z=True will resample with order 0 along z
    :param data:
    :param new_shape:
    :param is_seg:
    :param axis:
    :param order:
    :param do_separate_z:
    :param cval:
    :param order_z: only applies if do_separate_z is True
    :return:
    """
    assert len(data.shape) == 4, "data must be (c, x, y, z)"
    if is_seg:
        resize_fn = resize_segmentation
        kwargs = OrderedDict()
    else:
        resize_fn = resize
        kwargs = {'mode': 'edge', 'anti_aliasing': False}
    dtype_data = data.dtype
    shape = np.array(data[0].shape)
    new_shape = np.array(new_shape)
    if np.any(shape != new_shape):
        data = data.astype(float)
        if do_separate_z:
            print("separate z, order in z is",
                  order_z, "order inplane is", order)
            assert len(axis) == 1, "only one anisotropic axis supported"
            axis = axis[0]
            if axis == 0:
                new_shape_2d = new_shape[1:]
            elif axis == 1:
                new_shape_2d = new_shape[[0, 2]]
            else:
                new_shape_2d = new_shape[:-1]

            reshaped_final_data = []
            for c in range(data.shape[0]):
                reshaped_data = []
                for slice_id in range(shape[axis]):
                    if axis == 0:
                        reshaped_data.append(
                            resize_fn(data[c, slice_id], new_shape_2d, order, **kwargs))
                    elif axis == 1:
                        reshaped_data.append(
                            resize_fn(data[c, :, slice_id], new_shape_2d, order, **kwargs))
                    else:
                        reshaped_data.append(resize_fn(data[c, :, :, slice_id], new_shape_2d, order,
                                                       **kwargs))
                reshaped_data = np.stack(reshaped_data, axis)
                if shape[axis] != new_shape[axis]:

                    # The following few lines are blatantly copied and modified from sklearn's resize()
                    rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
                    orig_rows, orig_cols, orig_dim = reshaped_data.shape

                    row_scale = float(orig_rows) / rows
                    col_scale = float(orig_cols) / cols
                    dim_scale = float(orig_dim) / dim

                    map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
                    map_rows = row_scale * (map_rows + 0.5) - 0.5
                    map_cols = col_scale * (map_cols + 0.5) - 0.5
                    map_dims = dim_scale * (map_dims + 0.5) - 0.5

                    coord_map = np.array([map_rows, map_cols, map_dims])
                    if not is_seg or order_z == 0:
                        reshaped_final_data.append(map_coordinates(reshaped_data, coord_map, order=order_z,
                                                                   mode='nearest')[None])
                    else:
                        unique_labels = np.unique(reshaped_data)
                        reshaped = np.zeros(new_shape, dtype=dtype_data)

                        for i, cl in enumerate(unique_labels):
                            reshaped_multihot = np.round(
                                map_coordinates((reshaped_data == cl).astype(float), coord_map, order=order_z,
                                                mode='nearest'))
                            reshaped[reshaped_multihot > 0.5] = cl
                        reshaped_final_data.append(reshaped[None])
                else:
                    reshaped_final_data.append(reshaped_data[None])
            reshaped_final_data = np.vstack(reshaped_final_data)
        else:
            # print("no separate z, order", order)
            reshaped = []
            for c in range(data.shape[0]):
                reshaped.append(resize_fn(data[c], new_shape, order, **kwargs)[None])
            reshaped_final_data = np.vstack(reshaped)
        return reshaped_final_data.astype(dtype_data)
    else:
        print("no resampling necessary")
        return data


def normalize_intensity(data, intensity_properties):       
    """
    mean_intensity: mean intensity
    std_intensity: std intensity
    lower_bound: percentile_00_5
    upper_bound: percentile_99_5
    """
    mean_intensity = intensity_properties[1]
    std_intensity = intensity_properties[2]
    lower_bound = intensity_properties[6]
    upper_bound = intensity_properties[5]
    
    data_norm = np.clip(data, lower_bound, upper_bound)
    data_norm = (data_norm - mean_intensity) / std_intensity
        
    return data_norm


class collect_intensity_properties():
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir

    def intensity_properties_stat(self, dir_mask):
        patient_folders_name = os.listdir(self.root_dir)
        w_img, w_img_contrast = [], []
        for patient_folder_name in patient_folders_name:
            patient_folder_path = os.path.join(self.root_dir, patient_folder_name)
            mask_name = dir_mask + '/' + patient_folder_name
            intensity_img, intensity_img_contrast = self.get_intensity_folder(patient_folder_path, mask_name)
            w_img += intensity_img
            w_img_contrast += intensity_img_contrast
        intensity_properties_img = self.compute_stats(w_img)
        intensity_properties_img_contrast = self.compute_stats(w_img_contrast)

        return intensity_properties_img, intensity_properties_img_contrast

    def compute_stats(self, voxels):
        if len(voxels) == 0:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        median = np.median(voxels)
        mean = np.mean(voxels)
        sd = np.std(voxels)
        mn = np.min(voxels)
        mx = np.max(voxels)
        percentile_99_5 = np.percentile(voxels, 99.5)
        percentile_00_5 = np.percentile(voxels, 00.5)
        return median, mean, sd, mn, mx, percentile_99_5, percentile_00_5
    
    def get_voxels_in_foreground(self, data, mask):
        assert data.shape == mask.shape
        data_intensity = data[mask > 0]

        return list(data_intensity)
    
    def get_intensity_folder(self, patient_folder_path, seg_name):
        img = nii2array(patient_folder_path + '/image.nii.gz')
        img_contrast = nii2array(patient_folder_path + '/image_contrast.nii.gz')
        seg =  nii2array(seg_name)
        intensity_img = self.get_voxels_in_foreground(img, seg)
        intensity_img_contrast = self.get_voxels_in_foreground(img_contrast, seg)

        return intensity_img, intensity_img_contrast

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SegRap2023 preprocessing')
    parser.add_argument("--root_path", type=str, default='data_dir/raw_data')    
    parser.add_argument("--root_path_onehot", type=str, default='data_dir/one_hot_label')
    parser.add_argument("--task", type=str, default='OARs', choices=['OARs', 'GTVs'])
    parser.add_argument("--target_spacing", type=list, default=[1.0, 1.0, 3.0])
    args = parser.parse_args()
    root_path, target_spacing = args.root_path, args.target_spacing

    base = os.path.dirname(root_path)    
    """fuse multi-organs into a one-hot label"""
    if args.task == "OARs":
        dir_one_hot_label = args.root_path_onehot + '/Task001_OARs'
        save_path = base + '/Task001_OARs_preprocess'
    elif args.task == "GTVs":
        dir_one_hot_label = args.root_path_onehot + '/Task002_GTVs'
        save_path = base + '/Task002_GTVs_preprocess'

    path_preprocessed_image   = save_path + '/image'
    path_preprocessed_image_contrast   = save_path + '/image_contrast'
    path_preprocessed_label   = save_path + '/label'
    maybe_mkdir_p(path_preprocessed_image)
    maybe_mkdir_p(path_preprocessed_image_contrast)
    maybe_mkdir_p(path_preprocessed_label)
    
    """get data intensity properties"""    
    json_dict = {}
    get_intensity_properties = collect_intensity_properties(root_path)
    targets_intensity_properties_image, targets_intensity_properties_image_contrast = get_intensity_properties.intensity_properties_stat(dir_one_hot_label)
    json_dict['image_' + args.task] = np.array(targets_intensity_properties_image).tolist()
    json_dict['image_contrast_' + args.task] = np.array(targets_intensity_properties_image_contrast).tolist()
    save_json(json_dict, os.path.join(base, "SegRap2023_intensity_" + args.task + ".json"))


    """target class for fuse label"""
    json_dict_shape = {}
    patient_names = os.listdir(root_path)
    for patient_name in patient_names:       
        """load image and one-hot label"""
        img_obj = sitk.ReadImage("{}/{}/image.nii.gz".format(root_path, patient_name))
        image = sitk.GetArrayFromImage(img_obj)
        origin, spacing, direction = img_obj.GetOrigin(), img_obj.GetSpacing(), img_obj.GetDirection()
        raw_shape = image.shape
        image_contrast = nii2array("{}/{}/image_contrast.nii.gz".format(root_path, patient_name))
        seg = nii2array("{}/{}.nii.gz".format(dir_one_hot_label, patient_name))


        """resample data"""
        image, image_contrast, seg = np.expand_dims(image, 0), np.expand_dims(image_contrast, 0), np.expand_dims(seg, 0)
        spacing_transpose = (spacing[2], spacing[1], spacing[0])
        target_spacing_transpose = (target_spacing[2], target_spacing[1], target_spacing[0])
        image, image_contrast, seg = resample_patient(image, image_contrast, seg, spacing_transpose, target_spacing_transpose)
        

        """crop data"""
        image, bbox = crop_to_nonzero(image)
        image_contrast = crop_to_bbox(image_contrast, bbox)
        seg = crop_to_bbox(seg, bbox)
        cropped_shape = image.shape
        
        
        """normalize data based on intensity properties"""
        image = normalize_intensity(image, targets_intensity_properties_image)
        image_contrast = normalize_intensity(image_contrast, targets_intensity_properties_image_contrast)        
        
        target_origin = [origin[0] + target_spacing[0] * bbox[2][0], origin[1] + target_spacing[1] * bbox[1][0], origin[2] + target_spacing[2] * bbox[0][0]]
        
        save_nii(image, '{}/{}.nii.gz'.format(path_preprocessed_image, patient_name), spacing=target_spacing, origin=target_origin, direction=direction)
        save_nii(image_contrast, '{}/{}.nii.gz'.format(path_preprocessed_image_contrast, patient_name), spacing=target_spacing, origin=target_origin, direction=direction)
        save_nii(seg, '{}/{}.nii.gz'.format(path_preprocessed_label, patient_name), spacing=target_spacing, origin=target_origin, direction=direction)
        
        """save preprocessing parameters for each case"""
        json_dict_shape[patient_name] = [spacing_transpose, target_spacing_transpose, origin, direction, raw_shape, cropped_shape, bbox]
        
    save_json(json_dict_shape, os.path.join(base, "SegRap2023_dataset.json"))