import SimpleITK as sitk

def nii2array(path):
    mask_itk_ref = sitk.ReadImage(path)
    mask_arr_ref = sitk.GetArrayFromImage(mask_itk_ref)
    return mask_arr_ref

def save_nii(data, save_name, spacing=None, origin=None, direction=None):
    img = sitk.GetImageFromArray(data)
    if (spacing is not None):
        img.SetSpacing(spacing)
    if (origin is not None):
        img.SetOrigin(origin)
    if (origin is not None):
        img.SetDirection(direction)
    sitk.WriteImage(img, save_name)

