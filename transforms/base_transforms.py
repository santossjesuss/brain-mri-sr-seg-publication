import torch.nn.functional as F

class BaseTransforms():
    def __init__(self, scale_factor):
        self.scale_factor = scale_factor
        self.downsample_factor = self._calc_downsample_factor(scale_factor)
        self._bilinear_interpolation = 'bilinear'
        self._bicubic_interpolation = 'bicubic'
        self._nearest_neighbours = 'nearest'

    def normalize_image(self, image):
        return image.float() / 255.0

    def normalize_binary_mask(self, mask):
        return (mask > 0).long()

    def upsample_image_bicubic(self, image):
        image = image.unsqueeze(0)
        upscaled_image = F.interpolate(image, scale_factor=self.scale_factor, mode=self._bicubic_interpolation)

        return upscaled_image.squeeze(0)

    def downsample_image_bicubic(self, image):
        image = image.unsqueeze(0)
        downscaled_image = F.interpolate(image, scale_factor=self.downsample_factor, mode=self._bicubic_interpolation)

        return downscaled_image.squeeze(0)

    def downsample_mask_nearest_neighbours(self, mask):
        mask = mask.unsqueeze(0)
        downscaled_mask = F.interpolate(mask, scale_factor=self.downsample_factor, mode=self._nearest_neighbours)

        return downscaled_mask.squeeze(0)

    def upsample_mask_logits_bilinear(self, mask_logits):
        scaled_mask = F.interpolate(mask_logits, scale_factor=self.scale_factor, mode=self._bilinear_interpolation)

        return scaled_mask
    
    def _calc_downsample_factor(self, scale_factor):
        return 1 / scale_factor