import os

import cv2
import numpy as np
import torch

script_directory = os.path.dirname(os.path.abspath(__file__))


class AutoColorGimp:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "img": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process"

    CATEGORY = "image/postprocessing"

    def process(self, img):

        img = img.squeeze().numpy()
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # From @Canette Ouverture's answer: https://stackoverflow.com/a/56365560/4561887
        balanced_img = np.zeros_like(img)  # Initialize final image
        for i in range(3):  # i stands for the channel index
            hist, bins = np.histogram(img[..., i].ravel(), 256, (0, 256))
            bmin = np.min(np.where(hist > (hist.sum() * 0.0005)))
            bmax = np.max(np.where(hist > (hist.sum() * 0.0005)))
            balanced_img[..., i] = np.clip(img[..., i], bmin, bmax)
            balanced_img[..., i] = ((balanced_img[..., i] - bmin) / (bmax - bmin) * 255)

        img_pix = cv2.cvtColor(balanced_img, cv2.COLOR_BGR2RGB)
        img_pix_t = np.array(img_pix).astype(np.float32) / 255.0
        img_pix_t = torch.from_numpy(img_pix_t)[None,]
        return (img_pix_t,)


NODE_CLASS_MAPPINGS = {
    "AutoColorGimp": AutoColorGimp,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoColorGimp": "AutoColorGimp",
}
