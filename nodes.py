import os

import cv2
import numpy as np
import torch
from scipy.stats import chisquare

script_directory = os.path.dirname(os.path.abspath(__file__))


class AutoColorGimp:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "img": ("IMAGE",),
                "threshold": ("INT", {
                    "default": 145,
                    "min": 0,
                    "max": 99999999,
                    "step": 1,
                    "display": "number"
                })
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process"

    CATEGORY = "image/postprocessing"

    def process(self, img, threshold):

        img_p = img.squeeze().numpy()
        img_p = (img_p * 255).astype(np.uint8)
        img_p = cv2.cvtColor(img_p, cv2.COLOR_RGB2BGR)

        if (threshold == 0) or self.should_normalize(img_p, threshold):
            print("Normalization is recommended.")
            normalized_img = self.normalize_image(img_p)

            img_p = cv2.cvtColor(normalized_img, cv2.COLOR_BGR2RGB)
            img_p = np.array(img_p).astype(np.float32) / 255.0
            img_p = torch.from_numpy(img_p)[None,]

            return (img_p,)
        else:
            # Threshold lower than difference or is 0.
            return (img,)

    def calculate_histogram(self, img, channel_index):
        hist, bins = np.histogram(img[..., channel_index].ravel(), 256, (0, 256))
        return hist

    def compare_histograms(self, hist1, hist2):
        # Add Laplace smoothing to avoid division by zero
        smoothed_hist1 = hist1 + 1
        smoothed_hist2 = hist2 + 1

        # Using Chi-Square distance as an example
        chi2, p = chisquare(smoothed_hist1, smoothed_hist2)
        return chi2

    def normalize_image(self, img):
        balanced_img = img.copy()
        for i in range(3):  # Assuming img is a 3-channel image
            hist, bins = np.histogram(img[..., i].ravel(), 256, (0, 256))
            bmin = np.min(np.where(hist > (hist.sum() * 0.0005)))
            bmax = np.max(np.where(hist > (hist.sum() * 0.0005)))
            balanced_img[..., i] = np.clip(img[..., i], bmin, bmax)
            balanced_img[..., i] = ((balanced_img[..., i] - bmin) / (bmax - bmin) * 255)
        return balanced_img

    def should_normalize(self, img, threshold):
        # Calculate histograms before normalization
        histograms_before = np.array([self.calculate_histogram(img, i) for i in range(3)])

        # Normalize the image
        normalized_img = self.normalize_image(img)

        # Calculate histograms after normalization
        histograms_after = np.array([self.calculate_histogram(normalized_img, i) for i in range(3)])

        # Compare histograms for each channel
        differences = np.array([self.compare_histograms(histograms_before[i], histograms_after[i]) for i in range(3)])
        differences = differences / 100000000
        differences = differences.astype(int)

        # Check if any difference exceeds the threshold
        diff_mean = (np.mean(differences).astype(int))
        if diff_mean <= threshold:
            return True
        else:
            print(f"Difference between normalized and original is too high ({diff_mean}), skipping.")
            return False


NODE_CLASS_MAPPINGS = {
    "AutoColorGimp": AutoColorGimp,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoColorGimp": "AutoColorGimp",
}
