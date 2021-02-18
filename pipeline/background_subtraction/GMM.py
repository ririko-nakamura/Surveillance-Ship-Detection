import pickle

import numpy as np
from sklearn.mixture import GaussianMixture

from .base import BackgroundSubtractorBase

class GMM_BGS(BackgroundSubtractorBase):

    def __init__(self, saved_model=None, n_components=8):
        if saved_model is None:
            self.model = GaussianMixture(n_components)
        else:
            with open(saved_model, "rb") as input_file:
                self.model = pickle.load(input_file)

    def fit(self, dataset):
        bg_elements = []
        for img, bg_mask, _ in dataset:
            bg_elements.append(img[np.transpose(np.nonzero(bg_mask))])
        bg_elements = np.hstack(*bg_elements)
        self.model.fit(bg_elements)

    def save_model(self, path):
        with open(path, "wb") as output_file: 
            pickle.dump(self.model, output_file)   
        
    def apply(self, img):
        mask = np.full(img.shape, 255, dtype=np.uint8)
        for model in self.bg_models:
            for y in img.shape[0]:
                for x in img.shape[1]:
                    if model.predict(img[y, x]):
                        mask[y, x] = 0
        return mask

class GMM_BGS_With_Horizon(BackgroundSubtractorBase):

    def __init__(self):
        self.sky_bgs = AdaptiveGMM_BGS()
        self.sea_bgs = AdaptiveGMM_BGS()

    def fit(self, dataset):
        img_list, bg_mask_list, horizon_list = dataset
        #TODO

    def apply(self, img, horizon):
        sky_mask = self.sky_bgs.apply(img)
        sea_mask = self.sea_bgs.apply(img)
        #TODO: merge masks