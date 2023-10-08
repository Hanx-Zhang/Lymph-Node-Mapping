# -*- coding: utf-8 -*-

from test_lymph.lungmask_network import LMInferer
from test_lymph.lungmask_config import config
import SimpleITK as sitk
import torch
import numpy as np


class LungMaskExtractionModel(object):
    def __init__(self):
        self.config = config
        self.device = self.config['device']
        self.net = LMInferer()

        self.lung_boundingbox = None

    @torch.no_grad()
    # def predict(self, image: sitk.Image):
    #     return self.net.apply(image)

    # learn from lobe_model.py
    def predict(self, image: sitk.Image):
        pred = self.net.apply(image)
        if self.config["CalculateLungBoundingbox"]:
            self._locate_lung_boundingbox(pred)
        return pred

    def _locate_lung_boundingbox(self, image: np.ndarray):
        xx, yy, zz = np.where(image)
        self.lung_boundingbox = np.array([[np.min(xx), np.max(xx)], [np.min(yy), np.max(yy)], [np.min(zz), np.max(zz)]])
        margin = self.config['margin_lung_boundingbox']
        self.lung_boundingbox = np.vstack([np.max([[0, 0, 0], self.lung_boundingbox[:, 0] - margin], 0),
                                           np.min([np.array(image.shape), self.lung_boundingbox[:, 1] + margin],
                                                  axis=0).T]).T
        return
