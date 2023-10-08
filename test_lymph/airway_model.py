# -*- coding: utf-8 -*-

from test_lymph.airway_network import UNet3D, normalize_CT, lumTrans
from test_lymph.airway_config import config
from test_lymph.utils import InnerTransformer, sliding_window_inference

import torch
import numpy as np


class AirwayExtractionModel(object):
    def __init__(self):
        self.config = config
        self.device = []
        self.device.append(self.config['device'])

        self.net_for_model2 = UNet3D(
            in_channels=self.config['in_channels'],
            out_channels=self.config['out_channels'],
            finalsigmoid=self.config['finalsigmoid'],
            fmaps_degree=self.config['fmaps_degree'],
            fmaps_layer_number=self.config['fmaps_layer_number'],
            layer_order=self.config['layer_order_for_model2'],
            GroupNormNumber=self.config['GroupNormNumber'],
            device=self.device
        )

    @torch.no_grad()
    def predict(self, image: np.ndarray):
        self.net_for_model2.eval()
        if self.config['use_HU_window']:
            image = lumTrans(image)
        image = normalize_CT(image)
        image = InnerTransformer.ToTensor(image)
        image = InnerTransformer.AddChannel(image)
        image = InnerTransformer.AddChannel(image)
        image = image.to(self.device[0])

        self.net_for_model2.load_state_dict(
            torch.load(self.config['weight_path2'], map_location=lambda storage, loc: storage.cuda(0)))
        pred2 = sliding_window_inference(
            inputs=image,
            roi_size=self.config['roi_size'],
            sw_batch_size=self.config['sw_batch_size'],
            predictor=self.net_for_model2,
            overlap=self.config['overlap'],
            mode=self.config['mode'],
            sigma_scale=self.config['sigma_scale']
        )
        pred2 = InnerTransformer.AsDiscrete(pred2[:, 1, ...])

        return pred2