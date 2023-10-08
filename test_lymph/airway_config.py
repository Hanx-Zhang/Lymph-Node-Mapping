# -*- coding: utf-8 -*-

config = {
    "in_channels": 1,
    "out_channels": 2,
    "finalsigmoid": 1,
    "fmaps_degree": 16,
    "fmaps_layer_number": 4,
    "layer_order": "cip",
    "layer_order_for_model2": "cpi",
    "GroupNormNumber": 4,
    "device": "cuda:0",
    "weight_path1": 'airway_model1.pth',
    "weight_path2": 'test_lymph/airway_model2.pth',
    "weight_path3": 'airway_model3.pth',
    "roi_size": (128, 224, 304),
    # "roi_size": (128, 448, 576),
    "sw_batch_size": 1,
    # "overlap": 0.75,
    "overlap": 0.75,
    "mode": 'gaussian',
    "sigma_scale": 0.25,
    "KeepLargestConnectedComponent": True,
    "use_HU_window":True
}
