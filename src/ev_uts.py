GPU = 2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU)
from models import TransformerModel
from tasks import get_task_sampler
from samplers import get_data_sampler
import torch
import numpy as np
import matplotlib.pyplot as plt

from eval_constants import *

import time


def eval_batch(model, 
               percent_above_thresh=None, 
               use_clamped_y=True, 
               last_pt_above_thresh=True, 
               last_pt_use_clamped_y=True, 
               window_len=256, 
               thresh=0.5,
               b_size=1, 
               smoothing=0
    ):
    
    get_clamped_task = lambda : get_task_sampler( "clamped_chebyshev", 1, b_size)()
    get_orig_task = lambda : get_task_sampler( "chebyshev_kernel_linear_regression", 1, b_size)()
    get_ds = lambda : get_data_sampler('gaussian', 1)

    if torch.cuda.is_available() and model.name.split("_")[0] in ["gpt2", "lstm"]:
        device = "cuda"
    else:
        device = "cpu"

    # sample 100 x points
    # keep sampling a batch of 100 x points until we have % 0.5 ys>=percent_clamped
    # then, depending on the context length, select percent_clamped *context_window (x,y) pairs, and randomly select the rest from the remaining points

    orig_task, clamped_task = get_orig_task(), get_clamped_task()
    data_sampler = get_ds()
    xs = data_sampler.sample_xs(n_points=window_len, b_size=b_size)
    ys_orig, ys_clamped = orig_task.evaluate(xs, noise=False, separate_noise=False), clamped_task.evaluate(xs, noise=False, separate_noise=False)

    ys = ys_clamped if use_clamped_y else ys_orig

    # if we want the last pt above the threshold, switch it so that's the case
    if last_pt_above_thresh == -1:
        print("random last pt") 
    elif last_pt_above_thresh == True:
        above_thresh_indices = torch.where(ys_clamped[0] == thresh)[0]
        above_thresh_idx = above_thresh_indices[-1]

        if last_pt_use_clamped_y:
            last_y = torch.tensor([ys_clamped[above_thresh_idx]])
        else:
            last_y = torch.tensor([ys_orig[above_thresh_idx]])

        ys = torch.cat((ys_clamped[:above_thresh_idx], ys_clamped[above_thresh_idx+1:], last_y))
        xs = torch.cat((xs[:above_thresh_idx], xs[above_thresh_idx+1:], torch.tensor(xs[above_thresh_idx])))
    elif last_pt_above_thresh == False:
        below_thresh_indices = torch.where(ys_clamped[0] != thresh)[0]
        below_thresh_idx = below_thresh_indices[-1]

        last_y = torch.tensor([ys_orig[below_thresh_idx]])

        ys = torch.cat((ys_clamped[:below_thresh_idx], ys_clamped[below_thresh_idx+1:], last_y))
        xs = torch.cat((xs[:below_thresh_idx], xs[below_thresh_idx+1:], torch.tensor(xs[below_thresh_idx])))

    pred = model(xs.to(device), ys.to(device)).detach()

    perturbations = np.arange(-1 * smoothing, smoothing + 0.002, 0.002)
    predictions = torch.zeros(len(perturbations), xs.shape[0], xs.shape[1])
    predictions = pred.cpu() # (64, 41)

    predictions = predictions[:, window_len-1]

    return ys[:,  window_len-1], predictions