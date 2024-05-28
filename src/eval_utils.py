GPU = 0
import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU)
from models import TransformerModel
from tasks import get_task_sampler
from samplers import get_data_sampler
import torch
import numpy as np
import matplotlib.pyplot as plt
from models import *
from samplers import *

from eval_constants import *

import time


# def plot_clamped_and_predicted(model):
#     sampler = UniformSampler(n_dims=1)
#     get_clamped_task = lambda : get_task_sampler( "clamped_chebyshev", 1, 1)()
#     get_orig_task = lambda : get_task_sampler( "chebyshev_kernel_linear_regression", 1, 1)()

#     num_xs_to_pred = 100

#     xs_context = sampler.sample_xs((255,1))
#     xs_to_pred = sampler.sample_xs((num_xs_to_pred,1))

#     ys_clamped_context = clamped_task.evaluate(xs_context, noise=False, separate_noise=False)
#     ys_clamped_to_pred = clamped_task.evaluate(xs_to_pred, noise=False, separate_noise=False)

#     xs_gt, ys_gt = np.concatenate(xs_context.squeeze(-1), xs_to_pred.squeeze(-1)), np.concatenate(ys_clamped_context.squeeze(-1), ys_clamped_to_pred.squeeze(-1))

#     # plot the grount truth clamped polynomial
#     plt.plot(xs_gt, ys_gt, label="Ground Truth Clamped Polynomial")

#     predicted_ys_on_context = []
#     for i in range(0, num_xs_to_pred):
#         xs = xs_context.squeeze(-1) + xs_to_pred[i]
#         ys = ys_clamped_context.squeeze(-1) + ys_clamped_to_pred[i]
#         pred = model(xs.unsqueeze(-1), ys.unsqueeze(-1))
#         predicted_ys_on_context.append(pred)
    
#     plt.plot(xs_gt, predicted_ys_on_context, label="Predicted Clamped Polynomial")


    





def simpler_eval_batch(model, 
               use_clamped_y, 
               last_pt_above_thresh, 
               last_pt_use_clamped_y, 
               lowest_degree=2,
               highest_degree=2,
               window_len=256, 
               thresh=0.5,
               b_size=1, 
               smoothing=0,
    ):
    
    get_clamped_task = lambda : get_task_sampler( "clamped_chebyshev", 1, b_size, basis_dim=highest_degree, lowest_degree=lowest_degree, highest_degree=highest_degree)()
    get_orig_task = lambda : get_task_sampler( "chebyshev_kernel_linear_regression", 1, b_size, basis_dim=highest_degree, lowest_degree=lowest_degree, highest_degree=highest_degree)()
    get_ds = lambda : get_data_sampler('gaussian', 1)

    if torch.cuda.is_available() and model.name.split("_")[0] in ["gpt2", "lstm"]:
        device = "cuda"
    else:
        device = "cpu"

    # sample 100 x points
    # keep sampling a batch of 100 x points until we have % 0.5 ys>=percent_clamped
    # then, depending on the context length, select percent_clamped *context_window (x,y) pairs, and randomly select the rest from the remaining points


    if last_pt_above_thresh!=-1 or use_clamped_y!=-1 or last_pt_use_clamped_y!=-1:
        done=False
        while not done: 
            orig_task, clamped_task = get_orig_task(), get_clamped_task()
            data_sampler = get_ds()
            xs = data_sampler.sample_xs(n_points=window_len, b_size=b_size)
            ys_orig, ys_clamped = orig_task.evaluate(xs, noise=False, separate_noise=False), clamped_task.evaluate(xs, noise=False, separate_noise=False)
            
            if last_pt_above_thresh==True:
                done = torch.where(ys_clamped[0]==thresh)[0].size(0) > 0
            elif last_pt_above_thresh==False:
                done = torch.where(ys_clamped[0]!=thresh)[0].size(0) > 0
            else:
                done = True

            # above_thresh_percentage = (torch.where(ys_clamped[0] == thresh)[0]).size(0) / ys_clamped[0].size(0)
            # below_thresh_percentage = (torch.where(ys_clamped[0] != thresh)[0]).size(0) / ys_clamped[0].size(0)
            # assert below_thresh_percentage >=0.1
            # done = above_thresh_percentage>=0.25 and below_thresh_percentage>=0.1
            # done = above_thresh_percentage>=0.95
    


        xs, ys_clamped, ys_orig  = xs.squeeze(0).squeeze(-1), ys_clamped.squeeze(0), ys_orig.squeeze(0) # squeeze first dim = batch dim 
        ys = ys_clamped if use_clamped_y else ys_orig


        # if we want the last pt above the threshold, switch it so that's the case
        if last_pt_above_thresh == -1:
            # hi = -1
            # print("random last pt") 
            if not use_clamped_y and last_pt_use_clamped_y:
                ys[-1] = torch.tensor(thresh) if ys[-1]>thresh else ys[-1]

        elif last_pt_above_thresh == True:
            above_thresh_indices = torch.where(ys_clamped == thresh)[0]
            above_thresh_idx = above_thresh_indices[-1]

            if last_pt_use_clamped_y:
                last_y = torch.tensor([ys_clamped[above_thresh_idx]])
            else:
                last_y = torch.tensor([ys_orig[above_thresh_idx]])

            try:
                ys = torch.cat((ys_clamped[:above_thresh_idx], ys_clamped[above_thresh_idx+1:], last_y))
                xs = torch.cat((xs[:above_thresh_idx], xs[above_thresh_idx+1:], torch.tensor([xs[above_thresh_idx]])))
            except:
                import pdb; pdb.set_trace()
        elif last_pt_above_thresh == False:
            below_thresh_indices = torch.where(ys_clamped != thresh)[0]
            below_thresh_idx = below_thresh_indices[-1]

            last_y = torch.tensor([ys_orig[below_thresh_idx]])

            ys = torch.cat((ys_clamped[:below_thresh_idx], ys_clamped[below_thresh_idx+1:], last_y))
            xs = torch.cat((xs[:below_thresh_idx], xs[below_thresh_idx+1:], torch.tensor([xs[below_thresh_idx]])))
        # add back batch dim 
        xs, ys = xs.unsqueeze(0).unsqueeze(-1), ys.unsqueeze(0)
    else:
        task = get_orig_task()
        data_sampler = get_ds()
        xs = data_sampler.sample_xs(n_points=window_len, b_size=b_size)
        ys = task.evaluate(xs, noise=False, separate_noise=False)

    

    pred = model(xs.to(device), ys.to(device)).detach()

    perturbations = np.arange(-1 * smoothing, smoothing + 0.002, 0.002)
    predictions = torch.zeros(len(perturbations), xs.shape[0], xs.shape[1])
    predictions = pred.cpu() 

    # predictions = predictions[:, window_len-1]


    # ridge predictions 
    ridge_baseline = ChebyshevKernelLeastSquaresModelWithRidge(basis_dim=highest_degree, ridge=0.5)
    pred_no_ridge = ridge_baseline.return_trained_model(xs[:, :-1, :], ys[:, :-1])(xs[:, -1, :])

    return ys[:, window_len-1 ], predictions, pred_no_ridge


# window_len-1

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

    if not percent_above_thresh:
        # no need for any special sampling, just sample xs and ys randomly
        task = get_orig_task()
        data_sampler = get_ds()
        xs = data_sampler.sample_xs(n_points=window_len, b_size=b_size)
        ys = task.evaluate(xs, noise=False, separate_noise=False)
    else:
        num_clamped_needed = int(percent_above_thresh*window_len)
        num_other_needed = int(window_len - num_clamped_needed)

        done = False
        while not done: 
            task = get_clamped_task()
            data_sampler = get_ds()
            
            xs = data_sampler.sample_xs(n_points=100, b_size=b_size)
            ys = task.evaluate(xs, noise=False, separate_noise=False)
            if not use_clamped_y or not last_pt_use_clamped_y:
                orig_task = get_orig_task()
                ys_orig = orig_task.evaluate(xs, noise=False, separate_noise=False)
            
            # print(f'clamped {torch.where(ys[0]==0.5)[0].size(0)}, other {torch.where(ys[0]!=0.5)[0].size(0)}')
            done = torch.where(ys[0]==thresh)[0].size(0) >= num_clamped_needed and torch.where(ys[0]!=thresh)[0].size(0) >= num_other_needed # use 1 bc dont want batch dim
            
        # ys still equals the ys with clamping -- need this to randomly sample the right number of clamped indices
        # select percent_clamped_correct * window_len points
        clamped_indices = torch.where(ys[0]==thresh)[0]
        indices = torch.multinomial(torch.ones(len(clamped_indices)), num_clamped_needed, replacement=False)  # select indices
        clamped_indices = clamped_indices[indices]

        # select the rest randomly
        remaining_indices = torch.where(ys[0]!=thresh)[0]
        indices = torch.multinomial(torch.ones(len(remaining_indices)), num_other_needed, replacement=False)  # select indices
        remaining_indices = remaining_indices[indices]

        # now make ys = the unclamped y values if needed
        if not use_clamped_y:
            ys = ys_orig

        # we use the indices calculated from the clamping to index into the unclamped ys
        clamped_xs = xs[:, clamped_indices]
        clamped_ys = ys[:, clamped_indices]
        
        remaining_xs = xs[:, remaining_indices]
        remaining_ys = ys[:, remaining_indices]

        # save last point
        if last_pt_above_thresh:
            last_x= clamped_xs[:, -1]
            if last_pt_use_clamped_y:
                last_y = torch.tensor([thresh])
            else:
                last_y = ys_orig[:, clamped_indices][:, -1]#clamped_ys[:, -1]
            clamped_xs = clamped_xs[:, :-1]
        else:
            last_x, last_y = remaining_xs[:, -1], remaining_ys[:, -1]
            remaining_xs = remaining_xs[:, :-1]

        # combine clamped & remaining
        xs = torch.cat([clamped_xs, remaining_xs], dim=1)
        ys = torch.cat([clamped_ys, remaining_ys], dim=1)

        # shuffle order of xs and ys(but together)
        perm = torch.randperm(xs.shape[1])
        xs = xs[:, perm]
        ys = ys[:, perm]

        # add back last point
        xs = torch.cat((xs, last_x[:, :, None]), dim=1)
        ys = torch.cat((ys, last_y[:, None]), dim=1)

    pred = model(xs.to(device), ys.to(device)).detach()

    perturbations = np.arange(-1 * smoothing, smoothing + 0.002, 0.002)
    predictions = torch.zeros(len(perturbations), xs.shape[0], xs.shape[1])
    predictions = pred.cpu() # (64, 41)

    predictions = predictions[:, window_len-1]

    return ys[:,  window_len-1], predictions

def build_model(n_embd=256, n_layer=12, n_head=8, n_positions=256):
    model = TransformerModel(
        n_dims=1,
        n_positions=n_positions,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
    )
    return model

def get_model(ckpt_path, n_embd=256, n_layer=12, n_head=8, n_positions=256): 
    model = build_model(n_embd=n_embd, n_layer=n_layer, n_head=n_head, n_positions=n_positions)
    torch.cuda.set_device(0)
    model.cuda()

    base_model = os.path.join(ckpt_path, "state.pt")
    state = torch.load(base_model, map_location='cuda:0')
    model.load_state_dict(state["model_state_dict"])
    b_size = 1

    return model

def get_plot(x_axis, y_axis, title, msg="", x_label="", y_label=""): 
    plt.xlabel(x_label) # plt.xlabel('% Context Clamped')
    plt.ylabel(y_label) # plt.ylabel('MSE (averaged across 500 points)')
    plt.title(title) # plt.title('Finetuned Model: MSE vs % Clamped Examples in Context')
    # # add a note on the graph
    plt.text(0.6, 0.92, msg)# plt.text(0.6, 0.92, 'Last point SHOULD be clamped')
    plt.plot(x_axis, y_axis) # plt.plot(percent_clamped, mse)
    return plt


def get_pretty_plot(x_axis, y_axes, title, legend=[], x_label="", y_label=""): 
    SIZE_DEFAULT = 12
    SIZE_LARGE = 14
    plt.rc("font", family="Roboto")  # controls default font
    plt.rc("font", weight="light")  # controls default font
    plt.rc("font", size=SIZE_DEFAULT)  # controls default text sizes
    plt.rc("axes", titlesize=SIZE_LARGE)  # fontsize of the axes title
    plt.rc("axes", labelsize=SIZE_LARGE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels
    
    fig, ax = plt.subplots(figsize=(6, 5))
    # Define font sizes
    

    colors = ["#2B2F42", "#8D99AE", "#EF233C"]

    # Hide the all but the bottom spines (axis lines)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    ax.spines["bottom"].set_bounds(min(x_axis), max(x_axis))


    # # add a note on the graph
    # ax.text(0.6, 0.92, msg)# plt.text(0.6, 0.92, 'Last point SHOULD be clamped')

    for y_axis, name in zip(y_axes, legend):
        color = colors.pop()
        ax.plot(x_axis, y_axis, color=color)
        ax.text(
            x_axis[-1]* 1.01,
            y_axis[-1],
            name,
            color=color,
            fontweight="bold",
            horizontalalignment="left",
            verticalalignment="center",
        )

    ax.set_xlabel(x_label, fontweight="normal") # plt.xlabel('% Context Clamped')
    ax.set_ylabel(y_label, fontweight="normal") # plt.ylabel('MSE (averaged across 500 points)')
    ax.set_title(title) # plt.title('Finetuned Model: MSE vs % Clamped Examples in Context')

    return ax

import matplotlib.pyplot as plt

def get_pretty_plot2(x_axes, y_axes, title, legend=[], x_label="", y_label="", normalize=False):
    SIZE_DEFAULT = 12
    SIZE_LARGE = 14
    plt.rc("font", family="Roboto")
    plt.rc("font", weight="light")
    plt.rc("font", size=SIZE_DEFAULT)
    plt.rc("axes", titlesize=SIZE_LARGE)
    plt.rc("axes", labelsize=SIZE_LARGE)
    plt.rc("xtick", labelsize=SIZE_DEFAULT)
    plt.rc("ytick", labelsize=SIZE_DEFAULT)

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["#6929c4", "#9f1853", "#198038", "#b28600"]

    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")

    # Normalize each y_axis to scale from 0 to 1
    if normalize:
        y_axes = [(y - min(y)) / (max(y) - min(y)) if max(y) != min(y) else y for y in y_axes]
    

    # Plot each series with its own x_axis
    for x_axis, y_axis, name in zip(x_axes, y_axes, legend):
        color = colors.pop(0)
        ax.plot(x_axis, y_axis, color=color, label=name)
        # ax.text(
        #     x_axis[-1] * 1.01,
        #     y_axis[-1],
        #     name,
        #     color=color,
        #     fontweight="bold",
        #     horizontalalignment="left",
        #     verticalalignment="center",
        # )

    ax.set_xlabel(x_label, fontweight="normal")
    ax.set_ylabel(y_label, fontweight="normal")
    ax.set_title(title, fontweight='normal')
    ax.legend(prop={'weight': 'normal'})

    return ax


def get_baselines(bs):
    x_axes = []
    y_axes = []
    legends = []
    
    for name, b in bs.items(): 
        x, y = b[:, 0], b[:, 1]
        x_axes.append(x)
        y_axes.append(y)
        legends.append(name)

    return {
        'x': x_axes,
        'y': y_axes,
        'legends': legends
    }

def get_baselines_1a(): 
    baselines = {
        "Translation": TRANSLATION_ENGLISH_TAMIL,
        "Summarization": SUMMARIZATION_GEMXSUM,
        "Planning": PLANNING,
    }

    return get_baselines(baselines)


def get_baselines_1bc(): 
    baselines ={
        "Sentiment Analysis (Flipped)": SENTIMENT_ANALYSIS_FLIPPED,
        "Sentiment Analysis (Abstract)": SENTIMENT_ANALYSIS_ABSTRACT,
    }

    return get_baselines(baselines)

    
        