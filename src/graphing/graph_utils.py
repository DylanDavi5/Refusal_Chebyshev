import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np

palette = sns.color_palette("colorblind")

def plot_polynomial(context_x, context_y, prediction_x, prediction_y, models):
    '''
    models: dictionary {"modelname" : model}
    '''
    fig, ax = plt.subplots(1, 1)

    #each entry along batch represents different querey point with the same context
    context_x = context_x.repeat(prediction_x.shape[1], 1, 1)  # (num predictions, context size, 1)
    prompt_x = prediction_x.permute(1, 0, 2)  # (num predictions, 1, 1)
    input_x = torch.cat([context_x, prompt_x], dim=1) #add our x prediction points to the end of our fixed cntext

    #dont have a dimension size for our output arrays
    context_y = context_y.repeat(prediction_y.shape[1], 1)  # (number of predictions, context size)
    predict_y = prediction_y.view(-1, 1)  # (number of predictions, 1)
    input_y = torch.cat([context_y, predict_y], dim=1)

    #put everything on cpu for plotting
    context_x = context_x.detach().cpu()
    context_y = context_y.detach().cpu()
    prediction_x = prediction_x.detach().cpu()
    ax.scatter(context_x.squeeze(), context_y.squeeze(), label="Context points", color="fuchsia",marker='o', s=10)
    color = 0
    for name in models.keys():
        model = models[name]

        predictions = model(input_x, input_y).detach().cpu()

        ax.plot(prediction_x.squeeze(), predictions[:,-1], label=f"{name} Predictions", color=palette[color % 10],)

        color += 1
    ax.legend()

    return fig,ax



