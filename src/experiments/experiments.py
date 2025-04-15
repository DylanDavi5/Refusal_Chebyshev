import torch
from eval import get_model_from_run
from tasks import get_task_sampler, ClampedChebyshev, ChebyshevKernelLinearRegression
from samplers import UniformSampler
from plot_utils import basic_plot, plot_checkpoints
from tqdm import tqdm



def mean_squared_error(ys_pred, ys):
    return (ys - ys_pred).square().mean()

def squared_error(ys_pred, ys):
    return (ys - ys_pred).square()

def eval_model(model, task_class, sampler, context_size = 40):
    GPU = 1
    device = torch.device(f"cuda:{GPU}")
    
    num_tasks = 1000
    samples_per_task = 10

    predicted_ys = torch.tensor([])
    actual_ys = torch.tensor([])


    for _ in tqdm(range(num_tasks)):
        task = task_class(n_dims=1, batch_size=1, basis_dim=4, pool_dict=None, different_degrees=False, lowest_degree=1, highest_degree=11)

        xs_context = sampler.sample_xs(context_size, 1)
        xs_to_predict = sampler.sample_xs(samples_per_task, 1)

        ys_context = task.evaluate(xs_context, noise=False, separate_noise=False)
        ys_to_predict = task.evaluate(xs_to_predict, noise=False, separate_noise=False)

        context_x = xs_context.repeat(xs_to_predict.shape[1], 1, 1)
        prompt_x = xs_to_predict.permute(1, 0, 2) # (num predictions, 1, 1)
        input_x = torch.cat([context_x, prompt_x], dim=1) #add our x prediction points to the end of our fixed cntext

        #dont have a dimension size for our output arrays
        context_y = ys_context.repeat(ys_to_predict.shape[1], 1)  # (number of predictions, context size)
        predict_y = ys_to_predict.view(-1, 1)  # (number of predictions, 1)
        input_y = torch.cat([context_y, predict_y], dim=1)
        
        input_x = input_x.to(device)
        input_y = input_y.to(device)
        model.to(device)

        predictions = model(input_x, input_y).detach().cpu().squeeze()

        predicted_ys= torch.cat([predicted_ys, predictions[:,-1]])
        actual_ys= torch.cat([actual_ys, ys_to_predict.cpu().squeeze()])
    

    squared_errors = squared_error(predicted_ys, actual_ys)
    mse = squared_errors.mean()
    std_mse = squared_errors.std(unbiased=True)
    

    #print(squared_errors, squared_errors.shape)

    n = squared_errors.shape[0]
    stderr = std_mse / n**0.5
    ci_95 = 1.96 * stderr

    return mse.item(), (mse - ci_95).item(), (mse + ci_95).item()
        

        
def eval_checkpoints(run_dir, context_size = 40):
    start_checkpoint = 100_000
    end_checkpoint = 1_000_000
    save_frequency = 100_000

    task_class = ChebyshevKernelLinearRegression
    sampler = UniformSampler(n_dims=1)

    results = {"mean" : [],
               "checkpoints": [],
               "low": [],
               "high": []}

    for step in range(start_checkpoint, end_checkpoint + 1, save_frequency):
        print(f"getting results from checkpoint {step}")
        
        results["checkpoints"].append(step)
        
        model, _= get_model_from_run(run_dir, step=step)

        mean,low,high = eval_model(model, task_class, sampler, context_size=context_size)

        results["mean"].append(mean)
        results["low"].append(low)
        results["high"].append(high)
    
    return results





