import os
from random import randint
import uuid

from quinine import QuinineArgumentParser
from tqdm import tqdm
import torch
import yaml

from eval import get_run_metrics
from tasks import get_task_sampler
from samplers import get_data_sampler
from curriculum import Curriculum
from schema import schema
from models import build_model
from experiments import eval_checkpoints
import argparse
from plot_utils import basic_plot, plot_checkpoints

import wandb

torch.backends.cudnn.benchmark = True



def main(args):
    return


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Determine what evals to run')

    parser.add_argument('--run_dir', type=str, help='directory of model run')
    #parser.add_argument('--experiment_name', type=str, help='name of the experiment to run')
    parser.add_argument('--eval_checkpoints', help='Boolean. only run the beginning evals', action='store_true')

    
    # Parse the arguments
    args = parser.parse_args()

    #print("experiment name arg", args.experiment_name)
    #experiment = args.experiment_name

    print("run_dir name arg", args.run_dir)
    run_dir = args.run_dir
    print("eval_checkpoints name arg", args.eval_checkpoints)
    eval_ckpts = args.eval_checkpoints

    if eval_ckpts and run_dir:
        print(f"\n\nEvaluating model checkpoints from {run_dir}")
        
        context_sizes = [1, 6, 13, 20, 30, 40]
        metrics = {}
        for size in context_sizes:
            print(f"\nGetting results for context size {size}")
            
            results = eval_checkpoints(run_dir, context_size=size)

            metrics[f"context size {size}"] = results
            

        fig, _ = plot_checkpoints(metrics)
        plot_dir = os.path.join(run_dir, "figures")
        
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        fig.savefig(os.path.join(plot_dir, "model_checkpoints.pdf"), bbox_inches='tight')


    
    '''
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    assert args.model.family in ["gpt2", "lstm"]
    print(f"Running with: {args}")
    '''
