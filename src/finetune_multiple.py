import os
import subprocess
import yaml
import uuid

def create_config_file(base_config, variation, index, subdirectory='conf/finetune'):
    
    run_id = str(uuid.uuid4())
    subdirectory = os.path.join(subdirectory, run_id)

    # Ensure the subdirectory exists
    os.makedirs(subdirectory, exist_ok=True)
    
    # Construct the file path within the subdirectory
    config_filename = os.path.join(subdirectory, f'config_{index}.yaml')
    
    # Update the configuration and dump it into the file
    config = base_config.copy()
    config['training'].update(variation['training'])
    config['model'].update(variation['model'])
    config['alignment'].update(variation['alignment'])
    with open(config_filename, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
    return config_filename
    

def run_training(config_filename):
    # Adjust the command according to your specific needs
    command = f'python finetune.py --config {config_filename}'
    subprocess.run(command, shell=True)

def main():
    # Base configuration settings
    base_config = {
        "model": {
            "n_dims": 1,
            "n_positions": 256,
            "family": "gpt2",
            "n_embd": 512,
            "n_layer": 24,
            "n_head": 16
        },
        "training":{
            #"resume_id": "go_time",
            "task": "clamped_chebyshev",
            "data": "uniform",
            "task_kwargs": {"basis_dim": 11, "different_degrees": True, "lowest_degree": 1},
            "batch_size": 32,
            "learning_rate": 0.00005,
            "save_every_steps": 50000,
            "keep_every_steps": 50000,
            "train_steps": 1000000,
            "curriculum":{
                "dims":{
                    "start": 1,
                    "end": 1,
                    "inc": 1,
                    "interval": 2000
                },
                "points":{
                    "start": 5,
                    "end": 256,
                    "inc": 1,
                    "interval": 100
                },
                "deg":{ 
                    "start": 11,
                    "end": 11,
                    "inc": 0,
                    "interval": 5000001
                }
            }
        },
        "out_dir": "../models/finetune_256",
        "alignment": {
            "base_model": "/home/riadoshi/alignment/Alignment/models/train_multiple/c3b57089-5466-4a2d-a9a7-d82eff45732a" 
        },
        "wandb":{
            "name": "ft_256pts",
            "project": "alignment",
            "entity": "rdoshi21",
            "notes":"", 
            "log_every_steps": 100
        }
    }


    # Variations you want to test
    variations = [
        # medium 256 model
        {"model": {
            "n_embd": 256,
            "n_layer": 12,
            "n_head": 8,
        },
        "training":{
            "train_steps": 1000000
        },
        "alignment": {
            "base_model": "/home/riadoshi/alignment/Alignment/models/train_multiple/c3b57089-5466-4a2d-a9a7-d82eff45732a" 
        }},
        # small 128 model
        # {"model": {
        #     "n_embd": 128,
        #     "n_layer": 6,
        #     "n_head": 4,
        # },
        # "training":{
        #     "train_steps": 1000000
        # },
        # "alignment": {
        #     "base_model": "/home/riadoshi/alignment/Alignment/models/train_multiple/5d3b26ff-427f-40a2-9e2e-14434bd9b277" 
        # }},

        # {"model": {
        #     "n_embd": 128,
        #     "n_layer": 6,
        #     "n_head": 4,
        # },
        # "training":{
        #     "train_steps": 1000000
        # },
        # "alignment": {
        #     "base_model": "/home/riadoshi/alignment/Alignment/models/train_multiple/5d3b26ff-427f-40a2-9e2e-14434bd9b277" 
        # }},
    ]

    # Create and run training for each variation
    for i, variation in enumerate(variations):
        config_filename = create_config_file(base_config, variation, i)
        run_training(config_filename)
        print(f"Training completed for: {config_filename}")

if __name__ == "__main__":
    main()
