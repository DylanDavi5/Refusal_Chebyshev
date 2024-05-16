import os
import subprocess
import yaml
import uuid

def create_config_file(base_config, variation, index, subdirectory='conf/train_multiple'):
    
    # Ensure the subdirectory exists
    os.makedirs(subdirectory, exist_ok=True)
    
    # Construct the file path within the subdirectory
    config_filename = os.path.join(subdirectory, f'config_{index}.yaml')
    
    # Update the configuration and dump it into the file
    config = base_config.copy()
    config['training'].update(variation["training"])
    config['model'].update(variation["model"])
    #config["out_dir"] = config["out_dir"] + "/model_" + str(index)
    with open(config_filename, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
    

    return config_filename
    

def run_training(config_filename):
    # Adjust the command according to your specific needs
    command = f'python train.py --config {config_filename}'
    subprocess.run(command, shell=True)

def main():
    run_id = str(uuid.uuid4())

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
            "task": "kernel_linear_regression",
            "data": "uniform",
            "task_kwargs": {"basis_dim": 11, "different_degrees": True, "lowest_degree": 1},
            "batch_size": 32,
            "learning_rate": 0.00005,
            "save_every_steps": 50000,
            "keep_every_steps": 50000,
            "train_steps": 5000001,
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
                    "interval": 1000
                },
                "deg":{ 
                    "start": 11,
                    "end": 11,
                    "inc": 0,
                    "interval": 500001
                }
            }
        },

        "out_dir": "../models/train_multiple", #f"../models/train_multiple/{run_id}"

        "wandb":{
            "name": "base_256pts",
            "project": "alignment",
            "entity": "rdoshi21",
            "notes":"", 
            "log_every_steps": 100
        }
    }


    # Variations you want to test
    variations = [
        #model 1
        {"model": {
            "n_embd": 512,
            "n_layer": 24,
            "n_head": 16,
        },
        "training":{
            "train_steps": 500000
        }},
        #model 1
        # {"model": {
        #     "n_embd": 256,
        #     "n_layer": 12,
        #     "n_head": 8,
        # },
        # "training":{
        #     "train_steps": 500000
        # }},
        #model 2
        # {"model": {
        #     "n_embd": 64,
        #     "n_layer": 4,
        #     "n_head": 2,
        # },
        # "training":{
        #     "train_steps": 500000
        # }},
    ]


    
    location = f"conf/train_multiple/{run_id}"


    # Create and run training for each variation
    for i, variation in enumerate(variations):
        config_filename = create_config_file(base_config, variation, i, location)

        #print(config_filename)

        run_training(config_filename)
        print(f"Training completed for: {config_filename}")

if __name__ == "__main__":
    main()
