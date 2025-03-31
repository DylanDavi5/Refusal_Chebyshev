import torch

class ActivationRecorder:
    def __init__(self):
        self.clear()

    def clear(self):
        self.activations = {}

    def store(self, layer_idx, output):
        if layer_idx not in self.activations:
            self.activations[layer_idx] = []
        self.activations[layer_idx].append(output.detach().cpu())

    def get_mean_activations(self):
        return {
            layer: torch.stack(tensors).mean(dim=0)
            for layer, tensors in self.activations.items()
        }

def register_residual_hooks(model, recorder):
    handles = []
    for i, block in enumerate(model._backbone.h):
        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                recorder.store(layer_idx, output[0])
            return hook_fn
        handles.append(block.register_forward_hook(make_hook(i)))
    return handles

def clear_hooks(handles):
    for h in handles:
        h.remove()


# Before final prediction, inject:
def inject_vector(activation, vector):
    return activation + vector[None, None, :]  # broadcast over batch and seq

# In forward pass, use a hook to inject:
def make_injection_hook(layer, vector):
    def hook_fn(module, input, output):
        output[0][:] = inject_vector(output[0], vector)
    return hook_fn


def ablate_vector(activation, vector):
    unit_vector = vector / vector.norm()
    proj = (activation @ unit_vector)[:, :, None] * unit_vector[None, None, :]
    return activation - proj
#ablate at vector at a specific layer (the ablation is only applied at the given layer)
def make_ablation_hook(layer, vector):
    def hook_fn(module, input, output):
        output[0][:] = ablate_vector(output[0], vector)
    return hook_fn

#ablate a given vector across the entire model
def make_global_ablation_hooks(model, r_vector):
    """Apply directional ablation at all transformer layers."""
    handles = []

    r_hat = r_vector / r_vector.norm()  # Normalize once

    def make_hook(r_hat):
        def hook_fn(module, input, output):
            x = output[0]
            r_hat_device = r_hat.to(x.device) 

            proj = (x @ r_hat_device)[:, :, None] * r_hat_device[None, None, :]
            output[0][:] = x - proj
        return hook_fn

    for i in range(len(model._backbone.h)):
        h = model._backbone.h[i].register_forward_hook(make_hook(r_hat))
        handles.append(h)

    return handles
