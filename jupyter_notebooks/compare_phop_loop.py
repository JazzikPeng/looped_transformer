import os
import json
import numpy as np
from quinine import QuinineArgumentParser
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
import torch.nn.functional as F

import sys
sys.path.append('/home/jupyter/project/looped_transformer/scripts')
from nano_gpt import GPT2Model, GPT2Config

from utils import eval_unlooped_model, aggregate_metrics, eval_looped_model

fig_hparam = {
    'figsize': (8, 5),
    'labelsize': 28,
    'ticksize': 20,
    'linewidth': 5,
    'fontsize': 15,
    'titlesize': 20,
    'markersize': 15
}

# font specification
fontdict = {'family': 'serif',
         'size': fig_hparam['fontsize'],
         }


device = torch.device('cuda:0')
def get_model(model, result_dir, run_id, step, best=False):
    if best:
        model_path = os.path.join(result_dir, run_id, 'model_best.pt')
        state_dict = torch.load(model_path, map_location='cpu')['state_dict']
        best_err = torch.load(model_path, map_location='cpu')['loss']
        print("saved model with loss:", best_err)
    if step == -1:
        model_path = os.path.join(result_dir, run_id, 'state.pt')
        state_dict = torch.load(model_path, map_location='cpu')['model_state_dict']
    else:
        model_path = os.path.join(result_dir, run_id, 'model_{}.pt'.format(step))
        state_dict = torch.load(model_path, map_location='cpu')['model']
    
#     return state_dict
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=True)
    
    return model


result_dir = '/home/jupyter/project/looped_transformer/results2/phop_baseline_loop/'
run_id = '0614230044-LR_loop_L1_ends{12}_T{5}-a013'

from models import TransformerModelLoopedWithEmbLayer

n_dims = 20
n_positions = 101
n_embd = 256
n_layer = 1
n_head = 8
model = TransformerModelLoopedWithEmbLayer(n_dims, n_positions, n_embd, n_layer, n_head)
step = 10000
model = get_model(model, result_dir, run_id, step)
model = model.to(device)
model.eval()

# Construct phop eval data 
import numpy as np
p = 16
max_loop = 1
test_data_path = '/home/jupyter/project/looped_transformer/data/p_hop_sequences_test.txt'

def eval_looped_transformer_loss(
    model,
    test_data_path,
    p=16,
    device='cuda',
    batch_size=512,
    max_loop_values=[1, 2, 3, 4, 5, 6, 7, 8, 12, 15, 30]
):
    """
    Evaluate looped transformer loss for different max_loop values.

    Args:
        model: The looped transformer model.
        test_data_path: Path to test data file.
        p: p-hop parameter.
        device: Device for evaluation.
        batch_size: Batch size for evaluation.
        max_samples: Maximum number of samples to evaluate (None for all).
        max_loop_values: Iterable of max_loop values to evaluate.

    Returns:
        dict: {max_loop: loss}
    """
    # Load and prepare data
    test_data = np.loadtxt(test_data_path, dtype=np.int32, delimiter=' ')
    test_tensor = torch.from_numpy(test_data).long()
    test_dataset = TensorDataset(test_tensor)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    loss_fn = nn.CrossEntropyLoss(reduction='sum')
    model.eval()
    results = {}

    for max_loop in max_loop_values:
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for (batch_data,) in tqdm(test_loader, desc=f"max_loop={max_loop}"):
                batch_data = batch_data.to(device, non_blocking=True)
                batch_data = batch_data.unsqueeze(-1)
                batch_data.to(device)
                xs = batch_data[:, :-p-1, :].long().cuda()
                ys = batch_data[:, -p-1:, :].long().cuda()
                output = model(xs, torch.zeros_like(ys), 0, max_loop)
                print(len(output), output[0].shape, output[-1].shape)
                y_pred = output[-1][:, -p-1:, :]
                y_pred_flat = y_pred.reshape(-1, y_pred.size(-1))
                ys_flat = ys.flatten()
                batch_loss = loss_fn(y_pred_flat, ys_flat)
                total_loss += batch_loss.item()
                total_samples += ys_flat.numel()
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        results[max_loop] = avg_loss
        print(f"max_loop={max_loop}: Cross Entropy Loss = {avg_loss:.4f}")

    return results


# Plot results as a line chart
def plot_results_line(results, title='Looped Transformer Loss Evaluation'):
    max_loops = list(results.keys())
    losses = list(results.values())

    plt.figure(figsize=fig_hparam['figsize'])
    plt.plot(max_loops, losses, marker='o', color='skyblue')
    plt.xlabel('Max Loop', fontsize=fig_hparam['fontsize'])
    plt.ylabel('Cross Entropy Loss', fontsize=fig_hparam['fontsize'])
    plt.title(title, fontsize=fig_hparam['titlesize'])
    plt.xticks(max_loops, fontsize=fig_hparam['ticksize'])
    plt.yticks(fontsize=fig_hparam['ticksize'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
max_loop_values = [1, 2, 3, 4, 5, 6, 7, 8, 12, 15, 30]
results = eval_looped_transformer_loss(model, test_data_path=test_data_path, max_loop_values=max_loop_values)
plot_results_line(results, title='Looped Transformer Loss Evaluation for p-hop sequences')