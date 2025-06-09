import os
import json
import numpy as np
from quinine import QuinineArgumentParser
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
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


class LinearRegression():
    def __init__(self, batch_size, n_points, n_dims, n_dims_truncated, device, w_star=None):
        super(LinearRegression, self).__init__()
        self.device = device
        self.xs = torch.randn(batch_size, n_points, n_dims).to(device)
        self.xs[..., n_dims_truncated:] = 0
        w_b = torch.randn(batch_size, n_dims, 1) if w_star is None else w_star.to(device)  # [B, d, 1]
        w_b[:, n_dims_truncated:] = 0
        self.w_b = w_b.to(device)
        self.ys = (self.xs @ self.w_b).sum(-1)  # [B, n]
        
sample_size = 1280
batch_size = 128
n_points = 41
n_dims_truncated = 20
n_dims = 20

real_task = LinearRegression(sample_size, n_points, n_dims, n_dims_truncated, device)
xs, ys, w_b = real_task.xs, real_task.ys, real_task.w_b


result_dir = '/home/jupyter/project/looped_transformer/results2/linear_regression_baseline'
run_id = '0522072019-LR_baseline-1e03'

from models import TransformerModel

n_positions = 101
n_embd = 256
n_layer = 12
n_head = 8

model = TransformerModel(n_dims, n_positions, n_embd, n_layer, n_head)
step = -1
model = get_model(model, result_dir, run_id, step)
model = model.to(device)

err, y_pred_total = eval_unlooped_model(model, xs, ys)

result_errs = {}
result_errs['Transformer'] = err

# Plot err
plt.figure(figsize=(8, 5))
plt.plot(np.arange(1, len(err)+1), err, label='Transformer', color='blue', linewidth=5)
plt.xlabel('Loop times', fontdict=fontdict)
plt.ylabel('Error', fontdict=fontdict)
plt.xticks(fontsize=fig_hparam['ticksize'])
plt.yticks(fontsize=fig_hparam['ticksize'])
plt.legend(fontsize=fig_hparam['fontsize'])
plt.title('Error of Transformer', fontdict=fontdict)

# from models import TransformerModelLooped

# result_dir = '/home/jupyter/project/looped_transformer/results2/linear_regression_loop'
# run_id = '0524072558-LR_loop_L1_ends{12}_T{5}-3b39'

# n_positions = 101
# n_embd = 256
# n_head = 8
# T = 500
# n_layer = 1

# model = TransformerModelLooped(n_dims, n_positions, n_embd, n_layer, n_head)
# step = -1
# model = get_model(model, result_dir, run_id, step)
# model = model.to(device)
    
# err, loop_err = eval_looped_model(model, xs, ys, loop_max=T)

# result_errs['Looped Transformer'] = err