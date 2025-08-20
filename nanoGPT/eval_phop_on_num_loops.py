"""
Evaluate Phop Tasks.
For looped models, evaluate on num_loops at inference time.
"""
import os
import pickle
from contextlib import nullcontext
import torch
from model import GPTConfig, GPT, GPTLooped

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out-phop-16' # ignored if init_from is not 'resume'
model_output_name = 'ckpt_2_6_768.pt'
batch_size = 256
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 1 # number of samples to draw
max_new_tokens = 80 # number of tokens generated in each sample
temperature = 1.0 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 1 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda:1' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = True # use PyTorch 2.0 to compile the model to be faster
# Loop config
num_loops = 6
loop_start = 0
loop_func = 'z=f(x+z)'
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

num_loops_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
correct_last_tokens_accs = [] 
# model
for num_loops in num_loops_list:
    print("Evaluate on num_loops:", num_loops)
    if init_from == 'resume':
        # init from a model saved in a specific directory
        ckpt_path = os.path.join(out_dir, model_output_name)
        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        print("checkpoint model args:", checkpoint['model_args'])
        print("looped model:", 'looped' in out_dir)
        model = GPTLooped(gptconf, num_loops=num_loops, loop_start=loop_start, loop_func=loop_func) if 'looped' in out_dir else GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        
    model.eval()
    model.to(device)
    if compile:
        model = torch.compile(model) # requires PyTorch 2.0 (optional)
        

    # Read example input sequence
    with open('data/phop/p_hop_sequences_test.txt', 'r') as f:
        p_hop_sequences = f.readlines()    

    # Convert p_hop_sequences to tensor for model input
    data = torch.tensor([list(map(int, seq.split())) for seq in p_hop_sequences], device=device)
    # Split into x and y
    # Find index of 3 
    start_idx = torch.where(data[0]==3)[0].item()
    x = data[:, :start_idx+1]  # all but last token
    y = data[:, start_idx+1:start_idx+1+17]    # last token as target

    # run generation
    pred_batches = []

    with torch.no_grad():
        with ctx:
            for start in range(0, x.size(0), batch_size):
                end = min(start + batch_size, x.size(0))
                x_batch = x[start:end]  # take a batch of inputs
                
                pred = model.generate(
                    x_batch,
                    max_new_tokens,
                    temperature=temperature,
                    top_k=top_k
                )
                pred_batches.append(pred)  
                # print("Running batch", x_batch.size(0), "/", x.size(0))

    pred = torch.cat(pred_batches, dim=0)
    print(pred.shape)


    # Evaluate methods 3: Both last and intermediate results needs to be correct
    # Check if all tokens in the last 17 are correct
    all_results = pred[:, x.size(1):][:, :y.size(1)]
    correct_last_tokens = (all_results == y).all(dim=1)
    correct_last_tokens_acc = correct_last_tokens.sum().float() / correct_last_tokens.size(0)
    print(f'Accuracy on both last tokens and intermediate tokens: {correct_last_tokens_acc.item() * 100:.2f}%')
    correct_last_tokens_accs.append(correct_last_tokens_acc.item())
    print('---------------')
# Plot the results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(num_loops_list, correct_last_tokens_accs, marker='o')
plt.title('Accuracy on Both Last Tokens and Intermediate Tokens vs. Number of Loops')
plt.xlabel('Number of Loops')
plt.ylabel('Accuracy (%)')
plt.xticks(num_loops_list)
plt.grid()
plt.savefig('./eval_plots/accuracy_on_both_vs_num_loops(6).png')
plt.show()