# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such
init_from = 'scratch_loop' # 'scratch' or 'resume' or 'gpt2*' or 'scratch_loop'
out_dir = 'out-phop-16-looped'
model_output_name = 'ckpt_1_6_12.pt' # ckpt_<base_block_size>_<loop_start>_<num_loops>.pt
eval_interval = 1000 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 100 # don't print too too often

# system
device = 'cuda'
compile = False # do not torch compile the model
# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'phop-16'
wandb_run_name = 'phop-16-4M-looped_1_6_12' # <base_block_size>_<loop_start>_<num_loops>.pt

dataset = 'phop-16'
gradient_accumulation_steps = 1
batch_size = 256
block_size = 277 # context of up to 278 previous characters

# baby GPT model :)
n_layer = 1
n_head = 6
n_embd = 768
dropout = 0.2

# Loop config
num_loops = 12
loop_start = 6
loop_func = 'z=f(x+z)'

# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 60000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
warmup_iters = 100 # not super necessary potentially

# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = max_iters # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
