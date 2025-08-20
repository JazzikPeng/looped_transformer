# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*' or 'scratch_loop'
out_dir = 'out-phop-32'
model_output_name = 'ckpt_2_6_768.pt' # ckpt_<n_layer>_<n_head>_<nn_embd>.pt
eval_interval = 1000 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 100 # don't print too too often

# system
device = 'cuda'
compile = True # do not torch compile the model
# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True # override via command line if you like
wandb_project = 'phop-32'
wandb_run_name = 'phop-32-4M-2_6_768'

dataset = 'phop'
gradient_accumulation_steps = 1
batch_size = 256
block_size = 768 # context of up to 278 previous characters

# baby GPT model :)
n_layer = 2
n_head = 6
n_embd = 768
dropout = 0.0

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
