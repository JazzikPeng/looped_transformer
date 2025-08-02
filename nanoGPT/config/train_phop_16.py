# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*' or 'scratch_loop'
out_dir = 'out-phop-16'
model_output_name = 'ckpt_6_6.pt' # ckpt_<base_block_size>_<loop_start>_<num_loops>.pt
eval_interval = 1000 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 100 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True # override via command line if you like
wandb_project = 'phop-16'
wandb_run_name = 'phop-16-4M-6_6_768'

dataset = 'phop-16'
gradient_accumulation_steps = 1
batch_size = 256
block_size = 277 # context of up to 278 previous characters

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 768
dropout = 0.0

# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
warmup_iters = 100 # not super necessary potentially

# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# on macbook also add
device = 'cuda:1'  # run on cpu only
compile = True # do not torch compile the model
