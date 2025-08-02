# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such
init_from = 'scratch_loop' # 'scratch' or 'resume' or 'gpt2*' or 'scratch_loop'
out_dir = 'out-phop-16-looped'
model_output_name = 'ckpt_1_2_6.pt' # ckpt_<base_block_size>_<loop_start>_<num_loops>.pt
eval_interval = 1000 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 100 # don't print too too often

# system
device = 'cuda:1'
# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'phop-16'
wandb_run_name = 'phop-16-4M-looped_1_2_6'

dataset = 'phop-16'
gradient_accumulation_steps = 1
batch_size = 256
block_size = 277 # context of up to 278 previous characters

# baby GPT model :)
n_layer = 1
n_head = 6
n_embd = 384
dropout = 0.2

# Loop config
num_loops = 6
loop_start = 2
loop_func = 'z=f(x+z)'

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 20000
lr_decay_iters = max_iters # make equal to max_iters usually
min_lr = 1e-6 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
