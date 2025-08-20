"""
This file is used to generate phop tasks. 
Organize pHop task into a next token generation task.
"""

# Read from /home/jupyter/project/nanoGPT/data
# /home/jupyter/project/nanoGPT/data
import os
import time
import math
import pickle
import torch
import numpy as np

# file_path = '/home/jupyter/project/looped_transformer/nanoGPT/data/phop/p_hop_sequences_32_512_8.txt'
# train_npy = '/home/jupyter/project/looped_transformer/nanoGPT/data/phop/p_hop_sequences_32_512_8_train.npy'
# test_npy = '/home/jupyter/project/looped_transformer/nanoGPT/data/phop/p_hop_sequences_32_512_8_test.npy'

file_path = '/home/jupyter/project/looped_transformer/nanoGPT/data/phop/p_hop_sequences_16_256_4.txt'
train_npy = '/home/jupyter/project/looped_transformer/nanoGPT/data/phop/p_hop_sequences_16_256_4_train.npy'
test_npy = '/home/jupyter/project/looped_transformer/nanoGPT/data/phop/p_hop_sequences_16_256_4_test.npy'

# file_path = '/home/jupyter/project/looped_transformer/nanoGPT/data/phop/p_hop_sequences_64_1024_16.txt'
# train_npy = '/home/jupyter/project/looped_transformer/nanoGPT/data/phop/p_hop_sequences_64_1024_16_train.npy'
# test_npy = '/home/jupyter/project/looped_transformer/nanoGPT/data/phop/p_hop_sequences_64_1024_16_test.npy'

data = np.loadtxt(file_path, dtype=np.int32, delimiter=' ')
# Find p from the file sequence
# Read file
seq = [int(x) for x in data[0]]
    
p_idx = seq.index(0)
start_idx = seq.index(1)
end_idx = seq.index(2)
output_start_idx = seq.index(3)

p = seq[p_idx + 1]
input_seq = seq[start_idx + 1: end_idx]
BLOCK_SIZE = len(input_seq) + p + 1 + 4 # 4 is the number of special tokens
SEQ_LENGTH = BLOCK_SIZE
SPECIAL_MASK_TOKEN = -1 # Speical Mask Token Don't go into loss calculation
TRAIN_SIZE = 4000000
TEST_SIZE = 1000

print(f"p: {p}, BLOCK_SIZE: {BLOCK_SIZE}, SEQ_LENGTH: {SEQ_LENGTH}")
    
def generate_phop_training_data(data, p=16):
    """
    Generate pHop training data from the input data.
    The input data is a sequence of integers.
    The output is a list of sequences where each sequence is of length p.
    """
    sequences = []
    for i in range(0, p+1):
        print(i)
        x = data[0:i+BLOCK_SIZE]  # p + 1 to include the next token
        y = data[0:i+BLOCK_SIZE+1]
        # Pad x, y with leading 0s to length 277
        x = np.pad(x, (SEQ_LENGTH - len(x), 0), mode='constant')
        y = np.pad(y, (SEQ_LENGTH - len(y), 0), mode='constant')
        sequences.append((x, y))
    return np.array(sequences)

def generate_phop_training_data_vectorized(data, p=16):
    """
    Vectorized version that processes all batches in parallel
    """
    batch_size = data.shape[0]
    
    # Calculate all sequence lengths
    x_lengths = BLOCK_SIZE
    y_lengths = BLOCK_SIZE 
    
    # Create output arrays
    # This vectorized operation handles all batch dimensions at once
    x_padded = np.zeros((batch_size, x_lengths), dtype=data.dtype)
    y_padded = np.zeros((batch_size, y_lengths), dtype=data.dtype)
    
    y_padded[:, :] = SPECIAL_MASK_TOKEN  # Fill y with 100 token to mask out losses in Y
    
    x_padded[:, :] = data[:, :-1] 
    y_padded[:, -p-1:] = data[:, -p-1:]
    
    # Stack to create final array of shape (batch_size, 2, SEQ_LENGTH)
    sequences = np.stack([x_padded, y_padded], axis=1)
    return sequences

import functools
@functools.lru_cache(maxsize=2)
def load_phop_data(split, data_path):
    if split == 'train':
        path = data_path[0]
        total_samples = TRAIN_SIZE
    else:
        path = data_path[1]
        total_samples = TEST_SIZE

    data = np.load(path)
    data = data.reshape(total_samples, 2, SEQ_LENGTH)  # assumes (N, 2, seq_len)
    return data

def get_phop_batch(split, data_path, batch_size=12, device=None):
    data = load_phop_data(split, data_path)
    # Random select batch_size samples
    ix = torch.randint(0, data.shape[0], (batch_size,))
    x = data[ix, 0, :]  # Select x sequences
    y = data[ix, 1, :]  # Select y sequences
    # Mask out question in Y and 
    
    
    # x = x.reshape(batch_size, -1)  # Reshape to (batch_size, SEQ_LENGTH)
    # y = y.reshape(batch_size, -1)  # Reshape to (batch_size, SEQ_LENGTH)
    x, y = torch.from_numpy(x), torch.from_numpy(y)
    if device == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y.long()  # Convert y to long tensor for loss calculation

import functools
# @functools.lru_cache(maxsize=2)
def get_phop_batch_thinking_token(split, batch_size=64, device=None):
    if split == 'train':
        data = np.load('/home/jupyter/project/nanoGPT/data/phop/p_hop_sequences_dev_train_thinking.npy')
    else:
        data = np.load('/home/jupyter/project/nanoGPT/data/phop/p_hop_sequences_dev_test_thinking.npy')

    total_samples, p_hops, seq_len = 0, 16, 278

    if split == 'train':
        total_samples = 20480
    elif split == 'val':
        total_samples = 5120
    
    # -1 is Y, else is X
    

    # Random select batch_size samples
    ix = torch.randint(0, data.shape[0], (batch_size,))
    x = data[ix, :-1]  # Select x sequences
    y = data[ix, -1:]  # Select y sequences
    # x = x.reshape(batch_size, -1)  # Reshape to (batch_size, SEQ_LENGTH)
    # y = y.reshape(batch_size, -1)  # Reshape to (batch_size, SEQ_LENGTH)
    x, y = torch.from_numpy(x), torch.from_numpy(y)
    if device == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

def generate_phop_training_data_thinking_token_and_final_answer(data, p=16):
    """
    Generate training data for the pHop task with thinking tokens and final answers.
    """
    thinking_token_mask_mul = np.ones(SEQ_LENGTH, dtype=np.int32)
    thinking_token_mask_add = np.ones(SEQ_LENGTH, dtype=np.int32)
    # mask -17 to -1
    thinking_token_mask_mul[-(p+1):-1] = 0
    thinking_token_mask_add[-(p+1):-1] = THINKING_TOKEN_MASK

    # Apply mask to data
    # Replace the last p tokens with thinking tokens
    data_with_mask = data * thinking_token_mask_mul + thinking_token_mask_add

    return data_with_mask


if __name__== "__main__":
    # Read p_hop_sequences_dev.txt and prepare them in to array and write to bin file
    # Split Train Test data
    if True:
        train, test = data[:TRAIN_SIZE], data[TRAIN_SIZE:TRAIN_SIZE+TEST_SIZE]
        print("Train shape:", train.shape, "Test shape:", test.shape)
        train = generate_phop_training_data_vectorized(train, p)
        train = train.reshape(len(train), 2, SEQ_LENGTH) 
        # Save to np array
        np.save(train_npy, train)
        
        # output_file = '/home/jupyter/project/nanoGPT/data/phop/p_hop_sequences_dev_train.bin'
        # with open(output_file, 'wb') as f:
        #     pickle.dump(train, f)
        # print(f"Data saved to {output_file}", "Shape:", train.shape)
        
        test = generate_phop_training_data_vectorized(test, p)
        test = test.reshape(len(test), 2, SEQ_LENGTH) 
        np.save(test_npy, test)

        # output_file = '/home/jupyter/project/nanoGPT/data/phop/p_hop_sequences_dev_test.bin'
        # with open(output_file, 'wb') as f:
        #     pickle.dump(test, f)
        # print(f"Data saved to {output_file}", "Shape:", test.shape)
        
    # Test the get_phop_batch function
    if True:
        x, y = get_phop_batch('train', (train_npy, test_npy), batch_size=1, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        print("Batch x:", x, "Batch y:", y)
        
        
    # Generate training data with thinking tokens and final answers, it is not next token prediction 
    # It predicts the final answer given the thinking tokens directly.
    if False:
        train, test = np.split(data, [int(0.8 * len(data))])
        print("Train shape:", train.shape, "Test shape:", test.shape)
        train = generate_phop_training_data_thinking_token_and_final_answer(train, p=16)
        test = generate_phop_training_data_thinking_token_and_final_answer(test, p=16)
        # Save to np array
        np.save('/home/jupyter/project/nanoGPT/data/phop/p_hop_sequences_dev_train_thinking.npy', train)
        np.save('/home/jupyter/project/nanoGPT/data/phop/p_hop_sequences_dev_test_thinking.npy', test)
        print("Train shape:", train.shape, "Test shape:", test.shape)