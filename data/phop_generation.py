# Generate p-hop sequence training data for model training.
# Example: [a, b, c, a, b, c, a, b, c]  -> c, a, b, c, 0. 3-hop. 
# Write a function to generate such sequences.

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional

# For multi-threading
import concurrent.futures
import os

TOKEN_MAP = {
    "p": 0,  # Start of sequence
    "s": 1,  # Start of hops
    "e": 2,  # End of sequence
    "hop": 3,  # Hop token
}

RESERVED_TOKENS_SIZE = len(TOKEN_MAP)

def hop(seq: List[int], curr_idx: int) -> Optional[List[int]]:
    """
    Perform one hop on the sequence.
    :param seq: Input sequence of integers.
    :param curr_idx: Current index in the sequence.
    :return: penultimate index and value at that index.
        If no penultimate index exists, return None. Hop ends
    """
    # Find the penultimate index with the same seq[curr_idx] value
    penultimate_idx = curr_idx - 1
    while penultimate_idx >= 0 and seq[penultimate_idx] != seq[curr_idx]:
        penultimate_idx -= 1
    if penultimate_idx < 0:
        return -1, None
    # Perform the hop
    hop_value = seq[penultimate_idx + 1]
    return penultimate_idx + 1, hop_value
    
def p_hop(seq: List[int], p: int, curr_idx: int) -> List[int]:
    """
    Perform p hops on the sequence.
    :param seq: Input sequence of integers.
    :param p: Number of hops to perform.
    :param curr_idx: Current index in the sequence.
    :return: List of indices and values at those indices after p hops.
    """
    result = []
    result.append((curr_idx, seq[curr_idx]))  # Append the current index and value
    for _ in range(p):
        penultimate_idx, hop_value = hop(seq, curr_idx)
        if penultimate_idx == -1:
            break
        result.append((penultimate_idx, hop_value))
        curr_idx = penultimate_idx
    return result

# Generate sequence that satisfies the p-hop condition
def generate_k_hop_sequence_by_rule(vocab_size: int, p: int, max_gap: int) -> Tuple[List[int], List[int]]:
    """
    Generate a k-hop induction heads task sequence. 
    This sequence is suppose to be k-hoppable.
    :param seq_len: Total sequence length.
    :param vocab_size: Size of the vocabulary.
    :param p: Number of hops.
    :param pattern_len: Length of the repeating pattern.
    :param max_gap: Maximum gap between the same elements. 
        aps are randomly sampled between 1 and max_gap.
    """
    # Sample 4 max_gap values 
    gaps = np.random.randint(1, max_gap + 1, size=p) # [2,2,2]
    # Randomly generate p elements in vocabulary for p hops
    hops = np.random.randint(0, vocab_size, size=p)
    # generate the p hops sequence 
    seq = []
    seq.append(hops[0])
    init_run = True
    for i in range(p):
        size = gaps[i] if init_run else gaps[i] - 1
        seq.extend(np.random.randint(0, vocab_size, size=size))
        seq.append(hops[i])
        if gaps[i]-1 > 0 and i + 1 < p:
            # Insert
            # seq.insert(-1, hops[i+1])
            seq[-2] = hops[i+1]
        init_run = False
    
    return seq[::-1], gaps, hops.tolist()
    

def generate_one_k_hop_sequence(seq_len:int , vocab_size: int, p: int) -> Optional[Tuple[List[int], List[int]]]:
    """
    Generate a pure random sequence at fixed length, run p_hop 
        algorithm to get each steps and output train_x, train_y pairs. 
        This sequence is suppose to be k-hoppable.
    :param seq_len: Total sequence length.
    :param vocab_size: Size of the vocabulary.
    :param p: Number of hops.
    """
    seq = np.random.randint(0 + RESERVED_TOKENS_SIZE, vocab_size + RESERVED_TOKENS_SIZE, size=seq_len).tolist()
    # Apply p_hop
    res = p_hop(seq, p, len(seq)-1)
    
    # Check if results are valid and return
    if len(res) < p + 1: # p+1 means the first element doesn't count as a hop
        return None

    # Process sequence results for training 
    training_seq = [TOKEN_MAP['p']] + [p] + [TOKEN_MAP['s']] + seq + [TOKEN_MAP['e']] + [TOKEN_MAP['hop']] + [x[1] for x in res]
    assert training_seq[-1] == res[-1][1], "Last hop value should match the last element in the sequence"

    # Convert result to numpy array of int type
    return training_seq, res[-1][1]

def generate_k_hop_sequences(seq_len:int , vocab_size: int, p: int, num_samples: int, file_path: str) -> int:
    """
    Use multi-threading to call `generate_one_k_hop_sequence` to
    generate multiple k-hop sequences and write them to a file.
    :param num_samples: Number of sequences to generate.
    :param file_path: write to the output file.
    :return: Number of sequences generated.
    """

    if not os.path.exists(file_path):
        print(f"file_path, {file_path} does not exist, creating it.")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    num_sequences = 0
    with open(file_path, 'w') as f:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for _ in range(num_samples):
                futures.append(executor.submit(generate_one_k_hop_sequence, seq_len, vocab_size, p))
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    training_seq, last_hop_value = result
                    f.write(' '.join(map(str, training_seq)) + '\n')
                    num_sequences += 1
                    
    return num_sequences
    

def test_p_hop():
    seq = [1, 2, 3, 1, 2, 3, 1, 2, 3]
    p = 4
    curr_idx = 8
    result = p_hop(seq, p, curr_idx)
    assert len(result) == 4
    print(result)
    assert result[0] == (6, 1)  # First hop
    assert result[1] == (4, 2)  # Second hop
    assert result[2] == (2, 3)  # Third hop

# Write unit test for one_hop function
def test_one_hop():
    seq = [1, 2, 3, 1, 2, 3, 1, 2, 3]
    curr_idx = 8
    penultimate_idx, hop_value = hop(seq, curr_idx)
    assert penultimate_idx == 6
    assert hop_value == 1

def generate_mini_k_hop_sequences():
    # Example usage
    seq_len = 256
    vocab_size = 4
    p = 16
    num_samples = 1000
    # Generate k-hop sequences and save to file
    file_path = "../data/p_hop_sequences_mini.txt"
    
    num_sequences = generate_k_hop_sequences(seq_len, vocab_size, p, num_samples, file_path)
    print(f"Generated {num_sequences} k-hop sequences and saved to {file_path}")

def generate_full_k_hop_sequences():
    # Example usage
    seq_len = 256
    vocab_size = 4
    p = 16
    num_samples = 4262000
    # Generate k-hop sequences and save to file
    file_path = "../data/p_hop_sequences.txt"
    
    num_sequences = generate_k_hop_sequences(seq_len, vocab_size, p, num_samples, file_path)
    print(f"Generated {num_sequences} k-hop sequences and saved to {file_path}")

if __name__ == "__main__":
    # # Run the unit test
    # test_one_hop()
    # print("one_hop function passed the test.")
    
    # # Example usage
    # seq = [1, 2, 3, 1, 2, 3, 1, 2, 3]
    # p = 4
    # curr_idx = 8
    # penultimate_idx, hop_value = hop(seq, curr_idx)
    # print(f"Penultimate index: {penultimate_idx}, Hop value: {hop_value}")
    
    # test_p_hop()
    # print(generate_k_hop_sequence(vocab_size=5, p=3, max_gap=4))
    # generate_mini_k_hop_sequences()
    generate_full_k_hop_sequences()