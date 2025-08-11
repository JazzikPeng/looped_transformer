"""
This script is used to analyze the distribution of phop sequences.
1. Find the distribution of num steps between each hop and plot the distribution
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from phop_generation import hop

print(os.getcwd())



def analyze_phop_sequence_dist(file_path: str):
    """
    Analyze the distribution of phop sequences.
    """
    all_indices = []
    with open(file_path, "r") as f:
        for line in f:
            seq = [int(x) for x in line.strip().split()]

            # Find index of 0, 1, 2, 3
            p_idx = seq.index(0)
            start_idx = seq.index(1)
            end_idx = seq.index(2)
            output_start_idx = seq.index(3)

            p = seq[p_idx + 1]
            input_seq = seq[start_idx + 1: end_idx]
            output_seq = seq[output_start_idx + 1:]
            
            # find phops each element in output seq
            idx = len(input_seq) - 1
            indices = [idx]
            for i in range(p):
                idx, val = hop(input_seq, idx)
                indices.append(idx)
            
            check = []
            for i in indices:
                check.append(input_seq[i])

            assert check == output_seq, f"Output sequence does not match: {check} != {output_seq}"
            all_indices.append(indices)
    
    print("p_idx:", p_idx, "start_idx:", start_idx, "end_idx:", end_idx, "output_start_idx:", output_start_idx)
    print("Vocab size:", len(set(input_seq)))

        
    # Compute the gap between each hops in all_indices and plot distribution
    all_gaps = []
    for indices in all_indices:
        gaps = np.diff(indices[::-1]) # Make it ascending order
        all_gaps.extend(gaps)

    plt.hist(all_gaps, bins=30)
    plt.xlabel("Gap between hops")
    plt.ylabel("Frequency")
    plt.title("Distribution of Gaps between Hops")
    plt.show()

    print("Mean gap:", np.mean(all_gaps), "Total sequences analyzed:", len(all_indices))

    max_gaps = []
    for indices in all_indices:
        gaps = np.diff(indices[::-1])
        max_gaps.append(gaps.max())

    plt.hist(max_gaps, bins=30)
    plt.xlabel("Max Gap between hops")
    plt.ylabel("Frequency")
    plt.title("Distribution of Max Gaps between Hops")
    plt.show()
    
    print("Mean max gap:", np.mean(max_gaps))


if __name__ == "__main__":
    file_path = "./p_hop_sequences_16_256_4.txt"
    analyze_phop_sequence_dist(file_path)