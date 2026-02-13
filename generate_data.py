import torch
import numpy as np

def f_k(k, x):
    if k%2 == 0:
        return np.cos(x*(k+2)/2)
    else:
        return np.sin(x*(k+1)/2)
    
def quantize_y(a):
    i = (a + 1)*1000/2
    i = round(i)
    return max(0, min(999, i))

def generate_dataset(n_samples):
    x_indices_1 = np.random.choice(1000, n_samples, replace=False)
    x_indices_2 = np.random.choice(1000, n_samples, replace=False)
    
    x_values_1 = [23 + element * 2 * np.pi * 0.001 for element in x_indices_1]
    x_values_2 = [23 + element * 2 * np.pi * 0.001 for element in x_indices_2]
    
    x_values = x_values_1 + x_values_2
    x_indices = list(x_indices_1) + list(x_indices_2)

    tokens = torch.zeros((n_samples*2, 2), dtype=torch.long)
    targets = torch.zeros((n_samples*2,), dtype=torch.long)

    for i in range(n_samples*2):
        k = i // n_samples  
        y = f_k(k, x_values[i])

        tokens[i, 0] = x_indices[i] + 2  
        tokens[i, 1] = k 
        targets[i] = quantize_y(y)

    return tokens, targets

def generate_test():
    
    x_values_1 = [23 + element * 2 * np.pi * 0.001 for element in range(0, 1000)]
    x_values_2 = [23 + element * 2 * np.pi * 0.001 for element in range(0, 1000)]
    
    x_values = x_values_1 + x_values_2
    x_indices = [x for x in range(0, 1000)] + [x for x in range(0, 1000)]

    tokens_t = torch.zeros((1000*2, 2), dtype=torch.long)
    targets_t = torch.zeros((1000*2,), dtype=torch.long)

    for i in range(1000*2):
        k = i // 1000 
        y = f_k(k, x_values[i])

        tokens_t[i, 0] = x_indices[i] + 2  
        tokens_t[i, 1] = k  
        targets_t[i] = quantize_y(y)

    return tokens_t, targets_t

tokens, targets = generate_dataset(300)
tokens_t, targets_t = generate_test()
torch.save((tokens, targets, tokens_t, targets_t), "dataset.pt")
print(f"Tokens shape: {tokens.shape}")
print(f"Targets shape: {targets.shape}")