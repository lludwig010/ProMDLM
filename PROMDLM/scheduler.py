import torch
import torch.nn.functional as F

def noise_schedule(max_timesteps, t, batch_size, seq_len):
    """Returns a mask tensor for a batch where fraction of tokens are masked based on time step t (linear)."""
    mask_prob = t / max_timesteps  # Linearly increasing mask probability
    num_ones = int(seq_len * mask_prob)
    num_zeroes = seq_len - num_ones

    # Create a mask for each sequence in the batch
    masks = []
    for _ in range(batch_size):
        tensor = torch.cat([torch.ones(num_ones), torch.zeros(num_zeroes)])
        shuffled_indices = torch.randperm(seq_len)
        mask = tensor[shuffled_indices].int()
        masks.append(mask)

    return torch.stack(masks)  # Shape: (batch_size, seq_len)


def apply_noise(mask, input_seq, t):
    """Apply diffusion noise (masking) to a batch of input sequences."""
    
    masked_input_seq = input_seq.clone()
    masked_input_seq[mask == 1] = 20  # Apply mask token (ID 20)

    return masked_input_seq, mask


def scheduler_loss_fn(logits, target, mask, vocab_size):
    """Compute loss only on masked positions for a batch."""
    # both logits and mask flattened to -1 so that they can be directly compared to see which positions are masked and loss should not be 
    # computed on 
    # set reduction to NONE to have an individual loss per token.
    loss = F.cross_entropy(logits.view(-1, vocab_size), target.view(-1), reduction='none')
    loss = loss * mask.view(-1)  # Only penalize masked positions
    return loss.mean()
