import torch
import torch.nn.functional as F


def noise_schedule(max_timesteps, t, batch_size, seq_len):
    """
    Function to return a mask tensor for a batch where fraction of tokens are masked based on time step t (linear).\
    """
    # linearly increasing mask probability
    mask_prob = t / max_timesteps
    num_ones = int(seq_len * mask_prob)
    num_zeroes = seq_len - num_ones

    # create a mask for each sequence in the batch
    masks = []
    for _ in range(batch_size):
        tensor = torch.cat([torch.ones(num_ones), torch.zeros(num_zeroes)])
        shuffled_indices = torch.randperm(seq_len)
        mask = tensor[shuffled_indices].int()
        masks.append(mask)

    return torch.stack(masks)


def apply_noise(mask, input_seq, t):
    """
    Function to apply diffusion noise (masking) to a batch of input sequences.
    """

    masked_input_seq = input_seq.clone()
    masked_input_seq[mask == 1] = 32  # apply mask token (ID 32)

    return masked_input_seq, mask


def scheduler_loss_fn(logits, target, mask, vocab_size):
    """
    Function to compute loss only on masked positions for a batch.
    """
    # both logits and mask flattened to -1 so that they can be directly compared to see which positions are masked and loss should not be
    # computed on
    # set reduction to NONE to have an individual loss per token.
    loss = F.cross_entropy(
        logits.view(-1, vocab_size), target.view(-1), reduction="none"
    )
    # Only penalize masked positions
    loss = loss * mask.view(-1)
    return loss.mean()
