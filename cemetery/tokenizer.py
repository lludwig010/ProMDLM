import re

class AminoAcidTokenizer:
    def __init__(self):
        # Define special tokens explicitly
        self.special_tokens = ["<PAD>", "<MASK>", "<BOS>", "<EOS>", "<UNK>"]

        # Standard 20 amino acids (no X/Z)
        self.amino_acids = "ACDEFGHIKLMNPQRSTVWY"

        # Assign indices to special tokens first
        self.token_to_idx = {tok: i for i, tok in enumerate(self.special_tokens)}

        # Assign indices to amino acids after special tokens
        self.token_to_idx.update({
            aa: i + len(self.special_tokens) for i, aa in enumerate(self.amino_acids)
        })

        # Reverse mapping
        self.idx_to_token = {i: tok for tok, i in self.token_to_idx.items()}

    def tokenize(self, sequence, add_special_tokens=True):
        # Match full special tokens or single characters
        pattern = r"<[^>]+>|."  # Matches "<BOS>", "<MASK>", etc. OR any single character
        raw_tokens = re.findall(pattern, sequence)

        tokens = [
            self.token_to_idx.get(tok, self.token_to_idx["<UNK>"])
            for tok in raw_tokens
        ]

        if add_special_tokens and "<BOS>" not in raw_tokens:
            tokens = [self.token_to_idx["<BOS>"]] + tokens
        if add_special_tokens and "<EOS>" not in raw_tokens:
            tokens = tokens + [self.token_to_idx["<EOS>"]]

        return tokens

    def detokenize(self, tokens):
        sequence = [self.idx_to_token[token] for token in tokens if token in self.idx_to_token]
        return "".join(sequence)

    def get_vocab_size(self):
        return len(self.token_to_idx)

    def pad_token_id(self):
        return self.token_to_idx["<PAD>"]

    def mask_token_id(self):
        return self.token_to_idx["<MASK>"]

    def bos_token_id(self):
        return self.token_to_idx["<BOS>"]

    def eos_token_id(self):
        return self.token_to_idx["<EOS>"]