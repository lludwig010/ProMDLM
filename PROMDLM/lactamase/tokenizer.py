class AminoAcidTokenizer:
    def __init__(self):
        # X IS MASKED
        self.amino_acids = "ACDEFGHIKLMNPQRSTVWXY"  # Standard 20 amino acids
        self.special_tokens = ["<UNK>"]  # Special tokens

        # Assign indices to special tokens first
        self.token_to_idx = {tok: i for i, tok in enumerate(self.special_tokens)}

        # Assign indices to amino acids after special tokens
        self.token_to_idx.update({aa: i + len(self.special_tokens) for i, aa in enumerate(self.amino_acids)})

        # Reverse mapping for decoding
        self.idx_to_token = {i: tok for tok, i in self.token_to_idx.items()}

    def tokenize(self, sequence, add_special_tokens=True):
        """Convert an amino acid sequence into token indices."""
        tokens = [self.token_to_idx.get(aa, self.token_to_idx["<UNK>"]) for aa in sequence]
        return tokens

    def detokenize(self, tokens):
        """Convert a list of token indices back into an amino acid sequence."""
        sequence = [self.idx_to_token[token] for token in tokens if token in self.idx_to_token]
        return "".join(sequence)

    def get_vocab_size(self):
        """Get the size of the vocabulary."""
        return len(self.token_to_idx)