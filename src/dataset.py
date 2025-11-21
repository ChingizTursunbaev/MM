import torch
import pickle
import numpy as np
from torch.utils.data import Dataset

class PhoenixSpatialDataset(Dataset):
    def __init__(self, pkl_path: str, vocab, is_train: bool = False):
        """
        Args:
            pkl_path: Path to dataset pickle
            vocab: Vocabulary object (from utils.py)
            is_train: If True, applies random token masking (augmentation)
        """
        self.vocab = vocab
        self.is_train = is_train 
        
        print(f"Loading Data: {pkl_path}")
        with open(pkl_path, "rb") as f:
            self.data = pickle.load(f)
        
        # Filter valid entries
        self.keys = [k for k in self.data.keys() if "motion_tokens" in self.data[k]]
        print(f"Found {len(self.keys)} samples. Augmentation Active: {self.is_train}")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        entry = self.data[key]
        
        tokens = torch.from_numpy(entry["motion_tokens"]).long() # (T, 133)
        
        # --- AUGMENTATION 1: Random Frame Dropping (Time Warping) ---
        # Only on training set
        if self.is_train:
            T = tokens.shape[0]
            # We want to keep ~80% of frames, dropping ~20% randomly
            # This prevents the model from memorizing exact durations
            keep_prob = 0.8
            
            # Generate a random mask for TIME steps (T,)
            # We expand it to (T, 1) so it broadcasts over joints
            time_mask = torch.bernoulli(torch.full((T,), keep_prob)).bool()
            
            # Safety: Ensure we don't drop everything. If all False, keep all.
            if time_mask.sum() > 10: 
                tokens = tokens[time_mask] # Selection slices the time dimension
        
        # --- AUGMENTATION 2: Token Masking (Existing) ---
        if self.is_train: 
            prob = torch.rand(tokens.shape)
            mask = prob < 0.15 
            tokens[mask] = 4 # <unk>
        
        gloss_text = entry["gloss"]
        label_ids = torch.tensor(self.vocab.text_to_ids(gloss_text), dtype=torch.long)
        
        return tokens, label_ids

def collate_fn(batch):
    """
    Pads variable length sequences for Mamba and CTC.
    """
    inputs, labels = zip(*batch)
    
    # 1. Pad Inputs with 0 (Pad Token)
    input_lengths = torch.tensor([t.shape[0] for t in inputs], dtype=torch.long)
    inputs_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    
    # 2. Pad Labels with -1 (Ignore Index for CTC)
    target_lengths = torch.tensor([l.shape[0] for l in labels], dtype=torch.long)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-1)
    
    return inputs_padded, input_lengths, labels_padded, target_lengths