import torch
import pickle
import jiwer
import logging
import os
from typing import List

class Vocabulary:
    def __init__(self, gloss_path):
        print(f"Loading Vocabulary from: {gloss_path}")
        with open(gloss_path, "rb") as f:
            raw_list = pickle.load(f)
        
        # Shift-by-One for CTC Blank (Index 0)
        self.idx_to_gloss = {0: "<BLANK>"}
        self.gloss_to_idx = {"<BLANK>": 0}
        
        for i, gloss in enumerate(raw_list):
            new_idx = i + 1
            self.idx_to_gloss[new_idx] = gloss
            self.gloss_to_idx[gloss] = new_idx
            
        self.size = len(self.idx_to_gloss)

    # --- THIS WAS MISSING ---
    def text_to_ids(self, text: str) -> List[int]:
        # Input: "WETTER REGEN" -> Output: [45, 102]
        words = text.strip().split(" ")
        # Default to index 4 (<unk>) if word not found
        return [self.gloss_to_idx.get(w, self.gloss_to_idx.get('<unk>', 4)) for w in words]

    def ids_to_text(self, ids: List[int]) -> str:
        # Input: [45, 102] -> Output: "WETTER REGEN"
        # Skips Blank (0) automatically
        result = []
        for i in ids:
            if i != 0: # Skip CTC Blank
                result.append(self.idx_to_gloss.get(i, "<?>"))
        return " ".join(result)

def compute_wer(refs: list, hyps: list):
    """
    refs: List of reference strings ["WETTER GUT", ...]
    hyps: List of hypothesis strings ["WETTER SCHLECHT", ...]
    """
    return jiwer.wer(refs, hyps)

def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "train.log")),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

def greedy_decode(log_probs, lengths, vocab):
    """
    Decodes a batch of log_probs using CTC greedy decoding.
    """
    # log_probs: (B, T, Vocab)
    predictions = []
    with torch.no_grad():
        batch_preds = torch.argmax(log_probs, dim=-1) # (B, T)
        
        for i in range(batch_preds.shape[0]):
            length = lengths[i]
            pred_seq = batch_preds[i, :length].tolist()
            
            # Collapse repeats and blanks (Standard CTC)
            collapsed = []
            prev = None
            for p in pred_seq:
                if p != prev and p != 0: # 0 is blank
                    collapsed.append(p)
                prev = p
            
            predictions.append(vocab.ids_to_text(collapsed))
            
    return predictions