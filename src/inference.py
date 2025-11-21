import torch
import argparse
import yaml
from torch.utils.data import DataLoader
from dataset import PhoenixSpatialDataset, collate_fn
from model import SpatialQMamba
from utils import Vocabulary, compute_wer, greedy_decode

# Import pyctcdecode locally to apply the fix without changing utils.py
try:
    from pyctcdecode import build_ctcdecoder
    BEAM_AVAILABLE = True
except ImportError:
    BEAM_AVAILABLE = False

def get_fixed_beam_decoder(vocab):
    """
    Builds a decoder with the 'Space Hack' applied locally.
    We append a space to every word so the decoder separates them.
    """
    if not BEAM_AVAILABLE:
        print("Error: pyctcdecode not installed.")
        return None
    
    # --- THE FIX: MANUALLY BUILD LABELS WITH SPACES ---
    labels = []
    for i in range(vocab.size):
        word = vocab.idx_to_gloss.get(i, "")
        if i == 0:
            # Index 0 is Blank -> Must be empty string
            labels.append("") 
        else:
            # All other words -> Append space " "
            # This tells the decoder: "When you finish this word, add a space."
            # Result: "WETTER" + "REGEN" -> "WETTER REGEN "
            labels.append(word + " ")
            
    decoder = build_ctcdecoder(
        labels,
        kenlm_model_path=None,
        alpha=0.5,
        beta=1.0
    )
    return decoder

def run_beam_decode(decoder, log_probs, input_lengths):
    """
    Local beam decode loop. Uses CPU loop to avoid Multiprocessing errors.
    """
    # Convert to Numpy on CPU
    probs = torch.exp(log_probs).cpu().numpy()
    batch_size = probs.shape[0]
    
    predictions = []
    
    for i in range(batch_size):
        length = input_lengths[i].item()
        # Slice the valid time steps
        sequence_logits = probs[i, :length, :]
        
        # Decode
        decoded_text = decoder.decode(sequence_logits, beam_width=20)
        
        # Strip trailing whitespace resulting from our "Space Hack"
        predictions.append(decoded_text.strip())

    return predictions

def evaluate(model, loader, device, vocab, decoder=None):
    model.eval()
    all_refs = []
    all_hyps = []
    
    print(">>> Starting Inference Loop...")
    printed_sample = False

    with torch.no_grad():
        for i, (inputs, input_lens, targets, target_lens) in enumerate(loader):
            inputs = inputs.to(device)
            
            # Forward pass
            log_probs, comp_lens = model(inputs, input_lens)
            
            # --- DECODING ---
            if decoder is not None:
                # Use local beam function
                batch_hyps = run_beam_decode(decoder, log_probs, comp_lens)
            else:
                # Use greedy from utils
                batch_hyps = greedy_decode(log_probs, comp_lens, vocab)
            
            # Get References (Truth)
            for j in range(targets.shape[0]):
                true_seq = [x for x in targets[j].tolist() if x != -1]
                all_refs.append(vocab.ids_to_text(true_seq))
            
            all_hyps.extend(batch_hyps)
            
            # Print a few examples from the first batch
            if not printed_sample:
                print("\n--- Sample Predictions ---")
                for k in range(min(2, len(batch_hyps))):
                    print(f"Ref:  {all_refs[k]}")
                    print(f"Pred: {all_hyps[k]}")
                    print("-" * 30)
                printed_sample = True

    # Compute Final WER
    wer = compute_wer(all_refs, all_hyps) * 100
    return wer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--subset", type=str, default="dev", choices=["dev", "test"])
    parser.add_argument("--beam", action="store_true", help="Enable Beam Search")
    args = parser.parse_args()

    # 1. Load Config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Inference on: {device}")

    # 2. Prepare Data
    vocab = Vocabulary(cfg['paths']['gloss_dict'])
    
    data_path = cfg['paths']['dev_data'] if args.subset == "dev" else cfg['paths']['test_data']
    print(f"Loading Data: {data_path}")
    
    ds = PhoenixSpatialDataset(data_path, vocab, is_train=False)
    loader = DataLoader(ds, batch_size=cfg['training']['batch_size'], 
                        collate_fn=collate_fn, shuffle=False, num_workers=4)

    # 3. Build Model
    model = SpatialQMamba(
        vocab_size=vocab.size,
        num_joints=cfg['model']['num_joints'],
        token_embed_dim=cfg['model']['token_embed_dim'],
        model_dim=cfg['model']['model_dim'],
        compress_factor=cfg['model']['compress_factor'],
        num_layers=cfg['model']['num_layers'],
        dropout=cfg['model'].get('dropout', 0.5)
    ).to(device)

    # 4. Load Checkpoint
    print(f"Loading weights from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"Model Loaded (Saved at Epoch {checkpoint['epoch']})")

    # 5. Initialize Decoder (Locally fixed)
    decoder = None
    if args.beam:
        print("Initializing Fixed Beam Search Decoder...")
        decoder = get_fixed_beam_decoder(vocab)

    # 6. Run Evaluation
    wer = evaluate(model, loader, device, vocab, decoder)
    
    print("\n" + "="*30)
    print(f"Final {args.subset.upper()} WER: {wer:.2f}%")
    print("="*30)

if __name__ == "__main__":
    main()