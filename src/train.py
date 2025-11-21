import os
import torch
import argparse
import yaml
import time
from torch.utils.data import DataLoader
from dataset import PhoenixSpatialDataset, collate_fn
from model import SpatialQMamba
from utils import Vocabulary, setup_logger, compute_wer, greedy_decode
from tqdm import tqdm

def train_one_epoch(model, loader, criterion, optimizer, device, logger, log_interval):
    model.train()
    total_loss = 0
    
    for i, (inputs, input_lens, targets, target_lens) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        log_probs, comp_lens = model(inputs, input_lens)
        
        # CTC expects (T, B, C)
        loss_input = log_probs.transpose(0, 1)
        
        loss = criterion(loss_input, targets, comp_lens, target_lens)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        if i % log_interval == 0 and i > 0:
            logger.info(f"Batch {i}/{len(loader)} | Loss: {loss.item():.4f}")
            
    return total_loss / len(loader)

def validate(model, loader, criterion, device, vocab, logger):
    model.eval()
    total_loss = 0
    all_refs = []
    all_hyps = []
    
    with torch.no_grad():
        for inputs, input_lens, targets, target_lens in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            log_probs, comp_lens = model(inputs, input_lens)
            
            # Loss
            loss_input = log_probs.transpose(0, 1)
            loss = criterion(loss_input, targets, comp_lens, target_lens)
            total_loss += loss.item()
            
            # Decoding for WER (Greedy)
            # log_probs is (B, T, V) here because we didn't transpose yet
            batch_hyps = greedy_decode(log_probs, comp_lens, vocab)
            
            # Decode References
            for j in range(targets.shape[0]):
                # Remove padding (-1)
                true_seq = [x for x in targets[j].tolist() if x != -1]
                all_refs.append(vocab.ids_to_text(true_seq))
            
            all_hyps.extend(batch_hyps)

    # Compute WER
    wer_score = compute_wer(all_refs, all_hyps) * 100 # Percentage
    
    # Log examples
    logger.info("\n--- Validation Prediction Examples ---")
    for k in range(2):
        logger.info(f"Ref:  {all_refs[k]}")
        logger.info(f"Pred: {all_hyps[k]}")
        logger.info("-" * 20)
        
    return total_loss / len(loader), wer_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    # Load Config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Setup Logging
    logger = setup_logger(cfg['paths']['output_dir'])
    logger.info("Starting Spatial-Q Mamba Experiment (Dropout + Augmentation + Best WER Tracking)")
    logger.info(f"Config loaded: {cfg}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # 1. Vocab & Data
    vocab = Vocabulary(cfg['paths']['gloss_dict'])
    
    
    train_ds = PhoenixSpatialDataset(cfg['paths']['train_data'], vocab, is_train=True)
    dev_ds   = PhoenixSpatialDataset(cfg['paths']['dev_data'], vocab, is_train=False)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=cfg['training']['batch_size'], 
        collate_fn=collate_fn, 
        shuffle=True, 
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    dev_loader = DataLoader(
        dev_ds, 
        batch_size=cfg['training']['batch_size'], 
        collate_fn=collate_fn, 
        shuffle=False, 
        num_workers=8,
        pin_memory=True
    )

    # 2. Model
    model = SpatialQMamba(
        vocab_size=vocab.size,
        num_joints=cfg['model']['num_joints'],
        token_embed_dim=cfg['model']['token_embed_dim'],
        model_dim=cfg['model']['model_dim'],
        compress_factor=cfg['model']['compress_factor'],
        num_layers=cfg['model']['num_layers'],
        dropout=cfg['model'].get('dropout', 0.25) # Safe default if config misses it
    ).to(device)

    # 3. Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg['training']['learning_rate'], 
        weight_decay=0.05 
    )
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    # 4. Training Loop
    best_wer = float("inf")
    
    for epoch in range(cfg['training']['epochs']):
        start_time = time.time()
        logger.info(f"\nEpoch {epoch+1}/{cfg['training']['epochs']}")
        
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, logger, cfg['training']['print_every']
        )
        
        # Validate
        val_loss, val_wer = validate(model, dev_loader, criterion, device, vocab, logger)
        
        duration = time.time() - start_time
        
        # --- Best WER Logic ---
        is_new_best = False
        if val_wer < best_wer:
            best_wer = val_wer
            is_new_best = True
            
            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "wer": best_wer
            }
            torch.save(checkpoint, os.path.join(cfg['paths']['output_dir'], "best.pt"))
            logger.info(f">>> New Best Model Saved!")

        # Save Last
        checkpoint_last = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "wer": val_wer
        }
        torch.save(checkpoint_last, os.path.join(cfg['paths']['output_dir'], "last.pt"))

        # --- LOGGING with Best WER ---
        best_indicator = "<<< NEW BEST" if is_new_best else ""
        
        logger.info(
            f"Summary Ep {epoch+1}: "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"WER: {val_wer:.2f}% | "
            f"Best WER: {best_wer:.2f}% | "
            f"Time: {duration:.0f}s {best_indicator}"
        )

if __name__ == "__main__":
    main()