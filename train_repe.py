"""
Training script for RepE on CLUTRR dataset.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer

from repe_clutrr import (
    RepEConfig, 
    RepEModel, 
    create_dataloaders, 
    set_seed
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train RepE on CLUTRR")
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='gpt2',
                        help='Pretrained model name')
    parser.add_argument('--layer_idx', type=int, default=-1,
                        help='Layer index to extract representations from')
    parser.add_argument('--num_directions', type=int, default=10,
                        help='Number of concept directions to learn')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data/clutrr',
                        help='Directory containing CLUTRR dataset')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name (default: timestamp)')
    
    # Other
    parser.add_argument('--eval_every', type=int, default=100,
                        help='Evaluate every N steps')
    parser.add_argument('--save_every', type=int, default=500,
                        help='Save checkpoint every N steps')
    
    return parser.parse_args()


def train_step(
    model: RepEModel,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    device: str
) -> float:
    """
    Perform a single training step.
    
    Returns:
        loss: Training loss value
    """
    model.train()
    
    # Move batch to device
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    
    # Forward pass
    modified_reps, alignments = model(input_ids, attention_mask)
    
    # Compute loss
    # For now, use a contrastive loss on representations
    # In practice, you'd want task-specific supervision
    
    # Normalize representations
    norm_reps = torch.nn.functional.normalize(modified_reps, p=2, dim=1)
    
    # Compute pairwise similarities
    similarities = torch.matmul(norm_reps, norm_reps.t())
    
    # Contrastive loss: encourage diversity
    batch_size = similarities.size(0)
    mask = torch.eye(batch_size, device=device).bool()
    off_diagonal = similarities.masked_select(~mask)
    
    # Loss: minimize off-diagonal similarities (encourage diversity)
    diversity_loss = off_diagonal.pow(2).mean()
    
    # Regularization: encourage orthogonal concept directions
    concept_similarity = torch.matmul(
        torch.nn.functional.normalize(model.concept_directions, p=2, dim=1),
        torch.nn.functional.normalize(model.concept_directions, p=2, dim=1).t()
    )
    reg_loss = concept_similarity.masked_select(~torch.eye(
        model.config.num_directions, device=device).bool()
    ).pow(2).mean()
    
    # Total loss
    loss = diversity_loss + model.config.regularization_weight * reg_loss
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return loss.item()


@torch.no_grad()
def evaluate(
    model: RepEModel,
    dataloader: torch.utils.data.DataLoader,
    device: str
) -> dict:
    """
    Evaluate model on a dataset.
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Forward pass
        modified_reps, alignments = model(input_ids, attention_mask)
        
        # Compute loss (same as training)
        norm_reps = torch.nn.functional.normalize(modified_reps, p=2, dim=1)
        similarities = torch.matmul(norm_reps, norm_reps.t())
        batch_size = similarities.size(0)
        mask = torch.eye(batch_size, device=device).bool()
        off_diagonal = similarities.masked_select(~mask)
        loss = off_diagonal.pow(2).mean()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    return {
        'loss': avg_loss,
    }


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    if args.exp_name is None:
        args.exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_dir = Path(args.output_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"Output directory: {output_dir}")
    
    # Initialize tensorboard
    writer = SummaryWriter(log_dir=output_dir / 'logs')
    
    # Create config
    config = RepEConfig(
        model_name=args.model_name,
        layer_idx=args.layer_idx,
        num_directions=args.num_directions,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        max_length=args.max_length,
        seed=args.seed,
    )
    
    print("\n" + "="*50)
    print("Configuration:")
    print("="*50)
    for key, value in vars(config).items():
        print(f"  {key:25s}: {value}")
    print("="*50 + "\n")
    
    # Load tokenizer
    print(f"Loading tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataloaders
    print(f"Loading data from: {args.data_dir}")
    dataloaders = create_dataloaders(
        config=config,
        tokenizer=tokenizer,
        data_dir=Path(args.data_dir)
    )
    
    # Initialize model
    print(f"\nInitializing RepE model...")
    model = RepEModel(config).to(config.device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.01
    )
    
    # Training loop
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50 + "\n")
    
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        print("-" * 50)
        
        # Training
        train_loader = dataloaders['train']
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            loss = train_step(model, batch, optimizer, config.device)
            
            # Log
            writer.add_scalar('train/loss', loss, global_step)
            progress_bar.set_postfix({'loss': f'{loss:.4f}'})
            
            global_step += 1
            
            # Evaluate
            if global_step % args.eval_every == 0:
                print("\n")
                val_metrics = evaluate(model, dataloaders['val'], config.device)
                
                print(f"Step {global_step} - Val Loss: {val_metrics['loss']:.4f}")
                
                writer.add_scalar('val/loss', val_metrics['loss'], global_step)
                
                # Save best model
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    torch.save(
                        model.state_dict(),
                        output_dir / 'best_model.pt'
                    )
                    print(f"✓ Saved best model (loss: {best_val_loss:.4f})")
            
            # Save checkpoint
            if global_step % args.save_every == 0:
                checkpoint_path = output_dir / f'checkpoint_step_{global_step}.pt'
                torch.save({
                    'step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config,
                }, checkpoint_path)
                print(f"\n✓ Saved checkpoint: {checkpoint_path}")
    
    # Final evaluation on test set
    print("\n" + "="*50)
    print("Final evaluation on test set...")
    print("="*50 + "\n")
    
    test_metrics = evaluate(model, dataloaders['test'], config.device)
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    
    # Save final model
    torch.save(model.state_dict(), output_dir / 'final_model.pt')
    print(f"\n✓ Training complete! Results saved to: {output_dir}")
    
    writer.close()


if __name__ == "__main__":
    main()
