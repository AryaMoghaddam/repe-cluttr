"""
RepE (Representation Engineering) implementation for CLUTRR dataset.

This module implements representation engineering techniques to improve
relational reasoning in language models on the CLUTRR task.
"""

import torch
import torch.nn as nn
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModel
import numpy as np


@dataclass
class RepEConfig:
    """Configuration for RepE experiments."""
    
    model_name: str = "gpt2"
    layer_idx: int = -1  # Which layer to extract representations from
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_epochs: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    
    # RepE specific
    representation_dim: int = 768  # GPT-2 hidden size
    num_directions: int = 10  # Number of concept directions to learn
    regularization_weight: float = 0.01


class CLUTRRDataset(torch.utils.data.Dataset):
    """Dataset class for CLUTRR."""
    
    def __init__(self, data_path: Path, tokenizer, max_length: int = 512):
        """
        Args:
            data_path: Path to CLUTRR JSON file
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        print(f"Loaded {len(self.data)} examples from {data_path}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example."""
        item = self.data[idx]
        
        # Construct input text from story and question
        story = item.get('story', '')
        query = item.get('query', '')
        target = item.get('target', '')
        
        # Format: "Story: {story} Question: {query}"
        text = f"Story: {story}\nQuestion: {query}\nAnswer:"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'target': target,
            'story': story,
            'query': query,
        }


class RepEModel(nn.Module):
    """
    RepE model that learns task-relevant directions in representation space.
    """
    
    def __init__(self, config: RepEConfig):
        super().__init__()
        self.config = config
        
        # Load pretrained model
        self.model = AutoModel.from_pretrained(config.model_name)
        self.model.eval()  # Freeze base model
        
        # Learnable concept directions
        self.concept_directions = nn.Parameter(
            torch.randn(config.num_directions, config.representation_dim)
        )
        
        # Projection head for task-specific prediction
        self.projection = nn.Linear(config.representation_dim, config.representation_dim)
        self.classifier = nn.Linear(config.representation_dim, 1)  # Placeholder
        
    def extract_representations(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract representations from specified layer.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Representations [batch_size, hidden_dim]
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            # Get hidden states from specified layer
            hidden_states = outputs.hidden_states[self.config.layer_idx]
            
            # Pool: take mean of non-padding tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_hidden / sum_mask
            
        return pooled
    
    def compute_concept_alignment(
        self, 
        representations: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute alignment of representations with learned concept directions.
        
        Args:
            representations: [batch_size, hidden_dim]
            
        Returns:
            Alignments: [batch_size, num_directions]
        """
        # Normalize representations and directions
        norm_reps = torch.nn.functional.normalize(representations, p=2, dim=1)
        norm_dirs = torch.nn.functional.normalize(self.concept_directions, p=2, dim=1)
        
        # Compute dot products
        alignments = torch.matmul(norm_reps, norm_dirs.t())
        
        return alignments
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            - Modified representations [batch_size, hidden_dim]
            - Concept alignments [batch_size, num_directions]
        """
        # Extract base representations
        representations = self.extract_representations(input_ids, attention_mask)
        
        # Compute concept alignments
        alignments = self.compute_concept_alignment(representations)
        
        # Apply learned transformation
        modified_reps = self.projection(representations)
        
        # Add concept-specific modifications
        concept_contributions = torch.matmul(
            alignments, 
            self.concept_directions
        )
        modified_reps = modified_reps + concept_contributions
        
        return modified_reps, alignments


def create_dataloaders(
    config: RepEConfig,
    tokenizer,
    data_dir: Path = Path("data/clutrr")
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create dataloaders for train, val, and test splits.
    
    Args:
        config: RepE configuration
        tokenizer: HuggingFace tokenizer
        data_dir: Directory containing dataset files
        
    Returns:
        Dictionary of dataloaders
    """
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        dataset = CLUTRRDataset(
            data_path=data_dir / f"{split}.json",
            tokenizer=tokenizer,
            max_length=config.max_length
        )
        
        dataloaders[split] = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=(split == 'train'),
            num_workers=0,  # Set to 0 for compatibility
            pin_memory=True if config.device == 'cuda' else False
        )
    
    return dataloaders


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    # Quick test
    config = RepEConfig()
    print("RepE Configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  Device: {config.device}")
    print(f"  Layer: {config.layer_idx}")
    print(f"  Num directions: {config.num_directions}")
