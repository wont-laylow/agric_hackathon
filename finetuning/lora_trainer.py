import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import timm
from peft import LoraConfig, get_peft_model, TaskType
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import json
from datetime import datetime
import wandb
from tqdm import tqdm

logger = logging.getLogger(__name__)

class FarmerDataset(Dataset):
    """Custom dataset for farmer-uploaded data."""
    
    def __init__(self, data_dir: str, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform or self._get_default_transform()
        
        # Load dataset metadata
        metadata_path = self.data_dir.parent / "dataset_metadata.json"
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Create label mapping
        self.classes = sorted(list(set(sample["label"] for sample in self.metadata["samples"])))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Filter samples for this split (train/val)
        split_name = self.data_dir.name
        self.samples = [
            sample for sample in self.metadata["samples"]
            if (self.data_dir / sample["image_path"]).exists()
        ]
    
    def _get_default_transform(self):
        """Default transform for training."""
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = self.data_dir / sample["image_path"]
        
        # Load image
        image = datasets.folder.default_loader(str(image_path))
        
        if self.transform:
            image = self.transform(image)
        
        label = self.class_to_idx[sample["label"]]
        return image, label

class LoRATrainer:
    """
    LoRA fine-tuning trainer for personalized disease detection.
    """
    
    def __init__(self, 
                 base_model_name: str = "mobilenetv3_large_100",
                 num_classes: int = 22,
                 device: str = "cpu"):
        self.base_model_name = base_model_name
        self.num_classes = num_classes
        self.device = torch.device(device)
        
        # Initialize base model
        self.base_model = self._create_base_model()
        
        # LoRA configuration
        self.lora_config = LoraConfig(
            task_type=TaskType.IMAGE_CLASSIFICATION,
            inference_mode=False,
            r=16,  # Rank
            lora_alpha=32,  # Alpha parameter
            lora_dropout=0.1,
            target_modules=["classifier.4", "classifier.7"]  # Target specific layers
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.base_model, self.lora_config)
        self.model.to(self.device)
        
        logger.info(f"LoRA model initialized with {sum(p.numel() for p in self.model.parameters() if p.requires_grad)} trainable parameters")
    
    def _create_base_model(self) -> nn.Module:
        """Creates the base MobileNetV3 model."""
        model = timm.create_model(self.base_model_name, pretrained=True, num_classes=self.num_classes)
        
        # Custom classifier for better fine-tuning
        num_features = model.num_features
        model.reset_classifier(0)
        model.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.BatchNorm1d(num_features),
            nn.Dropout(0.2),
            nn.Linear(num_features, num_features // 4),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features // 4),
            nn.Dropout(0.1),
            nn.Linear(num_features // 4, self.num_classes)
        )
        
        return model
    
    def prepare_data(self, train_dir: str, val_dir: str, batch_size: int = 16) -> Tuple[DataLoader, DataLoader]:
        """Prepares data loaders for training."""
        
        # Create datasets
        train_dataset = FarmerDataset(train_dir)
        val_dataset = FarmerDataset(val_dir, transform=self._get_validation_transform())
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=2,
            pin_memory=True
        )
        
        logger.info(f"Data loaders created: {len(train_dataset)} train samples, {len(val_dataset)} val samples")
        return train_loader, val_loader
    
    def _get_validation_transform(self):
        """Transform for validation data."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def train(self, 
              train_loader: DataLoader, 
              val_loader: DataLoader,
              num_epochs: int = 10,
              learning_rate: float = 1e-4,
              weight_decay: float = 1e-4,
              save_dir: str = "lora_adapters",
              use_wandb: bool = False) -> Dict:
        """
        Trains the LoRA adapter on farmer data.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            save_dir: Directory to save LoRA adapters
            use_wandb: Whether to use Weights & Biases logging
            
        Returns:
            Dictionary containing training results
        """
        
        # Initialize wandb if requested
        if use_wandb:
            wandb.init(project="farmer-lora-finetuning", name=f"lora-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        
        # Setup optimizer and loss
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        # Training loop
        best_val_acc = 0.0
        training_history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }
        
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
            for batch_idx, (images, labels) in enumerate(train_pbar):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*train_correct/train_total:.2f}%'
                })
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
                for images, labels in val_pbar:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
                    
                    val_pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{100.*val_correct/val_total:.2f}%'
                    })
            
            # Calculate metrics
            train_loss_avg = train_loss / len(train_loader)
            train_acc = 100. * train_correct / train_total
            val_loss_avg = val_loss / len(val_loader)
            val_acc = 100. * val_correct / val_total
            
            # Update history
            training_history["train_loss"].append(train_loss_avg)
            training_history["train_acc"].append(train_acc)
            training_history["val_loss"].append(val_loss_avg)
            training_history["val_acc"].append(val_acc)
            
            # Log to wandb
            if use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss_avg,
                    "train_acc": train_acc,
                    "val_loss": val_loss_avg,
                    "val_acc": val_acc,
                    "learning_rate": scheduler.get_last_lr()[0]
                })
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_lora_adapter(save_path / "best_lora_adapter")
                logger.info(f"New best model saved with validation accuracy: {val_acc:.2f}%")
            
            scheduler.step()
            
            logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                       f"Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}%, "
                       f"Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save final model and training history
        self.save_lora_adapter(save_path / "final_lora_adapter")
        
        history_path = save_path / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        if use_wandb:
            wandb.finish()
        
        return {
            "best_val_acc": best_val_acc,
            "training_history": training_history,
            "save_path": str(save_path)
        }
    
    def save_lora_adapter(self, save_path: Path):
        """Saves the LoRA adapter weights."""
        self.model.save_pretrained(save_path)
        logger.info(f"LoRA adapter saved to {save_path}")
    
    def load_lora_adapter(self, adapter_path: str):
        """Loads a LoRA adapter."""
        self.model = self.model.from_pretrained(adapter_path)
        self.model.to(self.device)
        logger.info(f"LoRA adapter loaded from {adapter_path}")
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """Evaluates the model on test data."""
        self.model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        all_predictions = []
        all_labels = []
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Evaluating"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        test_acc = 100. * test_correct / test_total
        test_loss_avg = test_loss / len(test_loader)
        
        return {
            "test_loss": test_loss_avg,
            "test_acc": test_acc,
            "predictions": all_predictions,
            "labels": all_labels
        } 