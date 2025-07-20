import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
from PIL import Image
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class TrainingSample:
    """Represents a single training sample with image and label."""
    image_path: str
    label: str
    confidence: float
    farmer_id: str
    location: str
    upload_date: datetime
    metadata: Dict

class FarmerDataPipeline:
    """
    Pipeline for processing farmer-uploaded data for LoRA fine-tuning.
    Handles image validation, labeling, and dataset preparation.
    """
    
    def __init__(self, base_dir: str = "farmer_data"):
        self.base_dir = Path(base_dir)
        self.raw_data_dir = self.base_dir / "raw"
        self.processed_data_dir = self.base_dir / "processed"
        self.validation_data_dir = self.base_dir / "validation"
        
        # Create necessary directories
        for dir_path in [self.raw_data_dir, self.processed_data_dir, self.validation_data_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def validate_image(self, image_path: str) -> bool:
        """
        Validates uploaded image for quality and format.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            bool: True if image is valid for training
        """
        try:
            with Image.open(image_path) as img:
                # Check image format
                if img.format not in ['JPEG', 'PNG', 'JPG']:
                    logger.warning(f"Unsupported image format: {img.format}")
                    return False
                
                # Check image size (minimum 224x224 for MobileNetV3)
                if img.size[0] < 224 or img.size[1] < 224:
                    logger.warning(f"Image too small: {img.size}")
                    return False
                
                # Check if image is not corrupted
                img.verify()
                return True
                
        except Exception as e:
            logger.error(f"Image validation failed for {image_path}: {e}")
            return False
    
    def process_farmer_upload(self, 
                            images: List[str], 
                            labels: List[str], 
                            farmer_id: str,
                            location: str,
                            confidence_scores: Optional[List[float]] = None) -> List[TrainingSample]:
        """
        Processes farmer-uploaded images and labels.
        
        Args:
            images: List of image file paths
            labels: List of corresponding labels
            farmer_id: Unique identifier for the farmer
            location: Geographic location of the farm
            confidence_scores: Optional confidence scores for predictions
            
        Returns:
            List of validated TrainingSample objects
        """
        training_samples = []
        
        for i, (image_path, label) in enumerate(zip(images, labels)):
            if not self.validate_image(image_path):
                logger.warning(f"Skipping invalid image: {image_path}")
                continue
            
            # Use default confidence if not provided
            confidence = confidence_scores[i] if confidence_scores else 0.8
            
            # Create metadata
            metadata = {
                "farmer_id": farmer_id,
                "location": location,
                "upload_date": datetime.now().isoformat(),
                "original_filename": Path(image_path).name,
                "image_size": Image.open(image_path).size
            }
            
            sample = TrainingSample(
                image_path=image_path,
                label=label,
                confidence=confidence,
                farmer_id=farmer_id,
                location=location,
                upload_date=datetime.now(),
                metadata=metadata
            )
            
            training_samples.append(sample)
        
        return training_samples
    
    def save_training_samples(self, samples: List[TrainingSample], dataset_name: str) -> str:
        """
        Saves training samples to organized dataset structure.
        
        Args:
            samples: List of training samples
            dataset_name: Name for the dataset
            
        Returns:
            str: Path to the saved dataset
        """
        dataset_dir = self.processed_data_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for each class
        class_dirs = {}
        for sample in samples:
            if sample.label not in class_dirs:
                class_dir = dataset_dir / sample.label
                class_dir.mkdir(exist_ok=True)
                class_dirs[sample.label] = class_dir
        
        # Copy images and create metadata
        dataset_metadata = {
            "dataset_name": dataset_name,
            "creation_date": datetime.now().isoformat(),
            "total_samples": len(samples),
            "classes": list(class_dirs.keys()),
            "samples": []
        }
        
        for i, sample in enumerate(samples):
            # Copy image to appropriate class directory
            new_filename = f"{sample.farmer_id}_{i}_{Path(sample.image_path).name}"
            new_path = class_dirs[sample.label] / new_filename
            shutil.copy2(sample.image_path, new_path)
            
            # Add to metadata
            sample_metadata = {
                "id": i,
                "image_path": str(new_path.relative_to(dataset_dir)),
                "label": sample.label,
                "confidence": sample.confidence,
                "farmer_id": sample.farmer_id,
                "location": sample.location,
                "upload_date": sample.upload_date.isoformat(),
                "metadata": sample.metadata
            }
            dataset_metadata["samples"].append(sample_metadata)
        
        # Save dataset metadata
        metadata_path = dataset_dir / "dataset_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(dataset_metadata, f, indent=2)
        
        logger.info(f"Dataset saved to {dataset_dir} with {len(samples)} samples")
        return str(dataset_dir)
    
    def create_validation_split(self, dataset_path: str, validation_ratio: float = 0.2) -> Tuple[str, str]:
        """
        Creates training/validation split for the dataset.
        
        Args:
            dataset_path: Path to the dataset
            validation_ratio: Ratio of data to use for validation
            
        Returns:
            Tuple of (training_path, validation_path)
        """
        dataset_dir = Path(dataset_path)
        training_dir = dataset_dir / "train"
        validation_dir = dataset_dir / "val"
        
        training_dir.mkdir(exist_ok=True)
        validation_dir.mkdir(exist_ok=True)
        
        # Load metadata
        with open(dataset_dir / "dataset_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Split samples by class
        for class_name in metadata["classes"]:
            class_dir = dataset_dir / class_name
            if not class_dir.exists():
                continue
            
            # Get all images in class
            images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpeg"))
            np.random.shuffle(images)
            
            # Split into train/val
            split_idx = int(len(images) * (1 - validation_ratio))
            train_images = images[:split_idx]
            val_images = images[split_idx:]
            
            # Create class directories
            train_class_dir = training_dir / class_name
            val_class_dir = validation_dir / class_name
            train_class_dir.mkdir(exist_ok=True)
            val_class_dir.mkdir(exist_ok=True)
            
            # Copy images
            for img in train_images:
                shutil.copy2(img, train_class_dir / img.name)
            for img in val_images:
                shutil.copy2(img, val_class_dir / img.name)
        
        logger.info(f"Created train/val split: {len(list(training_dir.rglob('*.jpg')))} train, {len(list(validation_dir.rglob('*.jpg')))} val")
        return str(training_dir), str(validation_dir)
    
    def get_dataset_statistics(self, dataset_path: str) -> Dict:
        """
        Generates statistics for the dataset.
        
        Args:
            dataset_path: Path to the dataset
            
        Returns:
            Dictionary containing dataset statistics
        """
        dataset_dir = Path(dataset_path)
        
        if not (dataset_dir / "dataset_metadata.json").exists():
            return {"error": "Dataset metadata not found"}
        
        with open(dataset_dir / "dataset_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Count samples per class
        class_counts = {}
        for sample in metadata["samples"]:
            label = sample["label"]
            class_counts[label] = class_counts.get(label, 0) + 1
        
        # Calculate average confidence
        avg_confidence = np.mean([s["confidence"] for s in metadata["samples"]])
        
        # Count farmers
        unique_farmers = len(set(s["farmer_id"] for s in metadata["samples"]))
        
        return {
            "total_samples": metadata["total_samples"],
            "classes": metadata["classes"],
            "class_distribution": class_counts,
            "average_confidence": avg_confidence,
            "unique_farmers": unique_farmers,
            "creation_date": metadata["creation_date"]
        } 