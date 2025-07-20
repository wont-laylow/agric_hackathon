from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict
import os
import tempfile
import shutil
from pathlib import Path
import json
from datetime import datetime
import logging

from .data_pipeline import FarmerDataPipeline
from .lora_trainer import LoRATrainer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/finetuning", tags=["Fine-tuning"])

# Global instances
data_pipeline = FarmerDataPipeline()
trainer = None  # Will be initialized when needed

@router.post("/upload-farmer-data")
async def upload_farmer_data(
    background_tasks: BackgroundTasks,
    images: List[UploadFile] = File(...),
    labels: List[str] = Form(...),
    farmer_id: str = Form(...),
    location: str = Form(...),
    confidence_scores: Optional[List[float]] = Form(None)
):
    """
    Upload farmer data for fine-tuning.
    
    Args:
        images: List of image files
        labels: List of corresponding labels
        farmer_id: Unique farmer identifier
        location: Geographic location
        confidence_scores: Optional confidence scores
    """
    try:
        # Validate input
        if len(images) != len(labels):
            raise HTTPException(status_code=400, detail="Number of images must match number of labels")
        
        if not farmer_id or not location:
            raise HTTPException(status_code=400, detail="Farmer ID and location are required")
        
        # Save uploaded images temporarily
        temp_dir = Path(tempfile.mkdtemp())
        image_paths = []
        
        for i, image_file in enumerate(images):
            if not image_file.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail=f"File {image_file.filename} is not an image")
            
            # Save image
            image_path = temp_dir / f"{farmer_id}_{i}_{image_file.filename}"
            with open(image_path, "wb") as buffer:
                shutil.copyfileobj(image_file.file, buffer)
            image_paths.append(str(image_path))
        
        # Process farmer data
        training_samples = data_pipeline.process_farmer_upload(
            images=image_paths,
            labels=labels,
            farmer_id=farmer_id,
            location=location,
            confidence_scores=confidence_scores
        )
        
        if not training_samples:
            raise HTTPException(status_code=400, detail="No valid training samples found")
        
        # Save dataset
        dataset_name = f"farmer_{farmer_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        dataset_path = data_pipeline.save_training_samples(training_samples, dataset_name)
        
        # Get dataset statistics
        stats = data_pipeline.get_dataset_statistics(dataset_path)
        
        # Cleanup temp files
        background_tasks.add_task(shutil.rmtree, temp_dir)
        
        return JSONResponse({
            "message": "Farmer data uploaded successfully",
            "dataset_path": dataset_path,
            "dataset_name": dataset_name,
            "samples_processed": len(training_samples),
            "statistics": stats
        })
        
    except Exception as e:
        logger.error(f"Error uploading farmer data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/start-finetuning")
async def start_finetuning(
    background_tasks: BackgroundTasks,
    dataset_path: str = Form(...),
    num_epochs: int = Form(10),
    learning_rate: float = Form(1e-4),
    batch_size: int = Form(16),
    use_wandb: bool = Form(False)
):
    """
    Start LoRA fine-tuning process.
    
    Args:
        dataset_path: Path to the processed dataset
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Batch size for training
        use_wandb: Whether to use Weights & Biases logging
    """
    try:
        # Validate dataset path
        if not Path(dataset_path).exists():
            raise HTTPException(status_code=400, detail="Dataset path does not exist")
        
        # Start training in background
        background_tasks.add_task(
            run_finetuning,
            dataset_path=dataset_path,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            use_wandb=use_wandb
        )
        
        return JSONResponse({
            "message": "Fine-tuning started successfully",
            "job_id": f"finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "dataset_path": dataset_path,
            "parameters": {
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "use_wandb": use_wandb
            }
        })
        
    except Exception as e:
        logger.error(f"Error starting fine-tuning: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/training-status/{job_id}")
async def get_training_status(job_id: str):
    """Get the status of a fine-tuning job."""
    try:
        # This would typically check a database or file system for job status
        # For now, return a mock status
        status_file = Path(f"training_jobs/{job_id}_status.json")
        
        if status_file.exists():
            with open(status_file, 'r') as f:
                status = json.load(f)
            return JSONResponse(status)
        else:
            return JSONResponse({
                "job_id": job_id,
                "status": "not_found",
                "message": "Training job not found"
            })
            
    except Exception as e:
        logger.error(f"Error getting training status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/available-adapters")
async def get_available_adapters():
    """Get list of available LoRA adapters."""
    try:
        adapters_dir = Path("lora_adapters")
        if not adapters_dir.exists():
            return JSONResponse({"adapters": []})
        
        adapters = []
        for adapter_dir in adapters_dir.iterdir():
            if adapter_dir.is_dir():
                # Check if it's a valid LoRA adapter
                config_file = adapter_dir / "adapter_config.json"
                if config_file.exists():
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    
                    adapters.append({
                        "name": adapter_dir.name,
                        "path": str(adapter_dir),
                        "config": config,
                        "created": datetime.fromtimestamp(adapter_dir.stat().st_mtime).isoformat()
                    })
        
        return JSONResponse({"adapters": adapters})
        
    except Exception as e:
        logger.error(f"Error getting available adapters: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/apply-adapter")
async def apply_adapter(adapter_name: str = Form(...)):
    """
    Apply a LoRA adapter to the current model.
    
    Args:
        adapter_name: Name of the adapter to apply
    """
    try:
        global trainer
        
        adapter_path = Path("lora_adapters") / adapter_name
        if not adapter_path.exists():
            raise HTTPException(status_code=400, detail="Adapter not found")
        
        # Initialize trainer if not already done
        if trainer is None:
            trainer = LoRATrainer()
        
        # Load the adapter
        trainer.load_lora_adapter(str(adapter_path))
        
        return JSONResponse({
            "message": f"Adapter {adapter_name} applied successfully",
            "adapter_path": str(adapter_path)
        })
        
    except Exception as e:
        logger.error(f"Error applying adapter: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dataset-statistics/{dataset_name}")
async def get_dataset_statistics(dataset_name: str):
    """Get statistics for a specific dataset."""
    try:
        dataset_path = Path("farmer_data/processed") / dataset_name
        if not dataset_path.exists():
            raise HTTPException(status_code=400, detail="Dataset not found")
        
        stats = data_pipeline.get_dataset_statistics(str(dataset_path))
        return JSONResponse(stats)
        
    except Exception as e:
        logger.error(f"Error getting dataset statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def run_finetuning(dataset_path: str, num_epochs: int, learning_rate: float, batch_size: int, use_wandb: bool):
    """
    Background task to run fine-tuning.
    """
    try:
        logger.info(f"Starting fine-tuning for dataset: {dataset_path}")
        
        # Create train/val split
        train_dir, val_dir = data_pipeline.create_validation_split(dataset_path)
        
        # Initialize trainer
        trainer = LoRATrainer()
        
        # Prepare data
        train_loader, val_loader = trainer.prepare_data(train_dir, val_dir, batch_size)
        
        # Train
        results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            save_dir="lora_adapters",
            use_wandb=use_wandb
        )
        
        # Save training results
        job_id = f"finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        status_file = Path("training_jobs") / f"{job_id}_status.json"
        status_file.parent.mkdir(exist_ok=True)
        
        status = {
            "job_id": job_id,
            "status": "completed",
            "dataset_path": dataset_path,
            "results": results,
            "completed_at": datetime.now().isoformat()
        }
        
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)
        
        logger.info(f"Fine-tuning completed successfully. Results saved to {status_file}")
        
    except Exception as e:
        logger.error(f"Error in fine-tuning: {e}")
        
        # Save error status
        job_id = f"finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        status_file = Path("training_jobs") / f"{job_id}_status.json"
        status_file.parent.mkdir(exist_ok=True)
        
        status = {
            "job_id": job_id,
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.now().isoformat()
        }
        
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2) 