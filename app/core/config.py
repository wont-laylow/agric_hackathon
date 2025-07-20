# =============================================================================
# FILE: app/core/config.py (FINAL, COMPLETE VERSION)
# =============================================================================
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

# Define the base path of the project, which is the 'Ghana-ai' directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    """
    Application settings are defined here.
    All paths and configuration variables should be managed from this file.
    """
    # Application settings
    APP_NAME: str = "Crop Disease Analyzer"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Model name required by the MobileNetV3CropModel class definition
    MODEL_NAME: str = "mobilenetv3_large_100"

    # --- PATH SETTINGS ---
    # Path to the trained machine learning model pipeline
    MODEL_PATH: Path = BASE_DIR / "model_store" / "mobilenet_deployment_pipeline.joblib"
    
    # --- THIS IS THE MISSING LINE THAT FIXES THE ERROR ---
    # Path to the JSON file containing disease information
    KNOWLEDGE_BASE_PATH: Path = BASE_DIR / "app" / "data" / "knowledge_base.json"
    
    # Database settings
    DATABASE_URL: str = f"sqlite+aiosqlite:///{BASE_DIR / 'crop_analyzer.db'}"

    # List of class names the model was trained on
    CLASS_NAMES: list[str] = [
        "Cashew_gumosis", "Cashew_healthy", "Cashew_red rust", "Cashew_anthracnose", 
        "Cashew_leaf miner", "Cassava_green mite", "Cassava_bacterial blight", 
        "Cassava_mosaic", "Cassava_healthy", "Cassava_brown spot", "Maize_leaf beetle", 
        "Maize_healthy", "Maize_leaf blight", "Maize_grasshoper", "Maize_fall armyworm", 
        "Maize_streak virus", "Maize_leaf spot", "Tomato_verticulium wilt", 
        "Tomato_septoria leaf spot", "Tomato_healthy", "Tomato_leaf blight", 
        "Tomato_leaf curl"
    ]

    # Pydantic model configuration
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

# Create a single instance of the settings to be imported by other modules
settings = Settings()