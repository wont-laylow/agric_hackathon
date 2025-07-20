from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    """
    Application settings are defined here.
    All paths and configuration variables should be managed from this file.
    """
    APP_NAME: str = "Crop Disease Analyzer"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    MODEL_NAME: str = "mobilenetv3_large_100"

    MODEL_PATH: Path = BASE_DIR / "model_store" / "mobilenet_deployment_pipeline.joblib"
    KNOWLEDGE_BASE_PATH: Path = BASE_DIR / "app" / "data" / "knowledge_base.json"
    
    DATABASE_URL: str = f"sqlite+aiosqlite:///{BASE_DIR / 'crop_analyzer.db'}"

    CLASS_NAMES: list[str] = [
        "Cashew_gumosis", "Cashew_healthy", "Cashew_red rust", "Cashew_anthracnose", 
        "Cashew_leaf miner", "Cassava_green mite", "Cassava_bacterial blight", 
        "Cassava_mosaic", "Cassava_healthy", "Cassava_brown spot", "Maize_leaf beetle", 
        "Maize_healthy", "Maize_leaf blight", "Maize_grasshoper", "Maize_fall armyworm", 
        "Maize_streak virus", "Maize_leaf spot", "Tomato_verticulium wilt", 
        "Tomato_septoria leaf spot", "Tomato_healthy", "Tomato_leaf blight", 
        "Tomato_leaf curl"
    ]

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

settings = Settings()