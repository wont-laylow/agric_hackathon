# Crop Disease Analyzer - AI-Powered Agricultural Assistant

## 🌱 Overview

The Crop Disease Analyzer is a comprehensive AI-powered system designed to help farmers identify and manage crop diseases through image analysis and intelligent chatbot assistance. Built with FastAPI, PyTorch, and advanced machine learning techniques, this system provides real-time disease detection and expert agricultural advice.

## 🎯 Key Features

- **Multi-Crop Disease Detection**: Supports Cashew, Cassava, Maize, and Tomato crops
- **Dual-Model Architecture**: Combines CLIP gatekeeper for image validation and MobileNetV3 ensemble for disease classification
- **Intelligent Chatbot**: Local RAG-based chatbot providing detailed disease information and treatment advice
- **Web Interface**: User-friendly web application with real-time image upload and analysis
- **Comprehensive Knowledge Base**: Detailed information on 22 different disease classes
- **Confidence Scoring**: Top-5 predictions with confidence percentages
- **Database Logging**: Persistent storage of prediction results and user interactions

## 🏗️ System Architecture

### Core Components

1. **FastAPI Web Application** (`app/main.py`)
   - RESTful API endpoints
   - Web interface with Jinja2 templates
   - Static file serving
   - Application lifecycle management

2. **Machine Learning Pipeline** (`app/ml/inference.py`)
   - CLIP-based image validation gatekeeper
   - MobileNetV3 ensemble for disease classification
   - CPU-optimized inference for reliability

3. **Local Knowledge Bot** (`app/chatbot/local_service.py`)
   - Sentence transformer embeddings
   - FAISS vector search
   - Rule-based response generation

4. **RAG Pipeline** (`rag_pipeline/`)
   - Knowledge base ingestion
   - Vector store management
   - Document retrieval and processing

5. **Database Layer** (`app/db/`)
   - SQLite database with async support
   - Prediction logging
   - User management

## 🤖 Machine Learning Models

### 1. CLIP Gatekeeper Model
- **Purpose**: Validates uploaded images to ensure they contain plant leaves
- **Model**: OpenAI CLIP ViT-B/32
- **Function**: `is_image_a_leaf_permissive()`
- **Validation Logic**:
  - Plant prompts: "a close-up photo of a single plant leaf", "a diseased leaf from a farm crop"
  - Non-plant prompts: "a photo of a person", "a car, building, or street scene"
  - Decision criteria: Plant score > (non-plant score + 0.20) AND plant score > 0.60

### 2. MobileNetV3 Ensemble Model
- **Purpose**: Disease classification across 22 classes
- **Architecture**: MobileNetV3-Large-100 with custom classifier
- **Classes**: 22 disease categories across 4 crops
- **Ensemble**: Multiple models for improved accuracy
- **Output**: Top-5 predictions with confidence scores

#### Supported Disease Classes

**Cashew (5 classes):**
- Cashew_gumosis
- Cashew_healthy
- Cashew_red_rust
- Cashew_anthracnose
- Cashew_leaf_miner

**Cassava (5 classes):**
- Cassava_green_mite
- Cassava_bacterial_blight
- Cassava_mosaic
- Cassava_healthy
- Cassava_brown_spot

**Maize (7 classes):**
- Maize_leaf_beetle
- Maize_healthy
- Maize_leaf_blight
- Maize_grasshoper
- Maize_fall_armyworm
- Maize_streak_virus
- Maize_leaf_spot

**Tomato (5 classes):**
- Tomato_verticulium_wilt
- Tomato_septoria_leaf_spot
- Tomato_healthy
- Tomato_leaf_blight
- Tomato_leaf_curl

### 3. Local Knowledge Bot
- **Embedding Model**: SentenceTransformer 'all-MiniLM-L6-v2'
- **Vector Store**: FAISS IndexFlatL2
- **Search**: Semantic similarity search
- **Response Generation**: Rule-based keyword matching

## 📊 Knowledge Base Structure

The system maintains a comprehensive knowledge base (`app/data/knowledge_base.json`) with detailed information for each disease:

```json
{
  "crop_name": [
    {
      "name": "disease_name",
      "information": "General description",
      "causes": "Detailed cause information",
      "effects": "Symptoms and effects",
      "diagnosis": "How to identify the disease",
      "recommended_chemical_treatment": "Treatment recommendations"
    }
  ]
}
```

## 🔧 Technical Implementation

### Model Architecture Details

#### MobileNetV3 Crop Model
```python
class MobileNetV3CropModel(nn.Module):
    def __init__(self, model_name="mobilenetv3_large_100", num_classes=22):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False)
        num_features = self.backbone.num_features
        self.backbone.reset_classifier(0)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Flatten(), 
            nn.BatchNorm1d(num_features),
            nn.Dropout(0.2), 
            nn.Linear(num_features, num_features // 4),
            nn.ReLU(inplace=True), 
            nn.BatchNorm1d(num_features // 4),
            nn.Dropout(0.1), 
            nn.Linear(num_features // 4, num_classes)
        )
```

#### Local Knowledge Bot Architecture
```python
class LocalKnowledgeBot:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.raw_data, self.processed_chunks = self._load_and_process_kb()
        self.vector_store = self._build_vector_store()
```

### API Endpoints

#### Web Interface
- `GET /` - Main upload page
- `POST /predict` - Image upload and disease prediction

#### REST API
- `POST /api/chat` - Chatbot interaction endpoint

### Database Schema

#### PredictionLog Table
```sql
CREATE TABLE prediction_logs (
    id INTEGER PRIMARY KEY,
    image_filename VARCHAR,
    predicted_class VARCHAR,
    predicted_crop VARCHAR,
    predicted_disease VARCHAR,
    confidence FLOAT,
    is_healthy BOOLEAN,
    prediction_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

## 🚀 Installation and Setup

### Prerequisites
- Python 3.8+
- PyTorch
- FastAPI
- SQLite
- Required ML libraries (see requirements.txt)

### Environment Variables
Create a `.env` file with:
```
OPENAI_API_KEY=your_openai_api_key
DEBUG=False
```

### Installation Steps
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download the trained model to `model_store/mobilenet_deployment_pipeline.joblib`
4. Run the application: `uvicorn app.main:app --reload`

## 📁 Project Structure

```
hackathon/
├── app/
│   ├── api/                 # API endpoints
│   ├── chatbot/            # Local knowledge bot
│   ├── core/               # Configuration
│   ├── db/                 # Database models and CRUD
│   ├── ml/                 # Machine learning inference
│   ├── models/             # Data models and schemas
│   ├── routes/             # Additional routes
│   ├── static/             # CSS and static files
│   ├── templates/          # HTML templates
│   ├── utils/              # Utility functions
│   └── data/               # Knowledge base JSON
├── rag_pipeline/           # RAG implementation
├── model_store/            # Trained models
├── faiss_index/            # Vector store indices
├── knowledge_base/         # Raw knowledge base files
└── tests/                  # Test files
```

## 🔄 Workflow

### 1. Image Upload and Validation
1. User uploads image through web interface
2. CLIP gatekeeper validates image contains plant leaves
3. If validation fails, user receives error message

### 2. Disease Classification
1. Validated image is processed through MobileNetV3 ensemble
2. Model generates top-5 predictions with confidence scores
3. Results are displayed to user with detailed breakdown

### 3. Chatbot Interaction
1. User asks questions about detected disease
2. Local knowledge bot searches vector store for relevant information
3. Rule-based system generates contextual responses
4. Responses include symptoms, causes, diagnosis, and treatment advice

### 4. Data Logging
1. All predictions are logged to database
2. User interactions are tracked for analytics
3. System maintains audit trail of all activities

## 🎨 User Interface

The web interface provides:
- **Image Upload**: Drag-and-drop or file selection
- **Real-time Preview**: Image preview before analysis
- **Prediction Display**: Top-5 results with confidence scores
- **Interactive Chatbot**: Built-in chat interface for disease queries
- **Responsive Design**: Mobile-friendly interface

## 🔒 Security and Reliability

### Image Validation
- CLIP-based gatekeeper prevents non-plant images
- Fail-closed mechanism ensures system safety
- Comprehensive error handling

### Model Reliability
- CPU-only inference for maximum compatibility
- Ensemble models for improved accuracy
- Confidence thresholds for decision making

### Data Privacy
- Local processing for sensitive data
- No external API calls for core functionality
- Secure database storage

## 📈 Performance Characteristics

### Model Performance
- **CLIP Gatekeeper**: Real-time image validation
- **MobileNetV3 Ensemble**: Fast inference with high accuracy
- **Local Knowledge Bot**: Sub-second response times

### System Scalability
- Async database operations
- Efficient vector search with FAISS
- Optimized model loading and caching

## 🔮 Future Enhancements

### Planned Features
- **Multi-language Support**: Localization for different regions
- **Mobile App Integration**: Native mobile application
- **Advanced Analytics**: Detailed performance metrics
- **Model Retraining**: Continuous learning capabilities
- **Additional Crops**: Support for more agricultural crops

### Technical Improvements
- **GPU Acceleration**: Optional GPU support for faster inference
- **Model Compression**: Quantized models for mobile deployment
- **Real-time Updates**: Live knowledge base updates
- **Advanced RAG**: More sophisticated retrieval mechanisms

## 🤝 Contributing

This project is designed to support agricultural communities worldwide. Contributions are welcome in the following areas:
- Additional crop and disease support
- Model improvements and optimizations
- User interface enhancements
- Knowledge base expansion
- Testing and validation

## 📄 License

This project is developed to support agricultural communities and promote sustainable farming practices.

## 🙏 Acknowledgments

- Agricultural experts for disease knowledge
- Open source ML community for model architectures
- Farmers and agricultural workers for domain expertise

---

**Built with ❤️ for the agricultural community**
