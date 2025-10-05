"""
Configuration file for Search Ranking System
"""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 5000))
DEBUG_MODE = os.getenv("DEBUG", "False").lower() == "true"

# BM25 Parameters
BM25_K1 = 1.5  # Term frequency saturation parameter
BM25_B = 0.75  # Length normalization parameter

# TF-IDF Parameters
TFIDF_MAX_FEATURES = 10000
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.9

# LambdaMART Parameters
LAMBDAMART_N_ESTIMATORS = 500
LAMBDAMART_LEARNING_RATE = 0.1
LAMBDAMART_MAX_DEPTH = 6
LAMBDAMART_MIN_CHILD_WEIGHT = 1
LAMBDAMART_SUBSAMPLE = 0.8
LAMBDAMART_COLSAMPLE_BYTREE = 0.8

# Feature Extraction
MAX_QUERY_LENGTH = 100
MAX_DOC_LENGTH = 5000
USE_SEMANTIC_FEATURES = True

# Evaluation Metrics
EVAL_TOP_K = [1, 3, 5, 10, 20]
NDCG_K = 10

# A/B Testing
AB_TEST_TRAFFIC_SPLIT = 0.5  # 50-50 split
AB_TEST_MIN_SAMPLE_SIZE = 1000
AB_TEST_CONFIDENCE_LEVEL = 0.95

# Search Configuration
DEFAULT_TOP_K = 10
MAX_TOP_K = 100

# Model Paths
TFIDF_MODEL_PATH = MODELS_DIR / "tfidf_model.pkl"
BM25_MODEL_PATH = MODELS_DIR / "bm25_model.pkl"
LAMBDAMART_MODEL_PATH = MODELS_DIR / "lambdamart_model.pkl"

# Data Paths
DOCUMENTS_PATH = DATA_DIR / "sample_documents.json"
QUERIES_PATH = DATA_DIR / "sample_queries.json"
TRAINING_DATA_PATH = DATA_DIR / "training_data.json"

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
