# Setup Guide

Complete step-by-step guide to get the Search Ranking System up and running.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git
- (Optional) Docker for containerized deployment

## Local Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/search-ranking-system.git
cd search-ranking-system
```

### 2. Create Virtual Environment

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download NLTK Data (if needed)

```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### 5. Generate Sample Data

```bash
python scripts/generate_sample_data.py
```

This will create:
- `data/sample_documents.json` - 100 sample documents
- `data/sample_queries.json` - 50 sample queries
- `data/training_data.json` - 30 training examples

### 6. Train Models

```bash
python scripts/train_model.py
```

This will train and save:
- TF-IDF model
- BM25 model
- LambdaMART model

Expected output:
```
=== Training TF-IDF Ranker ===
Loaded 100 documents
Model saved to models/tfidf_model.pkl

=== Training BM25 Ranker ===
Loaded 100 documents
Model saved to models/bm25_model.pkl

=== Training LambdaMART Ranker ===
Loaded 100 documents
Loaded 30 training samples
Training model (this may take a few minutes)...
Model saved to models/lambdamart_model.pkl
```

### 7. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_rankers.py -v
```

### 8. Start the API Server

```bash
python src/api/app.py
```

The API will be available at `http://localhost:5000`

### 9. Test the API

**Index Documents:**
```bash
curl -X POST http://localhost:5000/index \
  -H "Content-Type: application/json" \
  -d @data/sample_documents.json
```

**Search:**
```bash
curl -X POST http://localhost:5000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning algorithms",
    "algorithm": "bm25",
    "top_k": 5
  }'
```

**Health Check:**
```bash
curl http://localhost:5000/health
```

## Docker Setup

### 1. Build Docker Image

```bash
docker build -t search-ranking-system .
```

### 2. Run Container

```bash
docker run -p 5000:5000 search-ranking-system
```

### 3. Test Docker Deployment

```bash
curl http://localhost:5000/health
```

## Running Experiments

### Run A/B Testing Experiments

```bash
python scripts/run_experiments.py
```

This will:
1. Evaluate ranking quality for all algorithms
2. Run A/B tests comparing different rankers
3. Save results to `data/` directory

Expected output includes:
- Ranking quality metrics (nDCG, MAP, MRR, etc.)
- Statistical significance tests
- Performance comparisons

## Project Structure After Setup

```
search-ranking-system/
├── data/
│   ├── sample_documents.json          # Generated
│   ├── sample_queries.json            # Generated
│   ├── training_data.json             # Generated
│   └── *_results.json                 # Experiment results
├── models/
│   ├── tfidf_model.pkl                # Trained model
│   ├── bm25_model.pkl                 # Trained model
│   └── lambdamart_model.pkl           # Trained model
└── ...
```

## Troubleshooting

### Issue: ModuleNotFoundError

**Solution:** Make sure you're in the virtual environment and have installed all dependencies:
```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Issue: XGBoost installation fails

**Solution:** Install build tools:

**macOS:**
```bash
brew install cmake libomp
pip install xgboost
```

**Ubuntu/Debian:**
```bash
sudo apt-get install build-essential
pip install xgboost
```

**Windows:**
Download Visual C++ Build Tools and install, then:
```bash
pip install xgboost
```

### Issue: Port 5000 already in use

**Solution:** Change the port in `config.py`:
```python
API_PORT = 5001  # or any available port
```

### Issue: Out of memory during training

**Solution:** Reduce the dataset size or model complexity in `config.py`:
```python
LAMBDAMART_N_ESTIMATORS = 100  # reduce from 500
```

## Configuration

Edit `config.py` to customize:

### BM25 Parameters
```python
BM25_K1 = 1.5  # Term frequency saturation
BM25_B = 0.75   # Length normalization
```

### LambdaMART Parameters
```python
LAMBDAMART_N_ESTIMATORS = 500
LAMBDAMART_LEARNING_RATE = 0.1
LAMBDAMART_MAX_DEPTH = 6
```

### A/B Testing
```python
AB_TEST_TRAFFIC_SPLIT = 0.5
AB_TEST_MIN_SAMPLE_SIZE = 1000
AB_TEST_CONFIDENCE_LEVEL = 0.95
```

## Next Steps

1. **Customize Data**: Replace sample data with your own documents
2. **Tune Parameters**: Experiment with different algorithm parameters
3. **Add Features**: Extend the feature extractor with domain-specific features
4. **Deploy**: Deploy to cloud platforms (Google Cloud Run, AWS, etc.)
5. **Monitor**: Add logging and monitoring for production use

## API Endpoints Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/index` | POST | Index documents |
| `/search` | POST | Search documents |
| `/evaluate` | POST | Evaluate ranking quality |
| `/experiment/create` | POST | Create A/B test |
| `/experiment/<id>/record` | POST | Record experiment metric |
| `/experiment/<id>/results` | GET | Get experiment results |
| `/experiments` | GET | List all experiments |
| `/documents` | GET | Get indexed documents |
| `/stats` | GET | Get system statistics |

## Development Workflow

1. Make changes to code
2. Run tests: `pytest tests/ -v`
3. Test locally: Start API and test endpoints
4. Run experiments: `python scripts/run_experiments.py`
5. Commit changes: `git add . && git commit -m "your message"`
6. Push to repository: `git push`

## Support

For issues or questions:
- Open an issue on GitHub
- Check the documentation
- Review test files for usage examples

---

**Happy Searching! 🔍**
