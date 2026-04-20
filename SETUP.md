# Setup Guide: AI Self-Pruning Engineering Suite

This guide provides step-by-step instructions to set up the environment and run the production-ready self-pruning neural network system.

## 1. Environment Requirements
- **Python**: 3.10 or 3.11 (Recommended)
- **OS**: Linux (Ubuntu 22.04+), macOS, or Windows (via WSL2)
- **Hardware**: CPU is sufficient for the demo; NVIDIA GPU (CUDA) is recommended for full CIFAR-10 training.

## 2. Local Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/AI-self-pruning.git
cd AI-self-pruning
```

### Step 2: Create a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Configuration
Create a `.env` file in the root directory if you wish to use the RAG Sparsity Explorer (Agentic feature):
```env
OPENAI_API_KEY=your_openai_api_key_here
```

## 4. Running the System

### A. Training the Model
To train the self-pruning network on synthetic data (Fast Demo):
```bash
python train.py --fast
```

### B. Starting the Inference Server (FastAPI)
```bash
python main.py
```
The API will be available at `http://localhost:8000`. You can access the interactive Swagger docs at `http://localhost:8000/docs`.

### C. Running Tests & Profiling
```bash
# Run Unit Tests
pytest test_core.py

# Run Performance Profiler
python profile_app.py
```

## 5. Docker Deployment (Recommended)
If you have Docker installed, you can run the entire suite without manual setup:

```bash
# Build the image
docker build -t ai-self-pruning .

# Run the container
docker run -p 8000:8000 ai-self-pruning
```

## 6. Project Structure
- `train.py`: Core ML logic and training loop.
- `main.py`: FastAPI backend for serving the model.
- `database.py`: SQLite/SQLAlchemy integration for logging.
- `explorer.py`: RAG-based analysis using FAISS and OpenAI.
- `test_core.py`: Validation suite.
- `profile_app.py`: Latency and memory profiling.
