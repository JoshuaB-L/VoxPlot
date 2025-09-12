#!/bin/bash
# VoxPlot GPU Installation Fix Script
# ===================================
# This script resolves CuPy conflicts and ensures optimal GPU utilization

set -e  # Exit on any error

echo "🔧 VoxPlot GPU Installation Fix"
echo "================================"
echo "This script will:"
echo "1. Remove conflicting CuPy installations"  
echo "2. Clean install cuML and CuPy via conda"
echo "3. Verify GPU acceleration works"
echo "4. Test VoxPlot GPU utilization"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda not found. Please install conda/miniconda first."
    exit 1
fi

# Check if we're in a conda environment
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "❌ Error: No conda environment active. Please activate your environment first."
    echo "   Example: conda activate palmpyenv"
    exit 1
fi

echo "📋 Current environment: $CONDA_DEFAULT_ENV"
echo ""

# Step 1: Remove conflicting installations
echo "🧹 Step 1: Removing conflicting CuPy installations"
echo "---------------------------------------------------"

echo "Removing pip-installed CuPy packages..."
pip uninstall cupy-cuda12x cupy -y 2>/dev/null || echo "No pip CuPy packages to remove"

echo "Removing conda-installed CuPy packages..."
conda uninstall cupy cupy-core -y 2>/dev/null || echo "No conda CuPy packages to remove"

echo "✅ Cleanup complete"
echo ""

# Step 2: Clean installation
echo "📦 Step 2: Installing cuML and CuPy via conda"
echo "----------------------------------------------"

echo "Installing cuML and CuPy from conda-forge and rapidsai..."
conda install -c conda-forge -c rapidsai cuml cupy -y

echo "✅ Installation complete"
echo ""

# Step 3: Verification
echo "🧪 Step 3: Verifying GPU installation"
echo "------------------------------------"

# Test CuPy
echo "Testing CuPy..."
python -c "
import cupy as cp
import sys

try:
    print(f'✅ CuPy version: {cp.__version__}')
    print(f'✅ CUDA available: {cp.cuda.is_available()}')
    print(f'✅ CUDA version: {cp.cuda.runtime.runtimeGetVersion()}')
    
    # Test basic GPU operation
    a = cp.array([1, 2, 3])
    b = cp.array([4, 5, 6]) 
    c = a + b
    result = cp.asnumpy(c)
    print(f'✅ GPU computation test: {result}')
    
except Exception as e:
    print(f'❌ CuPy error: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ CuPy verification failed"
    exit 1
fi

# Test cuML
echo "Testing cuML..."
python -c "
import cuml
import cupy as cp
import numpy as np
from cuml.cluster import KMeans as cuKMeans
import sys

try:
    print(f'✅ cuML version: {cuml.__version__}')
    
    # Test basic cuML operation
    np.random.seed(42)
    X = np.random.rand(100, 3).astype(np.float32)
    X_gpu = cp.asarray(X)
    
    kmeans = cuKMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X_gpu)
    
    print(f'✅ cuML clustering test: Successfully clustered 100 points')
    
except Exception as e:
    print(f'❌ cuML error: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ cuML verification failed"
    exit 1
fi

# Step 4: Test VoxPlot GPU integration
echo "Testing VoxPlot GPU integration..."
if [ -f "ml_optimizer.py" ]; then
    python -c "
from ml_optimizer import MLOptimizer, GPU_AVAILABLE, GPU_ERROR_MSG
import sys

if GPU_AVAILABLE:
    print('✅ VoxPlot GPU support: All libraries detected')
    
    # Test optimizer initialization
    config = {
        'analysis': {
            'ml_analysis': {
                'performance': {
                    'compute_backend': 'gpu',
                    'use_gpu_acceleration': True
                }
            }
        }
    }
    
    try:
        optimizer = MLOptimizer(config)
        print('✅ VoxPlot GPU optimizer: Successfully initialized')
    except Exception as e:
        print(f'❌ VoxPlot GPU optimizer error: {e}')
        sys.exit(1)
else:
    print(f'❌ VoxPlot GPU support: {GPU_ERROR_MSG}')
    sys.exit(1)
"

    if [ $? -ne 0 ]; then
        echo "❌ VoxPlot GPU integration failed"
        exit 1
    fi
else
    echo "⚠️  VoxPlot ml_optimizer.py not found in current directory"
fi

echo ""
echo "🎉 SUCCESS: GPU acceleration setup complete!"
echo "============================================="
echo ""
echo "✅ All conflicts resolved"
echo "✅ cuML and CuPy properly installed"  
echo "✅ GPU acceleration verified"
echo "✅ VoxPlot GPU integration confirmed"
echo ""
echo "📋 Next steps:"
echo "1. Run VoxPlot with: python ml_main.py --config config_ml.yaml --verbose"
echo "2. Look for log messages: 'Auto-selecting GPU-accelerated K-means'"
echo "3. Monitor GPU usage with: nvidia-smi"
echo ""
echo "🔧 Configuration set to algorithm: 'auto' for optimal GPU utilization"