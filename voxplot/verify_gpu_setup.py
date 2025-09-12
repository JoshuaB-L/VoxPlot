#!/usr/bin/env python3
"""
GPU Setup Verification Script for VoxPlot
=========================================

This script verifies that your system is properly configured for GPU-accelerated
ML analysis in VoxPlot. It checks for:

1. NVIDIA GPU hardware
2. CUDA runtime availability
3. CuPy installation and functionality
4. cuML installation and functionality
5. VoxPlot GPU optimization compatibility

Usage:
    python verify_gpu_setup.py
"""

import sys
import subprocess
import importlib.util

def print_header(title):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_status(item, status, details=""):
    """Print a status line."""
    status_symbol = "✅" if status else "❌"
    print(f"{status_symbol} {item}")
    if details:
        print(f"   {details}")

def check_nvidia_gpu():
    """Check if NVIDIA GPU is available."""
    print_header("NVIDIA GPU Hardware Check")
    
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print_status("NVIDIA GPU detected", True)
            # Extract GPU info from nvidia-smi output
            lines = result.stdout.split('\n')
            for line in lines:
                if 'NVIDIA' in line and ('GeForce' in line or 'Tesla' in line or 'Quadro' in line or 'RTX' in line):
                    gpu_info = line.strip()
                    print(f"   GPU: {gpu_info}")
                    break
            return True
        else:
            print_status("NVIDIA GPU", False, "nvidia-smi command failed")
            return False
    except FileNotFoundError:
        print_status("NVIDIA GPU", False, "nvidia-smi not found - no NVIDIA drivers installed")
        return False
    except subprocess.TimeoutExpired:
        print_status("NVIDIA GPU", False, "nvidia-smi command timed out")
        return False

def check_cupy():
    """Check CuPy installation and functionality."""
    print_header("CuPy Installation Check")
    
    try:
        import cupy as cp
        print_status("CuPy import", True, f"Version: {cp.__version__}")
        
        # Test CUDA availability
        try:
            device_count = cp.cuda.runtime.getDeviceCount()
            print_status("CUDA runtime", True, f"Detected {device_count} CUDA device(s)")
            
            # Test basic GPU operation
            try:
                a = cp.array([1, 2, 3])
                b = cp.array([4, 5, 6])
                c = a + b
                result = cp.asnumpy(c)
                print_status("Basic GPU computation", True, f"Test result: {result}")
                return True
                
            except Exception as e:
                print_status("Basic GPU computation", False, f"Error: {str(e)}")
                return False
                
        except Exception as e:
            print_status("CUDA runtime", False, f"Error: {str(e)}")
            return False
            
    except ImportError as e:
        print_status("CuPy import", False, f"ImportError: {str(e)}")
        print("   Install with: conda install -c conda-forge cupy")
        return False

def check_cuml():
    """Check cuML installation and functionality."""
    print_header("cuML Installation Check")
    
    try:
        import cuml
        print_status("cuML import", True, f"Version: {cuml.__version__}")
        
        # Test basic cuML functionality
        try:
            from cuml.cluster import KMeans as cuKMeans
            print_status("cuML KMeans import", True)
            
            # Test basic clustering operation
            try:
                import cupy as cp
                import numpy as np
                
                # Create small test dataset
                np.random.seed(42)
                X = np.random.rand(100, 3).astype(np.float32)
                X_gpu = cp.asarray(X)
                
                # Test cuML KMeans
                kmeans = cuKMeans(n_clusters=3, random_state=42)
                labels = kmeans.fit_predict(X_gpu)
                
                print_status("cuML KMeans test", True, f"Clustered 100 points into 3 clusters")
                return True
                
            except Exception as e:
                print_status("cuML KMeans test", False, f"Error: {str(e)}")
                return False
                
        except ImportError as e:
            print_status("cuML KMeans import", False, f"ImportError: {str(e)}")
            return False
            
    except ImportError as e:
        print_status("cuML import", False, f"ImportError: {str(e)}")
        print("   Install with: conda install -c conda-forge -c rapidsai cuml")
        return False

def check_voxplot_compatibility():
    """Check VoxPlot ML optimizer compatibility."""
    print_header("VoxPlot GPU Compatibility Check")
    
    try:
        # Try to import VoxPlot ML optimizer
        from ml_optimizer import MLOptimizer, GPU_AVAILABLE, GPU_ERROR_MSG
        
        if GPU_AVAILABLE:
            print_status("VoxPlot GPU support", True, "All GPU libraries detected by VoxPlot")
            
            # Test VoxPlot optimizer initialization
            try:
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
                
                optimizer = MLOptimizer(config)
                print_status("VoxPlot MLOptimizer", True, "GPU-enabled optimizer initialized successfully")
                return True
                
            except Exception as e:
                print_status("VoxPlot MLOptimizer", False, f"Error: {str(e)}")
                return False
        else:
            print_status("VoxPlot GPU support", False, f"Error: {GPU_ERROR_MSG}")
            return False
            
    except ImportError as e:
        print_status("VoxPlot ML optimizer import", False, f"ImportError: {str(e)}")
        print("   Make sure you're running this script from the VoxPlot directory")
        return False

def main():
    """Main verification function."""
    print("VoxPlot GPU Setup Verification")
    print("This script will check your GPU acceleration setup for VoxPlot ML analysis.")
    
    all_checks = []
    
    # Run all checks
    all_checks.append(check_nvidia_gpu())
    all_checks.append(check_cupy())
    all_checks.append(check_cuml())
    all_checks.append(check_voxplot_compatibility())
    
    # Summary
    print_header("Summary")
    
    passed_checks = sum(all_checks)
    total_checks = len(all_checks)
    
    if passed_checks == total_checks:
        print("🎉 All checks passed! Your system is ready for GPU-accelerated VoxPlot ML analysis.")
        print("\nTo enable GPU acceleration in VoxPlot:")
        print("1. Set compute_backend: 'gpu' in your config_ml.yaml")
        print("2. Set use_gpu_acceleration: true in your config_ml.yaml")
        print("3. Run your analysis with the --verbose flag to see GPU utilization")
    else:
        print(f"❌ {total_checks - passed_checks} out of {total_checks} checks failed.")
        print("\nTo fix issues:")
        print("1. Install CUDA Toolkit from NVIDIA")
        print("2. Install GPU libraries: conda install -c conda-forge -c rapidsai cuml cupy")
        print("3. Verify your NVIDIA drivers are up to date")
        print("4. Check that your GPU is CUDA-compatible")
        
    return passed_checks == total_checks

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)