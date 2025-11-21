import torch
import sys

print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")

print("\n--- Checking ctcdecode ---")
try:
    from ctcdecode import CTCBeamDecoder
    print("SUCCESS: ctcdecode imported correctly!")
except ImportError as e:
    print(f"FAILURE: Could not import ctcdecode.\nError: {e}")
except Exception as e:
    print(f"FAILURE: Crashed during import.\nError: {e}")

print("\n--- Checking pyctcdecode (Alternative) ---")
try:
    from pyctcdecode import build_ctcdecoder
    print("SUCCESS: pyctcdecode imported correctly!")
except ImportError:
    print("FAILURE: pyctcdecode not installed.")