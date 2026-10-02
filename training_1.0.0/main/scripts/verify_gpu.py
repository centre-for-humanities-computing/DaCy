"""
Sanity check that the GPU is available to CuPy and spaCy.

Prints the CuPy version and the name of CUDA device 0, then calls
spacy.require_gpu() so spaCy runs on the GPU. require_gpu() raises
an error if no GPU is available, and returns True on success.
"""
import cupy
import spacy

# Sanity: CuPy sees the GPU
print("CuPy:", cupy.__version__, "| device:",
      cupy.cuda.runtime.getDeviceProperties(0)["name"].decode())
print("Require gpu:", spacy.require_gpu())