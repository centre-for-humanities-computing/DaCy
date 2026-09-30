
import cupy
import spacy

# Sanity: CuPy sees the GPU
print("CuPy:", cupy.__version__, "| device:",
      cupy.cuda.runtime.getDeviceProperties(0)["name"].decode())

spacy.require_gpu()
print("Require gpu:", spacy.require_gpu())