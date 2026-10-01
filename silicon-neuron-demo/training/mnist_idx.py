"""
MNIST IDX loader with no third-party dependencies beyond NumPy.

IDX format (LeCun): big-endian header
    magic (4 B) : 0x00000803 for images, 0x00000801 for labels
    count (4 B)
    rows, cols (4 B each, images only)
followed by raw unsigned bytes.
"""
import gzip
import os
import struct

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def _open(path):
    return gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")


def read_images(path):
    with _open(path) as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 0x803, f"bad image magic {magic:#x}"
        buf = f.read(n * rows * cols)
    return np.frombuffer(buf, dtype=np.uint8).reshape(n, rows * cols)


def read_labels(path):
    with _open(path) as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 0x801, f"bad label magic {magic:#x}"
        buf = f.read(n)
    return np.frombuffer(buf, dtype=np.uint8)


def load_mnist(data_dir=DATA):
    """Returns (x_train, y_train, x_test, y_test); x is uint8 (N, 784)."""
    def p(name):
        for cand in (name + ".gz", name):
            full = os.path.join(data_dir, cand)
            if os.path.exists(full):
                return full
        raise FileNotFoundError(
            f"{name}[.gz] not found in {data_dir}. Download the four MNIST "
            "IDX files (e.g. from https://github.com/fgnt/mnist) into that folder.")

    return (read_images(p("train-images-idx3-ubyte")),
            read_labels(p("train-labels-idx1-ubyte")),
            read_images(p("t10k-images-idx3-ubyte")),
            read_labels(p("t10k-labels-idx1-ubyte")))


if __name__ == "__main__":
    xtr, ytr, xte, yte = load_mnist()
    print(xtr.shape, ytr.shape, xte.shape, yte.shape, xtr.dtype)
    print("fraction of non-zero pixels:", (xtr > 0).mean().round(3))
    print("fraction >= 64:", (xtr >= 64).mean().round(3))
