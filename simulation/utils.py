"""
Simulation Utilities for SAWM.
Provides image writing and helper functions without third-party imaging dependencies.
"""

import zlib
import struct
import numpy as np


def save_png(path: str, rgb_array: np.ndarray) -> None:
    """
    Save an RGB uint8 NumPy array (H, W, 3) directly to a PNG file.
    Uses only Python standard library (zlib, struct) without PIL or OpenCV.
    
    Args:
        path: Output file path (.png).
        rgb_array: NumPy array of shape (H, W, 3) and dtype uint8.
    """
    assert rgb_array.ndim == 3 and rgb_array.shape[2] == 3, "Array must be (H, W, 3)"
    assert rgb_array.dtype == np.uint8, "Array must be uint8"

    height, width, _ = rgb_array.shape

    # Prepend 0 filter byte to each scanline
    raw_data = b"".join(b"\x00" + rgb_array[y].tobytes() for y in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        length = struct.pack(">I", len(data))
        crc = struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        return length + tag + data + crc

    # PNG Signature + IHDR + IDAT + IEND
    header = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit, Truecolor RGB
    ihdr = chunk(b"IHDR", ihdr_data)
    idat = chunk(b"IDAT", zlib.compress(raw_data, level=6))
    iend = chunk(b"IEND", b"")

    with open(path, "wb") as f:
        f.write(header + ihdr + idat + iend)
