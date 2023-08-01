import enum
import struct
from dataclasses import dataclass
from typing import Iterable, Iterator

import largestinteriorrectangle as lir
import numpy as np
from numba import njit
from scipy.sparse import coo_matrix


def pad(a: np.ndarray, shape: tuple):
    return np.pad(
        a,
        ((0, shape[0] - a.shape[0]), (0, shape[1] - a.shape[1])),
        "constant",
        constant_values=0,
    )


@dataclass
class Patch:
    i: np.uint16
    j: np.uint16
    k: np.uint8
    brash: tuple[np.uint8, np.uint8] | tuple[np.uint16, np.uint16] | np.ndarray

    def __post_init__(self):
        if isinstance(self.brash, np.ndarray):
            if self.brash.shape not in ((4, 4), (8, 8), (16, 16)):
                raise ValueError(f"Invalid brash shape: {self.brash.shape}")
            self.brash = self.brash.astype("bool")

    class Flag(enum.Flag):
        BRASH = enum.auto()
        BIG_RECT = enum.auto()
        HEX = enum.auto()
        OCT = enum.auto()
        DENSE = enum.auto()

    @property
    def size(self):
        match self.brash:
            case np.ndarray:
                return self.brash.size
            case (a, b):
                return a * b

    @property
    def width(self):
        match self.brash:
            case np.ndarray():
                return self.brash.shape[1]
            case (a, _):
                return a

    @property
    def height(self):
        match self.brash:
            case np.ndarray():
                return self.brash.shape[0]
            case (_, b):
                return b

    @property
    def slices(self):
        return slice(self.i, self.i + self.height), slice(self.j, self.j + self.height)

    @property
    def flag(self):
        flag = Patch.Flag(0)
        if isinstance(self.brash, np.ndarray):
            flag |= Patch.Flag.BRASH
            if self.brash.shape[0] == 16:
                flag |= Patch.Flag.HEX
            elif self.brash.shape[0] == 8:
                flag |= Patch.Flag.OCT
            if self.brash.sum() >= self.brash.size / 8:
                flag |= Patch.Flag.DENSE
        if self.height >= 256 or self.width >= 256:
            flag |= Patch.Flag.BIG_RECT
        return flag

    def to_bytes(self):
        result = struct.pack("!HHBB", self.i, self.j, self.k, self.flag.value)
        if self.flag & Patch.Flag.BRASH and self.flag & Patch.Flag.DENSE:
            return result + np.packbits(self.brash).tobytes()
        elif self.flag & Patch.Flag.BRASH:
            coo = coo_matrix(self.brash)
            return result + bytes(r << 4 | c for r, c in zip(coo.row, coo.col))
        elif self.flag & Patch.Flag.BIG_RECT:
            return result + struct.pack("!HH", self.width, self.height)
        else:
            return result + bytes(self.brash)

    def __str__(self) -> str:
        if isinstance(self.brash, np.ndarray):
            brash = np.packbits(self.brash)
        else:
            brash = self.brash
        return f"Patch({self.i: >5}, {self.j: >5}, {self.k: >3}, {brash})"


@njit("boolean[:,:,::1](uint8[:,:,::1])", parallel=True, cache=True)
def split_channels_to_layers(image: np.ndarray):
    results = np.zeros(image.shape[:-1] + (image.shape[-1] * 8,), dtype="bool")
    for k in range(image.shape[-1]):
        for l in range(8):
            results[..., l] = (image[..., k] & (1 << l)) != 0
    return results


def scrape_off(image: np.ndarray, scraper_size: int = 8) -> Iterator[Patch]:
    """Scrape off the image into patches of size `core_size`.

    Parameters
    ----------
    image : np.ndarray
        An RGB(A) image, with shape (i, j, k), where i and j are
        less than 65535, k is less than 32 (256 / 8).
    scraper_size : {4, 8, 16}, default 8
        The size of the scraper to use. 4, 8, and 16 are supported.

    Yields
    ------
    Iterator[Patch]
        _description_
    """
    x, y = image.shape[:2]
    assert x <= 65535 and y <= 65535
    assert 4 <= scraper_size <= 8
    max_i = x // scraper_size + 1
    max_j = y // scraper_size + 1
    layers = split_channels_to_layers(image)
    for c in range(layers.shape[-1]):
        mask = layers[..., c].astype("bool")
        while True:
            j, i, ncols, nrows = lir.lir(mask)
            patch = Patch(i, j, c, (nrows, ncols))
            if patch.size < 8 * struct.calcsize("!HHBBHH" if nrows > 255 or ncols > 255 else "!HHBBBB"):
                break
            mask[patch.slices] = False
            yield patch
        for i, row in enumerate(np.array_split(mask, max_i, axis=0)):
            for j, col in enumerate(np.array_split(row, max_j, axis=1)):
                if col.any():
                    yield Patch(min((i + 1) * scraper_size, x), min((j + 1) * scraper_size, y), c, pad(col, (scraper_size, scraper_size)))


def paint(patches: Iterable[Patch], canvas_shape: tuple, canvas_dtype: np.dtype):
    canvas = np.zeros(canvas_shape, dtype=canvas_dtype)
    for patch in patches:
        k, l = patch.color // canvas.dtype.itemsize * 8, patch.color % canvas.dtype.itemsize * 8
        canvas[tuple(*patch.slices, k)] |= 1 << l
