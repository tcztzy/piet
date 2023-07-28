import datetime
import enum
import multiprocessing as mp
import struct
from functools import partial
from itertools import pairwise, permutations

import cv2
import largestinteriorrectangle as lir
import numpy as np
import skimage.io
from reedsolo import RSCodec
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix

base_index = {"A": 0, "C": 1, "G": 2, "T": 3}

index_base = {0: "A", 1: "C", 2: "G", 3: "T"}

trans = str.maketrans("ACGT", "TGCA")

GRAPHS = tuple(
    sorted(
        [
            list(r0),
            [*r1, r0[2]],
            [*r2, r1[1], r0[1]],
            [r3, r2[0], r1[0], r0[0]],
        ]
        for r0 in permutations(range(4))
        for r1 in permutations(set(range(4)) - {r0[2]})
        if r0[0] != r1[0] and r0[1] != r1[1]
        for r2 in permutations(set(range(4)) - {r0[1], r1[1]})
        if r2[0] not in [r0[0], r1[0]] and r2[1] not in [r0[1], r1[1]]
        for r3 in set(range(4)) - {r0[0], r1[0], r2[0]}
    )
)


class RecordType(enum.Enum):
    RECT = 0
    CORE = 1


class MatrixType(enum.Enum):
    DENSE = 0
    COO = 1
    CSR = 2
    CSC = 3
    SUB_MIN_COO = 5
    SUB_MIN_CSR = 6
    SUB_MIN_CSC = 7
    REV_COO = 9


MAX_BYTES = 50
MIN_BYTES = 25
EC_BYTES = 4
CORE_SIZE = 6
STRUCT_FMT = (
    ">"  # big-endian
    "B"  # matrix type (4 bits) + matrix data length (4 bits)
    "B"  # records (4 bits) record type (2 bits) channel index (2 bits)
)


def calculate_compatible_score(dna_sequence: str):
    # calculate the compatibility score.
    homopolymer = 6
    for homopolymer in range(6, 0, -1):
        if (
            "A" * homopolymer in dna_sequence
            or "C" * homopolymer in dna_sequence
            or "G" * homopolymer in dna_sequence
            or "T" * homopolymer in dna_sequence
        ):
            break
    gc_bias = abs(
        (dna_sequence.count("G") + dna_sequence.count("C")) / len(dna_sequence) - 0.5
    )

    h_score = (1.0 - (homopolymer - 1) / 5.0) / 2.0 if homopolymer < 6 else 0
    c_score = (1.0 - gc_bias / 0.3) / 2.0 if gc_bias < 0.3 else 0

    return h_score + c_score


def calculate_density_score(dna_sequence: str, data_size_in_bytes: int):
    return max(1.0 - len(dna_sequence) / (data_size_in_bytes * 8), 0)


def calculate_score(dna_sequence: str, data_size_in_bytes: int):
    return 0.2 * calculate_density_score(
        dna_sequence, data_size_in_bytes
    ) + 0.3 * calculate_compatible_score(dna_sequence)


def fill_bytes(data: bytes, length: int):
    return data + b"\0" * (length - len(data))


def pad(a: np.ndarray, shape: tuple):
    return np.pad(
        a,
        ((0, shape[0] - a.shape[0]), (0, shape[1] - a.shape[1])),
        "constant",
        constant_values=0,
    )


def coo_to_bytes(data: np.ndarray):
    if (data == 255).sum() >= 20:
        matrix_type = MatrixType.REV_COO
        result = coo_to_bytes(255 - data)
        if result is not None:
            return (
                matrix_type,
                result[1],
                result[2],
            )
    min_val = data.min()
    matrix_type = MatrixType.COO if min_val == 0 else MatrixType.SUB_MIN_COO
    coo = coo_matrix(data - min_val)
    length = len(coo.data)
    if length <= 15:
        return (
            matrix_type,
            length,
            coo.row.astype("uint8").tobytes()
            + coo.col.astype("uint8").tobytes()
            + coo.data.tobytes()
            + (min_val.tobytes() if min_val else b""),
        )


def csr_to_bytes(data: np.ndarray):
    min_val = data.min()
    matrix_type = MatrixType.CSR if min_val == 0 else MatrixType.SUB_MIN_CSR
    csr = csr_matrix(data - min_val)
    length = len(csr.data)
    if length <= 15:
        return (
            matrix_type,
            length,
            csr.indptr.astype("uint8").tobytes()
            + csr.indices.astype("uint8").tobytes()
            + csr.data.tobytes()
            + (min_val.tobytes() if min_val else b""),
        )


def csc_to_bytes(data: np.ndarray):
    min_val = data.min()
    matrix_type = MatrixType.CSC if min_val == 0 else MatrixType.SUB_MIN_CSC
    csc = csc_matrix(data - min_val)
    length = len(csc.data)
    if length <= 15:
        return (
            matrix_type,
            length,
            csc.indptr.astype("uint8").tobytes()
            + csc.indices.astype("uint8").tobytes()
            + csc.data.tobytes()
            + (min_val.tobytes() if min_val else b""),
        )


def matrix_to_bytes(data: np.ndarray):
    results = [(MatrixType.DENSE, 0, data.tobytes())]
    coo_bytes = coo_to_bytes(data)
    if coo_bytes:
        results.append(coo_bytes)
    csr_bytes = csr_to_bytes(data)
    if csr_bytes:
        results.append(csr_bytes)
    csc_bytes = csc_to_bytes(data)
    if csc_bytes:
        results.append(csc_bytes)
    return results


def bytes_to_dna(data: bytes, graph_index=0):
    if len(data) > MAX_BYTES:
        return None
    result = ""
    current_base = "A"
    for byte in data:
        for i in range(4):
            upper = byte >> (6 - 2 * i)
            current_base = index_base[
                GRAPHS[graph_index][base_index[current_base]].index(upper)
            ]
            result += current_base
            byte &= 0b111111 >> 2 * i
    if 100 <= len(result) <= 200:
        return result


class Mondrian:
    def __init__(self, team_id: str):
        super().__init__(team_id)
        self.core_size = CORE_SIZE
        self.need_logs = False
        self.error_correction = RSCodec(EC_BYTES)

    def image_to_dna(self, input_image_path: str, need_logs: bool = True):
        if need_logs:
            start_time = datetime.datetime.now()
        self.need_logs = need_logs
        raw_data = cv2.imread(input_image_path)
        x, y = raw_data.shape[:2]
        max_i = x // self.core_size + 1
        max_j = y // self.core_size + 1
        data = np.array(
            [
                (
                    i * self.core_size + d.shape[0],
                    j * self.core_size + d.shape[1],
                    k,
                    pad(d, (self.core_size, self.core_size)).tobytes(),
                )
                for i, row in enumerate(np.array_split(raw_data, max_i, axis=0))
                for j, col in enumerate(np.array_split(row, max_j, axis=1))
                for k, d in enumerate(cv2.split(col))
                if not (d == 0).all()
            ],
            dtype=[
                ("i", np.uint16),
                ("j", np.uint16),
                ("k", np.uint8),
                ("data", "|S36"),
            ],
        )
        if need_logs:
            print(f"There are {len(data)} blocks.")
        data = data[data.argsort(order="data", kind="stable")]
        data = np.split(data, np.unique(data["data"], return_index=True)[1][1:])
        with mp.Pool() as pool:
            results = pool.starmap(
                self.silicon_to_carbon, [(x, y, self.core_size, d) for d in data]
            )
        dna_sequences = [
            bytes_to_dna(
                self.error_correction.encode(
                    fill_bytes(
                        struct.pack(
                            ">BBHHHH",
                            MatrixType.COO.value << 4,
                            16 | RecordType.RECT.value << 2 | k,
                            0,
                            0,
                            y,
                            x,
                        ),
                        MIN_BYTES - EC_BYTES,
                    )
                )
            )
            for k in range(raw_data.shape[2])
        ] + [s for r in results for s in r]
        if need_logs:
            print(
                max(
                    1.0
                    - sum([len(seq) for seq in dna_sequences]) / (raw_data.size * 8),
                    0,
                )
            )
            print(np.mean([calculate_compatible_score(seq) for seq in dna_sequences]))
            print(datetime.datetime.now() - start_time)
        return dna_sequences

    def silicon_to_carbon(self, x, y, core_size, data: np.ndarray):
        array = np.frombuffer(
            fill_bytes(data["data"][0], core_size**2), dtype="uint8"
        ).reshape(core_size, core_size)
        dna_sequences = []
        channels = np.unique(data["k"])
        for k in channels:
            data_k = data[data["k"] == k]
            if len(data_k) == 1 or array.var() > 0:
                for data in data_k:
                    dna_sequences.append(
                        self.data_to_dna(
                            k,
                            array,
                            RecordType.CORE,
                            (data["i"], data["j"]),
                        )
                    )
                continue
            grid = np.zeros((x, y), dtype="bool")
            for i, j, _, _ in data_k:
                i0 = (i - 1) // core_size * core_size
                j0 = (j - 1) // core_size * core_size
                grid[i0:i, j0:j] = True
            jx, ix, ncols, nrows = lir.lir(grid)
            while ncols * nrows > 0:
                grid[ix : ix + nrows, jx : jx + ncols] = False
                jx, ix, ncols, nrows = lir.lir(grid)
                if ncols > core_size or nrows > core_size:
                    record_type = RecordType.RECT
                    d = (jx, ix, ncols, nrows)
                else:
                    record_type = RecordType.CORE
                    d = (ix, jx)
                dna_sequences.append(self.data_to_dna(k, array, record_type, d))
        return dna_sequences

    def data_to_dna(
        self, k: int, array: np.ndarray, record_type: RecordType, data
    ) -> bytes:
        dna_sequences = [
            bytes_to_dna(
                self.error_correction.encode(
                    fill_bytes(
                        struct.pack(
                            STRUCT_FMT,
                            matrix_type.value << 4 | length,
                            (1 << 4) | record_type.value << 2 | k,
                        )
                        + packed_bytes
                        + struct.pack(
                            ">HH" if record_type == RecordType.CORE else ">HHHH",
                            *data,
                        ),
                        MIN_BYTES - EC_BYTES,
                    )
                )
            )
            for matrix_type, length, packed_bytes in matrix_to_bytes(array)
        ]
        dna_sequences = [s for s in dna_sequences if s is not None]
        if record_type == RecordType.CORE:
            i, j = data
            i0 = (i - 1) // self.core_size * self.core_size
            j0 = (j - 1) // self.core_size * self.core_size
            size = (i - i0) * (j - j0)
        elif record_type == RecordType.RECT:
            j, i, h, w = data
            size = h * w
        best = max(dna_sequences, key=partial(calculate_score, data_size_in_bytes=size))
        return best

    def dna_to_image(self, dna_sequences: list, output_image_path: str, need_logs=True):
        draft = np.zeros((0, 0, 3), dtype="uint8")
        for seq in dna_sequences:
            try:
                matrix, (i, j, k), (nrows, ncols) = self.carbon_to_silicon(seq)
                if i + nrows > draft.shape[0]:
                    draft = np.pad(
                        draft,
                        ((0, i + nrows), (0, 0), (0, 0)),
                        "constant",
                        constant_values=[0],
                    )
                if j + ncols > draft.shape[0]:
                    draft = np.pad(
                        draft,
                        ((0, 0), (0, j + ncols), (0, 0)),
                        "constant",
                        constant_values=[0],
                    )
                draft[i : i + nrows, j : j + ncols, k] = matrix[:nrows, :ncols]
            except Exception:
                pass
        skimage.io.imsave(output_image_path, draft)

    def carbon_to_silicon(self, dna_sequence: str):
        if len(dna_sequence) % 4 != 0:
            return ValueError
        graph = GRAPHS[base_index[dna_sequence[0]]]
        count = 0
        ba = bytearray()
        byte = 0
        for a, b in pairwise(dna_sequence[3:]):
            two = graph[base_index[a]][base_index[b]]
            byte = (byte << 2) | two
            count += 1
            if count % 4 == 0:
                ba.append(byte)
                byte = 0
        data, decoded_msgecc, errata_pos = self.error_correction.decode(ba)
        first_byte = data.pop(0)
        matrix_type, matrix_data_length = first_byte >> 4, first_byte & 15
        matrix_type = MatrixType(matrix_type)
        second_byte = data.pop(0)
        record_type, k = (
            second_byte & 12,
            second_byte & 3,
        )
        record_type = RecordType(record_type)
        matrix, data = self.unpack_matrix(data, matrix_type, matrix_data_length)
        if record_type == RecordType.CORE:
            i, j = struct.unpack(">HH", data[:4])
            i0 = (i - 1) // self.core_size * self.core_size
            j0 = (j - 1) // self.core_size * self.core_size
            nrows = i - i0
            ncols = j - j0
            i, j = i0, j0
            data = data[4:]
        elif record_type == RecordType.RECT:
            j, i, ncols, nrows = struct.unpack(">HHHH", data[:8])
            data = data[8:]
        else:
            raise TypeError
        return matrix, (i, j, k), (nrows, ncols)

    def unpack_matrix(self, data, matrix_type, matrix_data_length):
        if matrix_type == MatrixType.DENSE:
            matrix, data = data[: self.core_size**2], data[self.core_size**2 :]
            matrix = np.frombuffer(
                matrix, dtype="uint8", count=self.core_size**2
            ).reshape(self.core_size, self.core_size)
        elif matrix_type == MatrixType.COO:
            row = np.frombuffer(data, dtype="uint8", count=matrix_data_length)
            col = np.frombuffer(
                data, dtype="uint8", offset=matrix_data_length, count=matrix_data_length
            )
            matrix = np.frombuffer(
                data, dtype="uint8", offset=matrix_data_length, count=matrix_data_length
            )
            matrix = coo_matrix(
                (matrix, (row, col)), shape=(self.core_size, self.core_size)
            ).todense()
            data = data[3 * matrix_data_length :]
        elif matrix_type == MatrixType.REV_COO:
            matrix, data = self.unpack_matrix(data, MatrixType.COO, matrix_data_length)
            matrix = 255 - matrix
        elif matrix_type == MatrixType.SUB_MIN_COO:
            matrix, data = self.unpack_matrix(data, MatrixType.COO, matrix_data_length)
            matrix += data.pop(0)
        elif matrix_type == MatrixType.CSR:
            indptr = np.frombuffer(data, count=self.core_size + 1)
            indices = np.frombuffer(
                data, offset=self.core_size + 1, count=matrix_data_length
            )
            matrix = np.frombuffer(
                data,
                offset=self.core_size + 1 + matrix_data_length,
                count=matrix_data_length,
            )
            matrix = csr_matrix(
                (matrix, indices, indptr), shape=(self.core_size, self.core_size)
            ).todense()
            data = data[self.core_size + 1 + 2 * matrix_data_length :]
        elif matrix_type == MatrixType.CSC:
            indptr = np.frombuffer(data, count=self.core_size + 1)
            indices = np.frombuffer(
                data, offset=self.core_size + 1, count=matrix_data_length
            )
            matrix = np.frombuffer(
                data,
                offset=self.core_size + 1 + matrix_data_length,
                count=matrix_data_length,
            )
            matrix = csc_matrix(
                (matrix, indices, indptr), shape=(self.core_size, self.core_size)
            ).todense()
            data = data[self.core_size + 1 + 2 * matrix_data_length :]
        elif matrix_type == MatrixType.SUB_MIN_CSR:
            matrix, data = self.unpack_matrix(data, MatrixType.CSR, matrix_data_length)
            matrix += data.pop(0)
        elif matrix_type == MatrixType.SUB_MIN_CSC:
            matrix, data = self.unpack_matrix(data, MatrixType.CSC, matrix_data_length)
            matrix += data.pop(0)
        else:
            raise TypeError
        return matrix, data


def mondrianize():
    ...
