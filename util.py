from typing import Generator, Tuple, List

import numpy as np

from datatypes import GridState


def convert_list_to_string(list: List[int]) -> str:
    return "".join(
        [
            "_" if element == GridState.OPEN else chr(element + ord("a"))
            for element in list
        ]
    )


def convert_matrix_to_letters(matrix: np.ndarray) -> np.ndarray:
    letter_matrix = np.vectorize(lambda x: "_" if x == GridState.OPEN else chr(x + 97))(
        matrix.astype(int)
    )
    return letter_matrix


def convert_letters_to_matrix(letters: np.ndarray) -> np.ndarray:
    matrix = np.vectorize(lambda x: GridState.OPEN if x == "_" else ord(x) - ord("a"))(
        letters
    )
    return matrix


def all_matrix_slices(
    arr: np.ndarray,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    rows, cols = arr.shape

    # Horizontal slices (rows)
    for i in range(rows):
        idx = (np.full(cols, i), np.arange(cols))
        yield idx, arr[idx]

    # Vertical slices (columns)
    for j in range(cols):
        idx = (np.arange(rows), np.full(rows, j))
        yield idx, arr[idx]

    # Diagonals: top-left to bottom-right
    for offset in range(-rows + 1, cols):
        i_start = max(0, -offset)
        j_start = max(0, offset)
        length = min(rows - i_start, cols - j_start)
        i = np.arange(i_start, i_start + length)
        j = np.arange(j_start, j_start + length)
        yield (i, j), arr[i, j]

    # Diagonals: top-right to bottom-left
    for offset in range(-rows + 1, cols):
        i_start = max(0, -offset)
        j_start = min(cols - 1, cols - 1 - offset)
        length = min(rows - i_start, j_start + 1)
        i = np.arange(i_start, i_start + length)
        j = np.arange(j_start, j_start - length, -1)
        yield (i, j), arr[i, j]


def all_matrix_slices_at(
    arr: np.ndarray, x: int, y: int
) -> Generator[Tuple[np.ndarray, np.ndarray, int], None, None]:
    rows, cols = arr.shape

    # Horizontal slice at row x
    idx_row = (np.full(cols, x), np.arange(cols))
    slice = arr[idx_row]
    if np.any(slice == GridState.OPEN):
        yield idx_row, slice, x

    # Vertical slice at column y
    idx_col = (np.arange(rows), np.full(rows, y))
    slice = arr[idx_col]
    if np.any(slice == GridState.OPEN):
        yield idx_col, slice, y

    # Diagonal: top-left to bottom-right through (x, y)
    offset = y - x
    i_start = max(0, -offset)
    j_start = max(0, offset)
    length = min(rows - i_start, cols - j_start)
    i = np.arange(i_start, i_start + length)
    j = np.arange(j_start, j_start + length)
    slice = arr[i, j]
    if np.any(slice == GridState.OPEN):
        yield (i, j), slice, min(x, y)

    # Diagonal: top-right to bottom-left through (x, y)
    offset = y + x
    i_start = max(0, offset - (cols - 1))
    j_start = min(cols - 1, offset)
    length = min(rows - i_start, j_start + 1)
    i = np.arange(i_start, i_start + length)
    j = np.arange(j_start, j_start - length, -1)
    slice = arr[i, j]
    if np.any(slice == GridState.OPEN):
        yield (i, j), slice, min(rows - x, cols - y)
