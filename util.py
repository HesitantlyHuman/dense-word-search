import numpy as np


def convert_matrix_to_letters(matrix: np.ndarray) -> np.ndarray:
    letter_matrix = np.vectorize(lambda x: chr(x + 97))(matrix.astype(int))
    return letter_matrix


def convert_letters_to_matrix(letters: np.ndarray) -> np.ndarray:
    matrix = np.vectorize(lambda x: ord(x) - ord("a"))(letters)
    return matrix
