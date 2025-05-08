from typing import Set, Tuple

import numpy as np

from dataclasses import dataclass
from trie import WordTrie
from util import convert_matrix_to_letters, all_matrix_slices, all_matrix_slices_at

from datatypes import GridState


@dataclass
class WordSearch:
    word_grid: np.ndarray
    words: Set[str]


# TODO: update this so that we stop when we get to an excluded entry
# TODO: update this to only generate one slice for each direction, since the Trie handles reversibility
# TODO: change this to just take the array, or something. We don't actually need to do this for every location, since the Trie function returns values for every entry in the given slice. Each slice of the array only needs to be given once. But we do need to split on empty entries, so maybe some custom slicer function? But we don't want to store all of the word lists, only the best so far, so maybe recalculating things is fine?
def slice_directions(arr, x, y, max_length=18):
    max_rows, max_cols = arr.shape

    # TODO: maybe move this out so that we only compute it once
    # Create an index matrix of shape (rows, cols, 2)
    index_matrix = np.indices(arr.shape).transpose(1, 2, 0)  # shape: (rows, cols, 2)

    # Horizontal →
    end_x = min(x + max_length, max_cols)
    h_right = arr[y, x:end_x]
    h_right_idx = index_matrix[y, x:end_x]

    # Horizontal ←
    start_x = max(0, x - max_length + 1)
    h_left = arr[y, start_x : x + 1][::-1]
    h_left_idx = index_matrix[y, start_x : x + 1][::-1]

    # Vertical ↓
    end_y = min(y + max_length, max_rows)
    v_down = arr[y:end_y, x]
    v_down_idx = index_matrix[y:end_y, x]

    # Vertical ↑
    start_y = max(0, y - max_length + 1)
    v_up = arr[start_y : y + 1, x][::-1]
    v_up_idx = index_matrix[start_y : y + 1, x][::-1]

    # Diagonal ↘
    ddr_len = min(max_rows - y, max_cols - x, max_length)
    diag_indices = np.arange(ddr_len)
    d_down_right = arr[y + diag_indices, x + diag_indices]
    d_down_right_idx = index_matrix[y + diag_indices, x + diag_indices]

    # Diagonal ↖
    dul_len = min(y + 1, x + 1, max_length)
    diag_indices = np.arange(dul_len)
    d_up_left = arr[y - diag_indices, x - diag_indices]
    d_up_left_idx = index_matrix[y - diag_indices, x - diag_indices]

    return {
        "horizontal_right": (h_right, h_right_idx) if len(h_right) > 1 else None,
        "horizontal_left": (h_left, h_left_idx) if len(h_left) > 1 else None,
        "vertical_down": (v_down, v_down_idx) if len(v_down) > 1 else None,
        "vertical_up": (v_up, v_up_idx) if len(v_up) > 1 else None,
        "diagonal_down_right": (
            (d_down_right, d_down_right_idx) if len(d_down_right) > 1 else None
        ),
        "diagonal_up_left": (d_up_left, d_up_left_idx) if len(d_up_left) > 1 else None,
    }


# TODO: use letter frequency to influence sampling for the chosen location. Maybe even use frequency based on all the options at that location.


def generate_options(
    word_search_grid: np.ndarray, inclusion_mask: np.ndarray, trie: WordTrie
):
    chosen_locations = word_search_grid != 26
    word_search_width, word_search_height = word_search_grid.shape
    valid_choices = np.zeros((word_search_width, word_search_height, 26), dtype=bool)
    for x in range(word_search_width):
        for y in range(word_search_height):
            if chosen_locations[x, y]:
                continue
            # Take slices from this location in every direction
            slices_and_indices = slice_directions(word_search_grid, x, y)
            for entry in slices_and_indices.values():
                if entry is None:
                    continue

                slice, slice_indices = entry
                valid_slice_fill = trie.get_valid(slice.tolist())

                if valid_slice_fill is None:
                    continue

                for (x, y), valid_set in zip(slice_indices, valid_slice_fill):
                    if valid_set is None:
                        continue
                    for valid_entry in valid_set:
                        valid_choices[x, y, valid_entry] = True

    # Now find the entry which has the fewest options, but hasn't been set
    # TODO: maybe weight this selection by location, since the central squares will be more restrictive
    num_options = np.sum(valid_choices, axis=-1)
    if np.sum(num_options) == 0:
        return None
    selected_location = np.unravel_index(
        np.argmin(num_options + chosen_locations * 30), word_search_grid.shape
    )

    return selected_location, valid_choices[selected_location]


def calculate_grid_coverage(
    word_search_grid: np.ndarray, trie: WordTrie
) -> Tuple[np.ndarray, Set[str]]:
    # Calculate which locations of the grid are already contained in a word, and which initial words we have
    coverage_grid = np.zeros_like(word_search_grid, dtype=bool)
    words = set()

    for indices, slice in all_matrix_slices(word_search_grid):
        slice_coverage, slice_words = trie.coverage(slice)
        coverage_grid[indices] = coverage_grid[indices] | slice_coverage
        words.update(slice_words)

    return coverage_grid, words


def fill_open_backtracking(
    word_grid: np.ndarray, coverage: np.ndarray, trie: WordTrie
) -> np.ndarray | None:
    return None


def fill_uncovered_backtracking(
    word_grid: np.ndarray, coverage: np.ndarray, trie: WordTrie
) -> np.ndarray | None:
    # First, identify all uncovered preset letters
    is_uncovered = (word_grid != GridState.OPEN) & (coverage == False)

    # If we have already covered all of the preset letters, then we can simply start the next backtracking step
    if np.all(is_uncovered == False):
        return fill_open_backtracking(word_grid, coverage, trie)

    best_location = None
    best_location_value = 0
    best_location_options = None

    # Now, iterate over the uncovered preset letter locations and find the most promising position
    for x, y in zip(*np.where(is_uncovered)):
        current_options = set()
        print(x, y)
        for indices, slice in all_matrix_slices_at(word_grid, x, y):
            # Ignore slices that are full, or are not long enough (>= 3)
            if len(slice) < 3 or np.all(slice != GridState.OPEN):
                continue
            valid_words = trie.get_valid_words(slice)
            index_in_slice = np.where((indices[0] == x) & (indices[1] == y))[0][0]
            valid_words = valid_words[index_in_slice]

            for word, slice_position in valid_words:
                # Now we need to score this word
                word_length = len(word)
                word_indices = (
                    indices[0][slice_position : slice_position + word_length],
                    indices[1][slice_position : slice_position + word_length],
                )
                word_coverage = np.sum(is_uncovered[word_indices])

                print(word)
                print(word_coverage)
                asdf


def solve(word_grid: np.ndarray, trie: WordTrie) -> WordSearch | None:
    initial_coverage, initial_words = calculate_grid_coverage(word_grid, trie)

    filled_grid = fill_uncovered_backtracking(word_grid, initial_coverage, trie)
    if filled_grid is None:
        return None

    final_coverage, final_words = calculate_grid_coverage(filled_grid)
    if not np.all(final_coverage):
        raise RuntimeError("Should not be possible!")  # TODO: better error
    return WordSearch(filled_grid, final_words)


if __name__ == "__main__":
    # 0-25 Letters
    # 26 Unset / Needs to be solved for
    # np.inf Not a valid location

    trie = WordTrie()

    grid = np.array(
        [
            [26, 8, 26, 26, 11],
            [26, 26, 14, 26, 26],
            [21, 3, 14, 1, 4],
            [26, 26, 24, 26, 14],
            [26, 20, 26, 26, 26],
        ],
        dtype=int,
    )
    inclusion_mask = np.array(
        [
            [1, 0, 1, 1, 0],
            [1, 1, 0, 1, 1],
            [0, 1, 1, 1, 0],
            [1, 1, 0, 1, 0],
            [1, 0, 1, 1, 1],
        ],
        dtype=bool,
    )
    print(convert_matrix_to_letters(grid))
    solve(grid, trie)
