import numpy as np

from tqdm import tqdm

from dataclasses import dataclass

from trie import WordTrie

from util import convert_matrix_to_letters


@dataclass
class WordSearch:
    word_inclusion_mask: np.ndarray
    word_search_grid: np.ndarray


# TODO: update this so that we stop when we get to an excluded entry
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


def solve(initial: WordSearch, trie: WordTrie) -> WordSearch | None:
    # First, we need to figure out valid choices for each of our unchosen locations
    for _ in range(20):
        current_grid = initial.word_search_grid
        next_location, options = generate_options(
            initial.word_search_grid, initial.word_inclusion_mask, trie
        )
        current_grid[next_location] = options[0]
        print(convert_matrix_to_letters(current_grid))


if __name__ == "__main__":
    # 0-25 Letters
    # 26 Unset / Needs to be solved for
    # np.inf Not a valid location

    trie = WordTrie()

    grid = np.array(
        [
            [26, 8, 26, 26, 11],
            [26, 26, 14, 26, 26],
            [21, 26, 26, 26, 4],
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
    solve(WordSearch(word_inclusion_mask=inclusion_mask, word_search_grid=grid), trie)
