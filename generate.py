from typing import Set, Tuple, List

import numpy as np
from tqdm import tqdm

from dataclasses import dataclass
from trie import WordTrie
from util import convert_matrix_to_letters, all_matrix_slices, all_matrix_slices_at

from datatypes import GridState


@dataclass
class WordSearch:
    word_grid: np.ndarray
    words: Set[str]


def calculate_grid_coverage(
    word_search_grid: np.ndarray, trie: WordTrie
) -> Tuple[np.ndarray, Set[str]]:
    # Calculate which locations of the grid are already contained in a word, and which
    # initial words we have
    coverage_grid = np.zeros_like(word_search_grid, dtype=bool)
    words = set()

    for indices, slice in all_matrix_slices(word_search_grid):
        slice_coverage, slice_words = trie.coverage(slice)
        coverage_grid[indices] = coverage_grid[indices] | slice_coverage
        words.update(slice_words)

    return coverage_grid, words


def at_least_one_option(
    word_search_grid: np.ndarray, trie: WordTrie, x: int, y: int
) -> bool:
    for _, slice, index_in_slice in all_matrix_slices_at(word_search_grid, x, y):
        for _ in trie.get_valid_words(slice, index_in_slice):
            return True
    return False


def at_least_one_option_per_spot(
    word_search_grid: np.ndarray, coverage: np.ndarray, trie: WordTrie
) -> bool:
    for x, y in zip(*np.where(coverage == False)):
        if not at_least_one_option(word_search_grid, trie, x, y):
            return False
    return True


def fill_open_backtracking(
    word_grid: np.ndarray,
    coverage: np.ndarray,
    trie: WordTrie,
    progress_bar: tqdm = None,
) -> np.ndarray | None:
    if progress_bar:
        num_filled = np.sum(word_grid != GridState.OPEN)
        progress_bar.n = num_filled
        progress_bar.refresh()

    # If we have filled the grid, then we are done
    if np.all(coverage):
        return word_grid

    # Identify all open letters
    is_open = word_grid == GridState.OPEN

    best_location = None
    best_location_score = 0
    best_location_options = None

    # Now, iterate over the uncovered preset letter locations and find the most
    # promising position
    for x, y in zip(*np.where(is_open)):
        current_options = []

        for slice_indices, slice, index_in_slice in all_matrix_slices_at(
            word_grid, x, y
        ):

            def _score_word(
                word: List[int],
                word_indices: Tuple[List[int], List[int]],
                word_rank: int,
            ) -> float:
                word_intersection = np.sum(coverage[word_indices])
                fills_open = np.sum(is_open[word_indices])

                # If placing the word would not change our board, then we don't care
                if fills_open == 0:
                    return 0

                # Now, we want to prioritize intersecting with existing words
                word_score = word_intersection + 0.5 * fills_open
                return word_score

            current_options.extend(
                trie.get_top_k_valid_words(
                    slice, slice_indices, index_in_slice, scoring_function=_score_word
                )
            )

        if len(current_options) == 0:
            # If we can't fill this spot with anything, then we need to backtrack
            return None

        current_location_score = sum([score for score, _, _ in current_options])
        if (
            best_location_options is None
            or len(current_options) < len(best_location_options)
            or (
                len(current_options) == len(best_location_options)
                and current_location_score > best_location_score
            )
        ):
            best_location = (x, y)
            best_location_options = current_options
            best_location_score = current_location_score

    if best_location is None or len(best_location_options) == 0:
        return None

    # Now that we have found the most promising position, we will order our
    # words by their scores, and try each one at a time, moving to the next if
    # we end up backtracking.
    best_location_options.sort(key=lambda x: x[0], reverse=True)
    for _, word, indices in best_location_options:
        if np.sum(is_open[indices]) == 0:
            continue

        # Place the word
        new_word_grid, new_coverage = word_grid.copy(), coverage.copy()
        new_word_grid[indices] = word
        new_coverage[indices] = True

        if not at_least_one_option_per_spot(new_word_grid, new_coverage, trie):
            continue

        solved_word_grid = fill_open_backtracking(
            new_word_grid, new_coverage, trie, progress_bar=progress_bar
        )
        if solved_word_grid is not None:
            return solved_word_grid

    return None


def fill_uncovered_backtracking(
    word_grid: np.ndarray,
    coverage: np.ndarray,
    trie: WordTrie,
    progress_bar: tqdm = None,
) -> np.ndarray | None:
    if progress_bar:
        num_filled = np.sum(word_grid != GridState.OPEN)
        progress_bar.n = num_filled
        progress_bar.refresh()

    # First, identify all uncovered preset letters
    is_uncovered = (word_grid != GridState.OPEN) & (coverage == False)
    open_spots = word_grid == GridState.OPEN

    # If we have already covered all of the preset letters, then we can simply start the
    # next backtracking step
    if np.all(is_uncovered == False):
        return fill_open_backtracking(word_grid, coverage, trie, progress_bar)

    best_location = None
    best_location_score = 0
    best_location_options = None

    # Now, iterate over the uncovered preset letter locations and find the most
    # promising position
    for x, y in zip(*np.where(is_uncovered)):
        current_options = []

        for slice_indices, slice, index_in_slice in all_matrix_slices_at(
            word_grid, x, y
        ):

            def _score_word(
                word: List[int],
                word_indices: Tuple[List[int], List[int]],
                word_rank: int,
            ) -> float:
                word_coverage = np.sum(is_uncovered[word_indices])
                word_intersection = np.sum(coverage[word_indices])
                fills_open = np.sum(open_spots[word_indices])

                # If placing the word would not change our board, then we don't care
                if fills_open == 0:
                    return 0

                # At this point, we mainly care about covering the uncovered
                # preset letters, but we will slightly prefer words which intersect with
                # other words and words which fill open spots, to break ties
                word_score = word_coverage + 0.6 * word_intersection - 0.1 * fills_open
                return word_score

            current_options.extend(
                trie.get_top_k_valid_words(
                    slice, slice_indices, index_in_slice, scoring_function=_score_word
                )
            )

        if len(current_options) == 0:
            # If we can't fill this spot with anything, then we need to backtrack
            return None

        current_location_score = sum([score for score, _, _ in current_options])
        if (
            best_location_options is None
            or len(current_options) < len(best_location_options)
            or (
                len(current_options) == len(best_location_options)
                and current_location_score > best_location_score
            )
        ):
            best_location = (x, y)
            best_location_options = current_options
            best_location_score = current_location_score

    if best_location is None or len(best_location_options) == 0:
        return None

    # Now that we have found the most promising position, we will order our
    # words by their scores, and try each one at a time, moving to the next if
    # we end up backtracking.
    best_location_options.sort(key=lambda x: x[0], reverse=True)
    for _, word, word_indices in best_location_options:
        if np.sum(open_spots[word_indices]) == 0:
            continue

        # Place the word
        new_word_grid, new_coverage = word_grid.copy(), coverage.copy()
        new_word_grid[word_indices] = word
        new_coverage[word_indices] = True

        if not at_least_one_option_per_spot(new_word_grid, new_coverage, trie):
            continue

        solved_word_grid = fill_uncovered_backtracking(
            new_word_grid, new_coverage, trie, progress_bar=progress_bar
        )
        if solved_word_grid is not None:
            return solved_word_grid

    return None


# TODO: do the seeding with actual words, so that we don't have issues filling them
def seed_grid(word_grid: np.ndarray, density: float = 0.10) -> np.ndarray:
    width, height = word_grid.shape
    num_samples = int(word_grid.size * density)
    rx, ry = np.random.randint(0, width, num_samples), np.random.randint(
        0, height, num_samples
    )
    rv = np.random.randint(0, 26, num_samples)

    new_word_grid = word_grid.copy()
    for x, y, v in zip(rx, ry, rv):
        new_word_grid[x, y] = v

    return new_word_grid


def solve(
    word_grid: np.ndarray, trie: WordTrie, progress_bar: bool = True
) -> WordSearch | None:
    if progress_bar:
        progress_bar = tqdm(total=word_grid.size)

    seeded_grid = seed_grid(word_grid)
    if progress_bar:
        num_filled = np.sum(seeded_grid != GridState.OPEN)
        progress_bar.n = num_filled
        progress_bar.refresh()

    initial_coverage, _ = calculate_grid_coverage(seeded_grid, trie)

    filled_grid = fill_uncovered_backtracking(
        seeded_grid,
        initial_coverage,
        trie,
        progress_bar=progress_bar if progress_bar else None,
    )
    if filled_grid is None:
        return None

    if progress_bar:
        progress_bar.n = word_grid.size
        progress_bar.refresh()

    final_coverage, final_words = calculate_grid_coverage(filled_grid, trie)
    if not np.all(final_coverage):
        raise RuntimeError("Should not be possible!")  # TODO: better error
    return WordSearch(filled_grid, final_words)


if __name__ == "__main__":
    # 0-25 Letters
    # 26 Unset / Needs to be solved for
    # np.inf Not a valid location

    trie = WordTrie()
    print(f"Constructed trie...")

    # grid = np.array(
    #     [
    #         [26, 8, 26, 26, 26, 11],
    #         [26, 26, 14, 26, 26, 26],
    #         [21, 26, 26, 26, 4, 26],
    #         [26, 26, 26, 26, 26, 26],
    #         [26, 26, 24, 26, 26, 14],
    #         [26, 26, 20, 26, 26, 26],
    #     ],
    #     dtype=int,
    # )
    grid = np.ones((15, 15), dtype=int) * 26
    print(convert_matrix_to_letters(grid))
    print(solve(grid, trie))
