from typing import Set, Tuple, List, Callable

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


def fill_grid_backtracking(
    to_fill: np.ndarray,
    word_grid: np.ndarray,
    coverage: np.ndarray,
    trie: WordTrie,
    progress_bar: tqdm = None,
    max_checked_locations: int = 50,
    word_score_function: Callable[
        [np.ndarray, np.ndarray, Tuple[List[int], List[int]], float], float
    ] = None,
) -> Tuple[np.ndarray, np.ndarray] | None:
    if progress_bar:
        num_filled = np.sum(word_grid != GridState.OPEN)
        progress_bar.n = num_filled
        progress_bar.refresh()

    # Identify our targets to fill (any of the to_fill which is still uncovered)
    to_fill_left = to_fill & ~coverage

    # If we have filled all of the to_fill spots, then we are done
    if not np.any(to_fill_left):
        return (word_grid, coverage)

    # Otherwise, we need to pick some of our to_fill spots for us to iterate over and check. Each of these are candidates for our choice to fill words from.
    candidate_locations = list(zip(*np.where(to_fill_left)))
    np.random.shuffle(candidate_locations)
    candidate_locations = candidate_locations[:max_checked_locations]

    # Prepare the scoring function with up to date information about the current grid state
    open = word_grid == GridState.OPEN

    if word_score_function:

        def _score(
            word_indices: Tuple[List[int], List[int]],
            word_rank: float,
            random_value: float,
        ) -> float:
            return word_score_function(
                coverage, open, word_indices, word_rank, random_value
            )

    else:

        def _score(
            word_indices: Tuple[List[int], List[int]],
            word_rank: float,
            random_value: float,
        ) -> float:
            return random_value

    best_location_score = 0
    best_location_options = None

    for x, y in candidate_locations[:max_checked_locations]:
        current_options = []

        for slice_indices, slice, index_in_slice in all_matrix_slices_at(
            word_grid, x, y
        ):
            if len(slice) < trie.word_limits()[0] or np.all(slice != GridState.OPEN):
                continue

            current_options.extend(
                trie.get_top_k_valid_words(
                    slice,
                    slice_indices,
                    index_in_slice,
                    scoring_function=_score,
                )
            )

        if len(current_options) == 0:
            # If we can't fill this spot with anything, then we need to backtrack, since nothing we do will change that.
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
            best_location_options = current_options
            best_location_score = current_location_score

    # Now we know that there must be a best location

    # Now that we have found the most promising position, we will order our
    # words by their scores, and try each one at a time, moving to the next if
    # we end up backtracking.
    best_location_options.sort(key=lambda x: x[0], reverse=True)
    for _, word, indices in best_location_options:
        # If this option would not change our current board, then we don't need to place it, since that won't progress our solution.
        if np.sum(~coverage[indices]) == 0:
            continue

        # Place the word
        new_word_grid, new_coverage = word_grid.copy(), coverage.copy()
        new_word_grid[indices] = word
        new_coverage[indices] = True

        if not at_least_one_option_per_spot(new_word_grid, new_coverage, trie):
            continue

        backtracking_result = fill_grid_backtracking(
            to_fill=to_fill,
            word_grid=new_word_grid,
            coverage=new_coverage,
            trie=trie,
            progress_bar=progress_bar,
            max_checked_locations=max_checked_locations,
            word_score_function=word_score_function,
        )
        if backtracking_result is not None:
            return backtracking_result

    return None


# TODO: we should only seed up to the target density, even if there are already
# other entries in the grid. We also should avoid overwriting those entries if
# they are there.
def seed_grid(word_grid: np.ndarray, density: float = 0.075) -> np.ndarray:
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
    word_grid: np.ndarray,
    trie: WordTrie,
    progress_bar: bool = True,
) -> WordSearch | None:
    if progress_bar:
        progress_bar = tqdm(total=word_grid.size)

    seeded_grid = seed_grid(word_grid)
    if progress_bar:
        num_filled = np.sum(seeded_grid != GridState.OPEN)
        progress_bar.n = num_filled
        progress_bar.refresh()

    initial_coverage, _ = calculate_grid_coverage(seeded_grid, trie)

    # First fill targets are the uncovered, but already set options
    is_uncovered = (seeded_grid != GridState.OPEN) & (initial_coverage == False)

    if progress_bar:
        progress_bar.set_description("Covering uncovered")

    def _uncovered_scoring_fn(
        coverage: np.ndarray,
        open: np.ndarray,
        word_indices: Tuple[List[int], List[int]],
        word_rank: float,
        random_value: float,
    ) -> float:
        word_coverage = np.sum(~coverage[word_indices])
        word_intersection = np.sum(coverage[word_indices])
        fills_open = np.sum(open[word_indices])

        # If placing the word would not change our board, then we don't care
        if fills_open == 0:
            return 0

        # At this point, we mainly care about covering the uncovered
        # preset letters, but we will slightly prefer words which intersect with
        # other words and words which fill open spots, to break ties
        word_score = (
            word_coverage
            + 0.6 * word_intersection
            - 1.5 * fills_open
            + 2 * word_rank
            + random_value
        )
        return word_score

    uncovered_result = fill_grid_backtracking(
        is_uncovered,
        seeded_grid,
        initial_coverage,
        trie,
        progress_bar,
        word_score_function=_uncovered_scoring_fn,
    )
    if uncovered_result is None:
        return None
    partially_completed, partial_coverage = uncovered_result

    if progress_bar:
        progress_bar.set_description("Covering open")

    # Now, we need to fill the remaining open spots
    def _open_scoring_fn(
        coverage: np.ndarray,
        open: np.ndarray,
        word_indices: Tuple[List[int], List[int]],
        word_rank: float,
        random_value: float,
    ) -> float:
        word_intersection = np.sum(coverage[word_indices])
        fills_open = np.sum(open[word_indices])

        # If placing the word would not change our board, then we don't care
        if fills_open == 0:
            return 0

        # Now, we want to prioritize intersecting with existing words,
        # and focus on words which are more frequent, if possible
        word_score = (
            word_intersection + 0.5 * fills_open + 2 * word_rank + 0.2 * random_value
        )
        return word_score

    fill_result = fill_grid_backtracking(
        ~partial_coverage,
        partially_completed,
        partial_coverage,
        trie,
        progress_bar,
        word_score_function=_open_scoring_fn,
    )
    if fill_result is None:
        return None
    filled_grid, _ = fill_result

    if progress_bar:
        progress_bar.n = filled_grid.size
        progress_bar.refresh()

    final_coverage, final_words = calculate_grid_coverage(filled_grid, trie)
    if not np.all(final_coverage):
        raise RuntimeError("Should not be possible!")  # TODO: better error
    return WordSearch(filled_grid, final_words)


if __name__ == "__main__":
    # 0-25 Letters
    # 26 Unset / Needs to be solved for
    # np.inf Not a valid location

    # TODO: the way we are doing this right now means that we don't backtrack up through the open cover stuff
    # when we hit a snag and need to backtrack

    trie = WordTrie()
    print(f"Constructed trie of depth {trie.depth()} from {len(trie)} words...")

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
    grid = np.ones((13, 13), dtype=int) * 26
    print(convert_matrix_to_letters(grid))
    print(solve(grid, trie))
