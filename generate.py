from typing import Set, Tuple, List

from dataclasses import dataclass
from tqdm import tqdm
import numpy as np

from util import (
    convert_matrix_to_letters,
    all_matrix_slices,
    all_matrix_slices_at,
)
from datatypes import GridState
from trie import WordTrie


def score_word(
    coverage: np.ndarray,
    open_slots: np.ndarray,
    word_indices: Tuple[List[int], List[int]],
    word_rank: float,
    random_value: float,
) -> float:
    word_intersection = np.sum(coverage[word_indices])
    fills_open_slots = np.sum(open_slots[word_indices])

    # If placing the word would not change our board, then we don't care
    if fills_open_slots == 0:
        return 0

    # Now, we want to prioritize intersecting with existing words,
    # and focus on words which are more frequent, if possible
    word_score = (
        1.5 * word_intersection
        + 0.5 * fills_open_slots
        + 2 * word_rank
        + 0.2 * random_value
    )
    return word_score


@dataclass
class WordSearch:
    word_grid: np.ndarray
    words: Set[str]


def grid_coverage(grid: np.ndarray, trie: WordTrie) -> np.ndarray:
    coverage_grid = np.zeros_like(grid, dtype=bool)

    for indices, slice in all_matrix_slices(grid):
        slice_coverage = trie.coverage(slice)
        coverage_grid[indices] = coverage_grid[indices] | slice_coverage

    return coverage_grid


def grid_words(grid: np.ndarray, trie: WordTrie) -> Set[str]:
    words = set()

    for _, slice in all_matrix_slices(grid):
        words.update(trie.words(slice))

    return words


def at_least_one_option(
    word_search_grid: np.ndarray,
    blocked: np.ndarray,
    trie: WordTrie,
    x: int,
    y: int,
) -> bool:
    for slice_indices, slice, index_in_slice in all_matrix_slices_at(
        word_search_grid,
        x,
        y,
    ):
        for _ in trie.get_valid_words(
            slice=slice, blocked=blocked[slice_indices], target_index=index_in_slice
        ):
            return True
    return False


def at_least_one_option_per_non_blocked_spot(
    word_search_grid: np.ndarray,
    blocked: np.ndarray,
    coverage: np.ndarray,
    trie: WordTrie,
) -> bool:
    for x, y in zip(*np.where(~(coverage | blocked))):
        if not at_least_one_option(
            word_search_grid=word_search_grid,
            blocked=blocked,
            trie=trie,
            x=x,
            y=y,
        ):
            return False
    return True


def fill_grid_backtracking(
    word_grid: np.ndarray,
    coverage: np.ndarray,
    trie: WordTrie,
    progress_bar: tqdm = None,
    max_checked_locations: int = 50,
) -> Tuple[np.ndarray, np.ndarray] | None:
    if progress_bar:
        num_filled = np.sum(word_grid != GridState.OPEN)
        progress_bar.n = num_filled
        progress_bar.refresh()

    uncovered_and_not_blocked = ~coverage & ~blocked
    if not np.any(uncovered_and_not_blocked):
        return word_grid

    uncovered_and_not_blocked_with_letter = (
        word_grid != GridState.OPEN
    ) & uncovered_and_not_blocked

    if np.any(uncovered_and_not_blocked_with_letter):
        to_fill = uncovered_and_not_blocked_with_letter
    else:
        to_fill = uncovered_and_not_blocked

    # Otherwise, we need to pick some of our to_fill spots for us to iterate over and check. Each of these are candidates for our choice to fill words from.
    candidate_locations = list(zip(*np.where(to_fill)))
    np.random.shuffle(candidate_locations)
    candidate_locations = candidate_locations[:max_checked_locations]

    # Prepare the scoring function with up to date information about the current grid state
    open_slots = word_grid == GridState.OPEN

    def _score(
        word_indices: Tuple[List[int], List[int]],
        word_rank: float,
        random_value: float,
    ) -> float:
        return score_word(coverage, open_slots, word_indices, word_rank, random_value)

    best_location_score = 0
    best_location_options = None

    for x, y in candidate_locations:
        current_options = []

        for slice_indices, slice, index_in_slice in all_matrix_slices_at(
            word_grid, x, y
        ):
            if len(slice) < trie.word_limits()[0] or np.all(slice != GridState.OPEN):
                continue

            current_options.extend(
                trie.get_top_k_valid_words(
                    slice=slice,
                    blocked=blocked[slice_indices],
                    indices=slice_indices,
                    target_index=index_in_slice,
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
        new_word_grid = word_grid.copy()
        new_word_grid[indices] = word
        new_coverage = grid_coverage(new_word_grid, trie)

        if not at_least_one_option_per_non_blocked_spot(
            word_search_grid=new_word_grid,
            blocked=blocked,
            coverage=new_coverage,
            trie=trie,
        ):
            # There is an unblocked, uncovered location which cannot be filled
            continue

        if np.any(new_coverage & blocked):
            # There is a blocked letter which was covered accidentally
            # This can happen if the blocked letter is an 's', for example, and
            # we put 'cat' right next to it, to make 'cats'.
            continue

        backtracking_result = fill_grid_backtracking(
            word_grid=new_word_grid,
            coverage=new_coverage,
            trie=trie,
            progress_bar=progress_bar,
            max_checked_locations=max_checked_locations,
        )
        if backtracking_result is not None:
            return backtracking_result

    return None


def seed_grid(word_grid: np.ndarray, density: float = 0.05) -> np.ndarray | None:  #
    width, height = word_grid.shape
    current_fill = np.sum(word_grid != GridState.OPEN)
    num_samples = int((word_grid.size - current_fill) * density)
    rx, ry = np.random.randint(0, width, num_samples * 5), np.random.randint(
        0, height, num_samples * 5
    )
    rv = np.random.randint(0, 26, num_samples * 5)

    new_word_grid = word_grid.copy()
    num_placed = 0
    for x, y, v in zip(rx, ry, rv):
        if new_word_grid[x, y] == GridState.OPEN:
            new_word_grid[x, y] = v
            num_placed += 1
            if num_placed >= num_samples:
                return new_word_grid

    return None


def solve(
    word_grid: np.ndarray,
    blocked: np.ndarray,
    trie: WordTrie,
    progress_bar: bool = True,
) -> WordSearch | None:
    if progress_bar:
        progress_bar = tqdm(total=word_grid.size)

    seeded_grid = seed_grid(word_grid)
    print(convert_matrix_to_letters(seeded_grid))
    if progress_bar:
        num_filled = np.sum(seeded_grid != GridState.OPEN)
        progress_bar.n = num_filled
        progress_bar.refresh()

    initial_coverage = grid_coverage(seeded_grid, trie)

    # Ensure that none of our blocked entries have been covered
    if np.any(initial_coverage & blocked):
        return None

    result = fill_grid_backtracking(
        word_grid=seeded_grid,
        coverage=initial_coverage,
        trie=trie,
        progress_bar=progress_bar,
    )
    if result is None:
        return None

    if progress_bar:
        progress_bar.n = result.size
        progress_bar.refresh()

    final_coverage = grid_coverage(result, trie)
    final_coverage = final_coverage & blocked
    if not np.all(final_coverage):
        raise RuntimeError("Should not be possible!")  # TODO: better error

    words = grid_words(result, trie)
    return WordSearch(result, words)


if __name__ == "__main__":
    from util import string_to_alphabet_positions

    # 0-25 Letters
    # 26 Unset / Needs to be solved for
    # np.inf Not a valid location

    # TODO: the way we are doing this right now means that we don't backtrack up through the open_slots cover stuff
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
    grid = np.ones((6, 6), dtype=int) * 26
    message = "te amo"
    message = message.replace(" ", "").lower().strip()
    message_integers = string_to_alphabet_positions(message)
    print(message_integers)
    blocked = np.zeros_like(grid)
    locations = [(0, 0), (4, 1), (4, 5), (1, 2), (0, 4)]
    locations.sort()
    for location, value in zip(locations, message_integers):
        blocked[location] = 1
        grid[location] = value
    print(convert_matrix_to_letters(grid))
    print(blocked)
    print(solve(grid, blocked, trie))
