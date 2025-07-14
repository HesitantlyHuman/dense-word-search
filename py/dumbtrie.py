from typing import List, Self, Tuple, Set, Callable, Generator

import heapq
import numpy as np

import string
from dataclasses import dataclass, field

from datatypes import GridState

from util import (
    convert_matrix_to_letters,
    convert_letters_to_matrix,
    convert_list_to_string,
    string_to_alphabet_positions,
)


def default_letter_list_factory():
    return [None for _ in string.ascii_lowercase]


@dataclass(slots=True)
class TriePath:
    word: List[int]
    is_reverse: bool
    word_rank: float


@dataclass(slots=True)
class TrieNode:
    children: List[Self | None] = field(default_factory=default_letter_list_factory)
    path: TriePath = None

    def depth(self) -> int:
        child_depths = [child.depth() for child in self.children if child is not None]
        if len(child_depths) == 0:
            return 0
        return max(child_depths) + 1


# TODO: what if it's faster to just do a list of the words, at the size of our dictionary?
class WordTrie:
    """
    Provides functionality for getting valid letter states based on a slice.
    Supports both forwards and backwards embeddings of the provided word list.
    """

    def __init__(self, word_length_limits: Tuple[int, int] = (3, 10)):
        with open("word_lists/coca.txt") as f:
            word_list = f.readlines()

        smallest_word = 100
        biggest_word = 0

        filtered_words = []
        for word in word_list:
            word = word.strip().lower()
            if word_length_limits and (
                len(word) < word_length_limits[0] or len(word) > word_length_limits[1]
            ):
                continue
            smallest_word = min(smallest_word, len(word))
            biggest_word = max(biggest_word, len(word))
            filtered_words.append(word)
        self._word_limits = (smallest_word, biggest_word)
        self._num_entries = len(filtered_words)

        self.word_ranks = {}
        self.word_list = []

        self.root = TrieNode()
        for word_placement, word in enumerate(filtered_words):
            word_rank = 1 - (word_placement / self._num_entries)
            word = string_to_alphabet_positions(word)
            self.word_list.append(word)
            self.word_ranks[tuple(word)] = word_rank
            word.reverse()
            self.word_list.append(word)
            self.word_ranks[tuple(word)] = word_rank

        np.random.shuffle(self.word_list)

        self._depth = self.root.depth()

    def __len__(self) -> int:
        return self._num_entries

    def depth(self) -> int:
        return self._depth

    def word_limits(self) -> Tuple[int, int]:
        return self._word_limits

    def get_valid_words(
        self, slice: List[int], blocked: List[bool], target_index: int
    ) -> Generator[Tuple[List[int], float, Tuple[int, int]], None, None]:
        slice_size = len(slice)
        _, maximum_word_length = self.word_limits()

        # Narrow in on only the part of the slice which is not blocked
        previous_blocked = -1
        next_blocked = slice_size
        for idx, is_blocked in enumerate(blocked):
            if is_blocked:
                if idx > target_index:
                    next_blocked = min(next_blocked, idx)
                if idx < target_index:
                    previous_blocked = max(previous_blocked, idx)

        # Find the possible window that our words might live in
        starting_index = max(previous_blocked + 1, target_index - maximum_word_length)
        ending_index = min(next_blocked, target_index + maximum_word_length)

        def _spot_fits(word, start):
            if slice_size - start > len(word):
                return False

            for slice_element, word_element in zip(slice[start:], word):
                if slice_element == GridState.OPEN:
                    continue

                if slice_element != word_element:
                    return False

            return True

        def _word_fit(word) -> int | None:
            if len(word) > ending_index - starting_index:
                return None

            for start_pos in range(starting_index, ending_index):
                if _spot_fits(word, start_pos):
                    return start_pos

            return None

        for word in self.word_list:
            pos = _word_fit(word)
            if pos is not None:
                yield word, self.word_ranks[tuple(word)], (pos, pos + len(word))

    def get_top_k_valid_words(
        self,
        slice: List[int],
        blocked: List[bool],
        indices: Tuple[List[int], List[int]],
        target_index: str,
        scoring_function: Callable[[Tuple[List[int], List[int]], float], float] = None,
        k: int = 10,
        max_check: int = 2_000,
    ) -> List[List[int]] | None:
        k_largest = []

        random_values = np.random.random(max_check)

        idx = 0
        for word, word_rank, (start, stop) in self.get_valid_words(
            slice=slice, blocked=blocked, target_index=target_index
        ):
            word_indices = (
                indices[0][start:stop],
                indices[1][start:stop],
            )
            score = scoring_function(word_indices, word_rank, random_values[idx])

            if idx < k:
                heapq.heappush(k_largest, (score, idx, word, word_indices))
            elif score > k_largest[0][0]:
                heapq.heappushpop(k_largest, (score, idx, word, word_indices))

            if idx >= max_check - 1:
                break

            idx += 1

        return [
            (score, word, word_indices) for score, _, word, word_indices in k_largest
        ]  # If we want to sort later, we can

    def _words_from_slice(
        self, slice: List[int]
    ) -> Generator[Tuple[int, TrieNode], None, None]:
        def _spot_fits(word, start):
            if len(word) > len(slice) - start:
                return False

            for slice_element, word_element in zip(slice[start:], word):
                if slice_element != word_element:
                    return False

            return True

        def _word_fit(word) -> int | None:
            if len(word) > len(slice):
                return None

            for start_pos in range(0, len(slice)):
                if _spot_fits(word, start_pos):
                    return start_pos

            return None

        for word in self.word_list:
            pos = _word_fit(word)
            if pos is not None:
                yield pos, word

    def coverage(self, slice: List[int]) -> np.ndarray:
        coverage = np.zeros(len(slice), dtype=bool)

        for start, word in self._words_from_slice(slice):
            coverage[start : start + len(word)] = True

        return coverage

    def words(self, slice: List[int], trim: bool = True) -> Set[str]:
        # Get all words in the given slice
        words = set()

        for _, word in self._words_from_slice(slice):
            new_word = convert_list_to_string(word)

            if trim:
                # Since we have gotten this word after any words which started
                # after this word and ended before (or right now), we can find
                # any words that are included in this one easily.
                words.difference_update({word for word in words if word in new_word})

            words.add(new_word)

        return words


if __name__ == "__main__":
    import cProfile
    import timeit

    trie = WordTrie()
    print(f"Created trie with depth {trie.root.depth()}")

    words = trie.words(slice=[])

    # def _score_word(
    #     word_indices: Tuple[List[int], List[int]],
    #     word_rank: int,
    #     random_value: float,
    # ) -> float:
    #     return word_rank + random_value

    # words = trie.get_valid_words(
    #     [GridState.OPEN, 2, 4, GridState.OPEN, GridState.OPEN, 8],
    #     blocked=[False, False, False, False, False, False],
    #     target_index=0,
    # )
    # print(list(words))

    # words = trie.get_top_k_valid_words(
    #     [GridState.OPEN for _ in range(40)],
    #     ([0 for _ in range(40)], [i for i in range(40)]),
    #     target_index=20,
    #     scoring_function=_score_word,
    # )
    # print(words)

    # cProfile.run(
    #     """words = trie.get_top_k_valid_words(
    #     [GridState.OPEN for _ in range(40)],
    #     ([0 for _ in range(40)], [i for i in range(40)]),
    #     target_index=20,
    #     scoring_function=_score_word,
    # )"""
    # )

    # print(
    #     "time per",
    #     timeit.timeit(
    #         """words = trie.get_top_k_valid_words(
    #     [GridState.OPEN for _ in range(40)],
    #     ([0 for _ in range(40)], [i for i in range(40)]),
    #     target_index=35,
    #     scoring_function=_score_word,
    # )""",
    #         globals={"trie": trie, "GridState": GridState, "_score_word": _score_word},
    #         number=20,
    #     )
    #     / 20,
    # )
