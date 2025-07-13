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

        self.root = TrieNode()
        for word_placement, word in enumerate(filtered_words):
            word_rank = 1 - (word_placement / self._num_entries)
            word = string_to_alphabet_positions(word)
            self.add(word, word_rank)
            word.reverse()
            self.add(word, word_rank, reverse=True)

        self._depth = self.root.depth()

    def __len__(self) -> int:
        return self._num_entries

    def depth(self) -> int:
        return self._depth

    def word_limits(self) -> Tuple[int, int]:
        return self._word_limits

    def add(self, word: List[int], word_rank: float, reverse: bool = False):
        node = self.root
        for char in word:
            if char > 25:
                raise ValueError(
                    f"Recevied character '{convert_list_to_string([char])}' as part of word '{convert_list_to_string(word)}', which is unsupported!"
                )
            if node.children[char] is None:
                node.children[char] = TrieNode()
            node = node.children[char]
        if node.path is None or node.path.word_rank < word_rank:
            node.path = TriePath(word.copy(), reverse, word_rank)

    def get_valid_words(
        self, slice: List[int], blocked: List[bool], target_index: int
    ) -> Generator[Tuple[List[int], float, Tuple[int, int]], None, None]:
        max_children = len(self.root.children)
        slice_size = len(slice)
        minimum_word_length, maximum_word_length = self.word_limits()

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

        # If we are closer to the end of our valid word window, reverse it, so that we
        # don't have to check as many starting locations
        if target_index - starting_index > ending_index - target_index:
            # Reverse slice
            reversed_slice = slice.copy()
            if isinstance(reversed_slice, np.ndarray):
                reversed_slice = np.flip(reversed_slice)
            else:
                reversed_slice.reverse()
            # Reverse blocked
            reversed_blocked = blocked.copy()
            if isinstance(reversed_blocked, np.ndarray):
                reversed_blocked = np.flip(reversed_blocked)
            else:
                reversed_blocked.reverse()
            # Plug in the reversed inputs
            reversed_target = slice_size - 1 - target_index
            for word, value, (start, stop) in self.get_valid_words(
                slice=reversed_slice,
                blocked=reversed_blocked,
                target_index=reversed_target,
            ):
                word = word.copy()
                word.reverse()
                yield word, value, (slice_size - stop, slice_size - start)
            return

        # TODO: do not yield entries which include the blocked letter, but also don't include
        # entries which could include the blocked letter, in the future
        def _get_all_starting_at(starting_offset: int):
            tree_path = [
                None for _ in range(self.depth())
            ]  # Stack of (node, child_index)
            current_tree_index = 0
            current_node = self.root
            current_child_index = (
                0
                if slice[starting_offset] == GridState.OPEN
                else slice[starting_offset]
            )

            slice_index = starting_offset
            slice_entry = slice[slice_index]
            while True:
                # Check if we have exceeded the valid children of this node
                if (
                    current_child_index >= max_children
                    or slice_entry != GridState.OPEN
                    and slice_entry != current_child_index
                ):
                    if current_tree_index == 0:
                        return
                    current_tree_index -= 1
                    current_node, current_child_index = tree_path[current_tree_index]
                    tree_path[current_tree_index] = (
                        None  # TODO: hopefully remove this after debugging
                    )
                    current_child_index += 1
                    slice_index -= 1
                    slice_entry = slice[slice_index]
                    # Since we have updated our node, we want to restart the loop
                    continue

                # Check if the child at current_child_index exists and is valid
                if current_node.children[current_child_index] is not None and (
                    slice_entry == GridState.OPEN or slice_entry == current_child_index
                ):
                    next_node = current_node.children[current_child_index]

                    if next_node.path is not None:
                        if slice_index >= target_index:
                            yield next_node.path.word, next_node.path.word_rank, (
                                starting_offset,
                                slice_index + 1,
                            )

                    # If our next step will take us out of the slice boundaries,
                    # then we will not step forward, and instead continue to
                    # cycle through siblings of this node.
                    next_slice_index = slice_index + 1
                    if (
                        next_slice_index >= ending_index
                        or next_slice_index <= starting_index - 1
                    ):
                        current_child_index += 1
                    else:
                        # Append to our history
                        tree_path[current_tree_index] = (
                            current_node,
                            current_child_index,
                        )
                        current_tree_index += 1

                        # Get the next node
                        current_node = current_node.children[current_child_index]

                        # Get the next states, and figure out where to start our child index from
                        slice_index = next_slice_index
                        slice_entry = slice[slice_index]
                        current_child_index = (
                            0 if slice_entry == GridState.OPEN else slice_entry
                        )

                    # Since we have updated our node, we want to restart the loop
                    continue

                current_child_index += 1

        for starting_offset in range(
            starting_index,
            min(target_index, ending_index - minimum_word_length) + 1,
        ):
            yield from _get_all_starting_at(starting_offset)

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
        current_nodes = [(0, self.root)]

        for current_index, entry in enumerate(slice):
            if entry != GridState.OPEN:
                # Traverse each of the current valid nodes
                current_nodes = [
                    (node_start, node.children[entry])
                    for node_start, node in current_nodes
                    if node.children[entry] is not None
                ]
            else:
                # Empty cells stop all current nodes
                current_nodes = []

            current_nodes.append((current_index + 1, self.root))

            # Check each of our current tries to see if we have a word
            # We are going in reverse order so that words that start earlier are
            # provided afterwards. Words which end later are also provided
            # afterwards. Both of these properties are useful to the `words`
            # function
            for start_position, node in reversed(current_nodes):
                if node.path is not None:
                    # We have found a word
                    yield start_position, node

    def coverage(self, slice: List[int]) -> np.ndarray:
        coverage = np.zeros(len(slice), dtype=bool)

        for start, node in self._words_from_slice(slice):
            coverage[start : start + len(node.path.word)] = True

        return coverage

    def words(self, slice: List[int], trim: bool = True) -> Set[str]:
        # Get all words in the given slice
        words = set()

        for _, node in self._words_from_slice(slice):
            new_word = node.path.word.copy()
            if node.path.is_reverse:
                new_word.reverse()
            new_word = convert_list_to_string(new_word)

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
