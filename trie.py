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
)


def string_to_alphabet_positions(s):
    return [ord(char) - ord("a") for char in s if char.isalpha()]


def default_letter_list_factory():
    return [None for _ in string.ascii_lowercase]


@dataclass(slots=True)
class _Node:
    children: List[Self | None] = field(default_factory=default_letter_list_factory)
    # TODO: maybe merge these into one object
    path: List[int] | None = None
    reverse: bool | None = None

    def depth(self) -> int:
        child_depths = [child.depth() for child in self.children if child is not None]
        if len(child_depths) == 0:
            return 0
        return max(child_depths) + 1


class WordTrie:
    """
    Provides functionality for getting valid letter states based on a slice.
    Supports both forwards and backwards embeddings of the provided word list.
    """

    def __init__(self):
        with open("word_lists/coca.txt") as f:
            word_list = f.readlines()

        self.root = _Node()
        for word in word_list:
            word = word.strip().lower()
            word = string_to_alphabet_positions(word)
            self.add(word)
            word.reverse()
            self.add(word, reverse=True)

        self._depth = self.root.depth()

    def depth(self) -> int:
        return self._depth

    def add(self, word: List[int], reverse: bool = False):
        node = self.root
        for char in word:
            if char > 25:
                raise ValueError(
                    f"Recevied character '{convert_list_to_string([char])}' as part of word '{convert_list_to_string(word)}', which is unsupported!"
                )
            if node.children[char] is None:
                node.children[char] = _Node()
            node = node.children[char]
        node.path = word.copy()
        node.reverse = reverse

    # def get_valid_words(
    #     self, slice: List[int], target_index: int
    # ) -> Generator[Tuple[List[int], Tuple[int, int]], None, None]:
    #     starting_index = max(0, target_index - self.depth())
    #     ending_index = min(len(slice), target_index + self.depth())
    #     nodes = [(0, self.root)]

    #     for current_index in range(starting_index, ending_index):
    #         entry = slice[current_index]
    #         if not nodes:
    #             return

    #         if entry == GridState.OPEN:
    #             new_nodes = []
    #             for offset, node in nodes:
    #                 new_nodes.extend(
    #                     [
    #                         (offset, child)
    #                         for child in node.children
    #                         if child is not None
    #                     ]
    #                 )
    #             nodes = new_nodes
    #         else:
    #             nodes = [
    #                 (offset, node.children[entry])
    #                 for offset, node in nodes
    #                 if node.children[entry] is not None
    #             ]

    #         # We could finish a word here
    #         if current_index >= target_index:
    #             for offset, node in nodes:
    #                 if node.path is not None:
    #                     yield node.path, (offset, current_index)

    #         # Since we could start a new word here, add the root node
    #         if current_index < target_index:
    #             nodes.append((current_index + 1, self.root))

    def get_valid_words(
        self, slice: List[int], target_index: int
    ) -> Generator[Tuple[List[int], Tuple[int, int]], None, None]:
        starting_index = max(0, target_index - self.depth())
        ending_index = min(len(slice), target_index + self.depth())
        max_children = len(self.root.children)

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
                # Check if we have reached the end of our slice
                if slice_index >= ending_index or current_child_index >= max_children:
                    if current_tree_index == 0:
                        return
                    current_tree_index -= 1
                    current_node, current_child_index = tree_path[current_tree_index]
                    current_child_index += 1
                    slice_index -= 1
                    slice_entry = slice[slice_index]
                    # Since we have updated our node, we want to restart the loop
                    continue

                # Check if the child at current_child_index exists and is valid
                if current_node.children[current_child_index] is not None and (
                    slice_entry == GridState.OPEN or slice_entry == current_child_index
                ):
                    # If it does, we will enter
                    tree_path[current_tree_index] = (current_node, current_child_index)
                    current_tree_index += 1
                    current_node = current_node.children[current_child_index]
                    current_child_index = (
                        0 if slice_entry == GridState.OPEN else slice_entry
                    )

                    if current_node.path is not None and slice_index >= target_index:
                        yield current_node.path, (starting_offset, slice_index + 1)

                    if slice_index + 1 >= ending_index:
                        current_tree_index -= 1
                        current_node, current_child_index = tree_path[
                            current_tree_index
                        ]
                        current_child_index += 1
                    else:
                        slice_index += 1
                        slice_entry = slice[slice_index]

                    # Since we have updated our node, we want to restart the loop
                    continue

                current_child_index += 1

        # TODO: if our target index is at the end, we should start from there
        # because we will only have to check 1 offset, instead of like 20

        for starting_offset in range(
            starting_index, min(target_index + 1, ending_index + 1 - 3)
        ):
            yield from _get_all_starting_at(starting_offset)

    def get_top_k_valid_words(
        self,
        slice: List[int],
        indices: Tuple[List[int], List[int]],
        target_index: str,
        scoring_function: Callable[
            [List[int], Tuple[List[int], List[int]], int], float
        ] = None,
        k: int = 10,
        max_check: int = 1_000,
    ) -> List[List[int]] | None:
        k_largest = []

        idx = 0
        for word, (start, stop) in self.get_valid_words(slice, target_index):
            word_indices = (
                indices[0][start:stop],
                indices[1][start:stop],
            )
            score = scoring_function(
                word, word_indices, 0
            )  # TODO: add support for word rank

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

    def coverage(self, slice: List[int]) -> Tuple[np.ndarray, Set[str]]:
        coverage = np.zeros(len(slice), dtype=bool)
        words = set()
        current_nodes = [(0, self.root)]

        for current_index, entry in enumerate(slice):
            if entry != GridState.OPEN:
                current_nodes = [
                    (node_start, node.children[entry])
                    for node_start, node in current_nodes
                    if node.children[entry] is not None
                ]

            current_nodes.append((current_index + 1, self.root))

            # Now, update coverage if we have a word
            for node_start, node in current_nodes:
                if node.path is not None:
                    coverage[node_start : current_index + 1] = True
                    path = node.path.copy()
                    if node.reverse:
                        path.reverse()
                    word = convert_list_to_string(path)
                    words.add(word)

        return coverage, words


if __name__ == "__main__":
    import cProfile
    import timeit

    trie = WordTrie()
    print(f"Created trie with depth {trie.root.depth()}")

    def _score_word(
        word: List[int],
        word_indices: Tuple[List[int], List[int]],
        word_rank: int,
    ) -> float:
        word_length = len(word)
        return word_length

    words = trie.get_valid_words([GridState.OPEN for _ in range(5)], target_index=0)
    list(words)

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
    #     target_index=20,
    #     scoring_function=_score_word,
    # )""",
    #         globals={"trie": trie, "GridState": GridState, "_score_word": _score_word},
    #         number=15,
    #     ),
    # )
