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

    def get_valid_words(
        self, slice: List[int], target_index: int
    ) -> Generator[Tuple[List[int], Tuple[int, int]], None, None]:
        starting_index = max(0, target_index - self.depth())
        ending_index = min(len(slice), target_index + self.depth())
        nodes = [(0, self.root)]

        for current_index in range(starting_index, ending_index):
            entry = slice[current_index]
            if len(nodes) == 0:
                break

            if entry == GridState.OPEN:
                new_nodes = []
                for offset, node in nodes:
                    new_nodes.extend(
                        [
                            (offset, child)
                            for child in node.children
                            if child is not None
                        ]
                    )
                nodes = new_nodes
            else:
                nodes = [
                    (offset, node.children[entry])
                    for offset, node in nodes
                    if node.children[entry] is not None
                ]

            # We could finish a word here
            if current_index >= target_index:
                for offset, node in nodes:
                    if node.path is not None:
                        yield node.path, (offset, current_index)

            # Since we could start a new word here, add the root node
            if current_index < target_index:
                nodes.append((current_index + 1, self.root))

    def get_top_k_valid_words(
        self,
        slice: List[int],
        indices: Tuple[List[int], List[int]],
        target_index: str,
        scoring_function: Callable[
            [List[int], Tuple[List[int], List[int]], int], float
        ] = None,
        k: int = 100,
    ) -> List[List[int]] | None:
        k_largest = []

        for word, (start, stop) in self.get_valid_words(slice, target_index):
            word_indices = (
                indices[0][start:stop],
                indices[1][start:stop],
            )
            score = scoring_function(
                word, word_indices, 0
            )  # TODO: add support for word rank

            if len(k_largest) < k:
                heapq.heappush(k_largest, (score, word, word_indices))
            elif score > k_largest[0][0]:
                try:
                    heapq.heappushpop(
                        k_largest, (score, word, word_indices)
                    )  # TODO: causes errors when we have the same word twice at different locations, maybe just add an index, and then filter it out?
                except ValueError as e:
                    print(score, word, word_indices)
                    raise e

        return k_largest  # If we want to sort later, we can

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
    trie = WordTrie()
    print(f"Created trie with depth {trie.root.depth()}")

    valid_words = trie.get_valid_words([GridState.OPEN for _ in range(20)], 10)
    print(len(valid_words))
