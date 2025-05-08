from typing import List, Self, Tuple, Set

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
    Supports both forwards and backwards english letters.
    """

    def __init__(self):
        with open("word_lists/english.txt") as f:
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
            if node.children[char] is None:
                node.children[char] = _Node()
            node = node.children[char]
        node.path = word.copy()
        node.reverse = reverse

    def get_valid_words(self, slice: List[int]) -> List[List[int]] | None:
        nodes = [(0, self.root)]
        output = [set() for _ in slice]

        for current_index, entry in enumerate(slice):
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

            for offset, node in nodes:
                if node.path is not None:
                    # We have a terminal word at this node, update our valid values
                    path_length = len(node.path)
                    for idx in range(len(output)):
                        if offset <= idx and offset + path_length > idx:
                            output[idx].add((tuple(node.path), offset))

            # Since we could start a new word here, add the root node
            nodes.append((current_index + 1, self.root))

        return output

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
    import time
    import random

    import numpy as np

    # with open("word_lists/english.txt") as f:
    #     word_list = f.readlines()

    # for word in word_list:
    #     word = word.strip().lower()
    #     if not check(word):
    #         print(f"Didn't find {word}")

    trie = WordTrie()
    print(f"Created trie with depth {trie.root.depth()}")

    # words = trie.get_valid_words(
    #     convert_letters_to_matrix(np.array(["b", "o", "{", "y", "b"]))
    # )
    # print(words)
    # for word, offset in words[2]:
    #     a = -np.ones(5)
    #     for idx, value in enumerate(word):
    #         a[offset + idx] = value
    #     print(convert_matrix_to_letters(a))

    print(trie.coverage(convert_letters_to_matrix(np.array(["b", "o", "y", "{", "b"]))))
    print(trie.coverage(convert_letters_to_matrix(np.array(["b", "y", "d", "o", "b"]))))

    # while True:
    #     to_check = input("Does the trie have?:")
    #     print(check(to_check))

    # performance_test = []
    # N_EXAMPLES = 1_000_000

    # with open("word_lists/english.txt") as f:
    #     word_list = f.readlines()

    # word_list = [
    #     string_to_alphabet_positions(word.strip().lower()) for word in word_list
    # ]

    # for _ in range(N_EXAMPLES):
    #     word = random.choice(word_list)
    #     length = random.randint(1, len(word))
    #     performance_test.append(word[:length])

    # start_time = time.time()
    # for fragment in performance_test:
    #     trie.contains(fragment)
    # end_time = time.time()

    # print(
    #     f"Lookup took {(end_time - start_time) / (N_EXAMPLES / 1_000_000)}ms per fragment"
    # )
