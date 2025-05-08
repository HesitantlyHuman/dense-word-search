from typing import List, Self

import string
from dataclasses import dataclass, field

from datatypes import GridState

from util import convert_matrix_to_letters, convert_letters_to_matrix


def string_to_alphabet_positions(s):
    return [ord(char) - ord("a") for char in s if char.isalpha()]


def default_letter_list_factory():
    return [None for _ in string.ascii_lowercase]


@dataclass(slots=True)
class _Node:
    children: List[Self | None] = field(default_factory=default_letter_list_factory)
    path: List[int] | None = None

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
            self.add(word)

        self._depth = self.root.depth()

    def depth(self) -> int:
        return self._depth

    def add(self, word: List[int]):
        node = self.root
        for char in word:
            if node.children[char] is None:
                node.children[char] = _Node()
            node = node.children[char]
        node.path = word.copy()

    def get_valid_words(self, slice: List[int]) -> List[List[int]] | None:
        nodes = [(0, self.root)]
        output = [None if value != GridState.OPEN else set() for value in slice]
        idxs_to_set = [
            idx for idx in range(len(output)) if slice[idx] == GridState.OPEN
        ]

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
                    for idx in idxs_to_set:
                        if offset <= idx and offset + path_length > idx:
                            output[idx].add((tuple(node.path), idx - offset - 1))

            # Since we could start a new word here, add the root node
            nodes.append((current_index, self.root))

        return output


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

    words = trie.get_valid_words(
        convert_letters_to_matrix(np.array(["b", "o", "{", "y", "b"]))
    )
    print(words)
    for word, offset in words[2]:
        a = -np.ones(5)
        for idx, value in enumerate(word):
            a[offset + idx] = value
        print(convert_matrix_to_letters(a))

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
