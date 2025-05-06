from typing import List, Self

import string
from dataclasses import dataclass, field


from util import convert_matrix_to_letters, convert_letters_to_matrix


def string_to_alphabet_positions(s):
    return [ord(char) - ord("a") for char in s if char.isalpha()]


def default_letter_list_factory():
    return [None for _ in string.ascii_lowercase]


@dataclass(slots=True)
class _Node:
    children: List[Self | None] = field(default_factory=default_letter_list_factory)
    path: List[int] | None = None

    # def add(self, string: List[int]):
    #     node = self
    #     for char in string:
    #         if node.children[char] is None:
    #             node.children[char] = _Node()
    #         node = node.children[char]

    # def get(self, string: List[int]):
    #     node = self
    #     for char in string:
    #         node = node.children[char]
    #         if node is None:
    #             return None

    #     return node

    # def contains(self, string: List[int]) -> bool:
    #     node = self
    #     for char in string:
    #         node = node.children[char]
    #         if node is None:
    #             return False

    #     return True

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

    def get_valid(self, slice: List[int]) -> List[List[int]] | None:
        nodes = [(0, self.root)]
        output = [None if value != 26 else set() for value in slice]
        idxs_to_set = [idx for idx in range(len(output)) if slice[idx] == 26]

        for current_depth, entry in enumerate(slice):
            if entry == 26:
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

            depth_contains_terminating_node = False
            for offset, node in nodes:
                if node.path is not None:
                    # We have a terminal word at this node, update our valid values
                    path_length = len(node.path)
                    for idx in idxs_to_set:
                        if offset <= idx and idx - offset < path_length:
                            output[idx].add(node.path[idx - offset])

                        # if idx == 2 and node.path[idx - offset] == 1:
                        #     print(convert_matrix_to_letters(node.path))

                    depth_contains_terminating_node = True

            if depth_contains_terminating_node:
                nodes.append(
                    (current_depth + 1, self.root)
                )  # We could start a new word at the next idx

            if len(nodes) == 0:
                return None

        for idx in idxs_to_set:
            if len(output[idx]) == 0:
                return None

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

    print(
        trie.get_valid(convert_letters_to_matrix(np.array(["b", "o", "{", "y", "b"])))
    )

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
