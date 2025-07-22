use crate::consts::OPEN;
use crate::trie::consts::{MIN_WORD, NUM_LETTERS};
use crate::trie::node::TrieNode;
use crate::trie::types::WordInfo;
use crate::util;

pub struct TargetedValidWordIterator<'a> {
    ending_index: usize,
    current_starting_index: usize,
    target_index: usize,

    previous_nodes: Vec<(&'a TrieNode, usize)>,
    current_node: &'a TrieNode,
    current_node_child: usize,

    slice: &'a [usize],
    slice_position: usize,
}

impl<'a> TargetedValidWordIterator<'a> {
    pub fn new(
        node: &'a TrieNode,
        slice: &'a [usize],
        target_index: usize,
        starting_index: usize,
        ending_index: usize,
    ) -> Self {
        let current_child = if slice[starting_index] == OPEN {
            0
        } else {
            slice[starting_index]
        };

        TargetedValidWordIterator {
            ending_index: ending_index,
            current_starting_index: starting_index,
            target_index: target_index,
            previous_nodes: Vec::with_capacity(target_index - starting_index + 1),
            current_node: node,
            current_node_child: current_child,
            slice: slice,
            slice_position: starting_index,
        }
    }
}

// TODO: Only need to keep track of places that can branch, for the previous nodes
impl<'a> Iterator for TargetedValidWordIterator<'a> {
    type Item = (&'a WordInfo, (usize, usize));

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Check if we need to walk up the tree
            if self.current_node_child >= NUM_LETTERS
                || (self.slice[self.slice_position] != OPEN
                    && self.slice[self.slice_position] != self.current_node_child)
            {
                // Increment the current starting index, if necessary
                if self.previous_nodes.len() == 0 {
                    // Reset with the next starting index
                    self.current_starting_index += 1;
                    if self.current_starting_index > self.target_index {
                        // Stop, we have found all valid words
                        return None;
                    }

                    // previous_nodes is already empty, so the current node will already be correct
                    self.slice_position = self.current_starting_index;
                    self.current_node_child = if self.slice[self.slice_position] == OPEN {
                        0
                    } else {
                        self.slice[self.slice_position]
                    };

                    // Restart the loop, so that we don't try to do the tree walk stuff
                    continue;
                }

                // Walk up the tree and step to next child
                (self.current_node, self.current_node_child) = self.previous_nodes.pop().unwrap();
                self.current_node_child += 1;

                // Step back in the slice
                self.slice_position -= 1;

                // Restart the loop
                continue;
            }

            // Check if there is a valid child at the current node child
            if (self.slice[self.slice_position] == OPEN
                || self.slice[self.slice_position] == self.current_node_child)
            {
                if let Some(next_node) = &self.current_node.children[self.current_node_child] {
                    let word_bounds = (self.current_starting_index, self.slice_position + 1);

                    // First, update our internal state, for the next time this function is called
                    if self.slice_position + 1 >= self.ending_index {
                        // Since we cannot step forward, we must increment the child
                        self.current_node_child += 1;
                    } else {
                        // Step forward
                        self.previous_nodes
                            .push((self.current_node, self.current_node_child));
                        self.current_node = next_node;

                        self.slice_position += 1;
                        self.current_node_child = if self.slice[self.slice_position] == OPEN {
                            0
                        } else {
                            self.slice[self.slice_position]
                        };
                    }

                    // Now, see if we have something to return
                    if word_bounds.1 > self.target_index {
                        if let Some(word_info) = &next_node.word_info {
                            return Some((word_info, word_bounds));
                        }
                    }

                    continue;
                }
            }

            // No valid child, check the next one
            self.current_node_child += 1;
        }
    }
}

pub struct WordIterator<'a> {
    ending_index: usize,
    current_starting_index: usize,

    root_node: &'a TrieNode,
    current_node: &'a TrieNode,

    slice: &'a [usize],
    slice_position: usize,
}

impl<'a> WordIterator<'a> {
    pub fn new(
        node: &'a TrieNode,
        slice: &'a [usize],
        starting_index: usize,
        ending_index: usize,
    ) -> Self {
        WordIterator {
            ending_index: ending_index,
            current_starting_index: starting_index,
            root_node: node,
            current_node: node,
            slice: slice,
            slice_position: starting_index,
        }
    }
}

impl<'a> Iterator for WordIterator<'a> {
    type Item = (&'a WordInfo, (usize, usize));

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Check if we need to restart
            if self.slice_position >= self.ending_index
                || self.slice[self.slice_position] == OPEN
                || self.current_node.children[self.slice[self.slice_position]].is_none()
            {
                self.current_starting_index += 1;
                if self.current_starting_index > self.ending_index - MIN_WORD {
                    // Stop, we have found all valid words
                    return None;
                }

                self.current_node = self.root_node;
                self.slice_position = self.current_starting_index;

                continue; // Restart the loop
            }

            let next_node = unsafe {
                self.current_node.children[self.slice[self.slice_position]]
                    .as_ref()
                    .unwrap_unchecked()
            };
            let word_bounds = (self.current_starting_index, self.slice_position + 1);

            // First, update our internal state, for the next time this function is called
            self.current_node = next_node;
            self.slice_position += 1;

            // Now, see if we have something to return
            if let Some(word_info) = &next_node.word_info {
                return Some((word_info, word_bounds));
            }
        }
    }
}
