use crate::consts::OPEN;
use crate::trie::consts::{MIN_WORD, NUM_LETTERS};
use crate::trie::node::TrieNode;
use crate::trie::types::WordInfo;
use crate::util;

pub struct TargetedValidWordIterator<'a> {
    ending_index: usize,
    current_starting_index: usize,
    target_index: usize,

    previous_nodes: Vec<(&'a TrieNode, usize, usize)>,
    root_node: &'a TrieNode,
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
            previous_nodes: Vec::with_capacity(ending_index - starting_index),
            root_node: node,
            current_node: node,
            current_node_child: current_child,
            slice: slice,
            slice_position: starting_index,
        }
    }
}

impl<'a> Iterator for TargetedValidWordIterator<'a> {
    type Item = (&'a WordInfo, (usize, usize));

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Check if we need to backtrack
            if self.current_node_child >= NUM_LETTERS
                || self.slice_position >= self.ending_index
                || (self.slice[self.slice_position] != OPEN
                    && self.current_node.children[self.slice[self.slice_position]].is_none())
            {
                match self.previous_nodes.pop() {
                    Some(value) => {
                        // Step back to previous open spot
                        (
                            self.current_node,
                            self.current_node_child,
                            self.slice_position,
                        ) = value;
                        self.current_node_child += 1;
                    }
                    None => {
                        // Restart from next starting index
                        self.current_starting_index += 1;
                        if self.current_starting_index > self.target_index
                            || self.current_starting_index > self.ending_index - MIN_WORD
                        {
                            // Stop, we have found all valid words
                            return None;
                        }

                        self.current_node = self.root_node;
                        self.slice_position = self.current_starting_index;
                        self.previous_nodes.clear();
                        self.current_node_child = match self.slice[self.slice_position] {
                            OPEN => 0,
                            _ => self.slice[self.slice_position],
                        };
                    }
                }

                continue; // Restart the loop
            }

            // Check if we have a valid node, or need to check the next child
            if self.slice[self.slice_position] == OPEN
                && self.current_node.children[self.current_node_child].is_none()
            {
                // Check the next child
                self.current_node_child += 1;
                continue;
            }

            // Since we now know that we have a valid node, step forward
            let next_node = match self.slice[self.slice_position] {
                OPEN => {
                    self.previous_nodes.push((
                        self.current_node,
                        self.current_node_child,
                        self.slice_position,
                    ));
                    unsafe {
                        self.current_node.children[self.current_node_child]
                            .as_ref()
                            .unwrap_unchecked()
                    }
                }
                _ => unsafe {
                    self.current_node.children[self.slice[self.slice_position]]
                        .as_ref()
                        .unwrap_unchecked()
                },
            };
            let word_bounds = (self.current_starting_index, self.slice_position + 1);

            // First, update our internal state, for the next time this function is called
            self.current_node = next_node;
            self.slice_position += 1;
            self.current_node_child = if self.slice_position < self.ending_index {
                match self.slice[self.slice_position] {
                    OPEN => 0,
                    _ => self.slice[self.slice_position],
                }
            } else {
                NUM_LETTERS
            };

            // Now, see if we have something to return
            if self.slice_position > self.target_index {
                if let Some(word_info) = &next_node.word_info {
                    return Some((word_info, word_bounds));
                }
            }
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
