use std::collections::HashSet;

use crate::consts::OPEN;
use crate::util::string_to_ints;

struct WordInfo {
    path: Vec<usize>,
    word: String,
    word_rank: f32,
}

const EMPTY: Option<Box<TrieNode>> = None;
const NUM_LETTERS: usize = 26;

const MIN_WORD: usize = 3;
const MAX_WORD: usize = 10;

pub struct ValidWordIterator<'a> {
    starting_index: usize,
    ending_index: usize,
    current_starting_index: usize,

    previous_nodes: Vec<(&'a TrieNode, usize)>,
    current_node: &'a TrieNode,
    current_node_child: usize,

    slice: &'a [usize],
    slice_position: usize,
}

impl<'a> ValidWordIterator<'a> {
    fn new(
        node: &'a TrieNode,
        slice: &'a [usize],
        starting_index: usize,
        ending_index: usize,
    ) -> Self {
        let current_child = if slice[starting_index] == OPEN {
            0
        } else {
            slice[starting_index]
        };

        ValidWordIterator {
            starting_index: starting_index,
            ending_index: ending_index,
            current_starting_index: starting_index,
            previous_nodes: Vec::with_capacity(NUM_LETTERS),
            current_node: node,
            current_node_child: current_child,
            slice: slice,
            slice_position: starting_index,
        }
    }
}

impl<'a> Iterator for ValidWordIterator<'a> {
    type Item = (&'a [usize], f32, (usize, usize));

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
                    if self.current_starting_index == self.ending_index {
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

                    // Restart the loop, so that we don't try to to the tree walk stuff
                    continue;
                }
            }
        }
    }
}

pub struct TrieNode {
    children: [Option<Box<TrieNode>>; NUM_LETTERS],
    word_info: Option<WordInfo>,
}

impl TrieNode {
    fn new() -> Self {
        TrieNode {
            children: [EMPTY; NUM_LETTERS],
            word_info: None,
        }
    }

    pub fn build() -> Result<Self, String> {
        let mut root = TrieNode::new();
        // Load and filter words
        let words = include_str!("coca.txt");
        let filtered_words: Vec<&str> = words
            .split("\n")
            .map(|chunk| chunk.trim())
            .filter(|word| (word.len() > MIN_WORD) & (word.len() < MAX_WORD))
            .collect();
        // Add all of our words
        let num_words = filtered_words.len();
        for (rank_num, word) in filtered_words.iter().enumerate() {
            let word_rank = 1.0 - (rank_num as f32 / (num_words - 1) as f32);
            root.add(word.to_string(), word_rank)?;
        }
        Ok(root)
    }

    fn _add_one(&mut self, word_list: &Vec<usize>, word: &String, word_rank: f32) {
        let mut current_node = self;

        for character in word_list.iter() {
            let idx: usize = *character as usize;
            if current_node.children[idx].is_none() {
                current_node.children[idx] = Some(Box::new(TrieNode::new()));
            }
            current_node = current_node.children[idx].as_mut().unwrap();
        }
        current_node.word_info = Some(WordInfo {
            path: word_list.to_vec(),
            word: word.to_string(),
            word_rank: word_rank,
        })
    }

    fn add(&mut self, word: String, word_rank: f32) -> Result<(), String> {
        // Add the word in both directions
        let mut word_ints = string_to_ints(&word)?;
        self._add_one(&word_ints, &word, word_rank);
        word_ints.reverse();
        self._add_one(&word_ints, &word, word_rank);
        Ok(())
    }

    pub fn depth(self) -> u8 {
        let mut depth: u8 = 1;
        for potential_child in self.children {
            match potential_child {
                Some(child) => depth = depth.max(child.depth() + 1),
                None => {}
            }
        }
        depth
    }

    pub fn get_valid_words<'a>(
        &self,
        slice: &'a [usize],
        blocked: &'a [bool],
        target_index: usize,
    ) -> ValidWordIterator {
        let slice_length = slice.len();

        let mut previous_blocked = None;
        let mut next_blocked = None;
        for (idx, is_blocked) in blocked.iter().enumerate() {
            if *is_blocked {
                if idx < target_index {
                    previous_blocked = Some(idx);
                } else if next_blocked.is_none() {
                    next_blocked = Some(idx);
                }
            }
        }

        let start = previous_blocked
            .map_or(0, |i| i + 1)
            .max(target_index.saturating_sub(MAX_WORD));
        let end = next_blocked
            .unwrap_or(slice_length)
            .min(target_index + MAX_WORD);

        // TODO: Add reversing logic

        println!("{}, {}", start, end);

        (start..=target_index.min(end - MAX_WORD)).flat_map(|starting_index| {
            let mut previous_nodes = Vec::with_capacity(MAX_WORD);
            let mut current_node = self;
            let mut current_child = if slice[starting_index] == OPEN {
                0
            } else {
                slice[starting_index]
            };

            let mut slice_position = starting_index;
            let mut slice_entry = slice[slice_position];
            loop {
                // Check if we have exceeded the valid children of our current node
                if current_child >= NUM_LETTERS
                    || (slice_entry != OPEN && slice_entry != current_child)
                {}
            }
        });

        ValidWordIterator::new(self, slice, 0, 0)
    }

    pub fn get_top_k_valid_words<'a>(
        &self,
        slice: &'a [usize],
        blocked: &'a [bool],
        indices: (&'a [usize], &'a [usize]),
        target_index: usize,
        scoring_function: fn((&[usize], &[usize]), f32, f32) -> f32,
        k: u16,
        max_words_to_check: u16,
    ) -> Vec<(f32, Vec<u8>, (&'a [usize], &'a [usize]))> {
        Vec::new()
    }

    // TODO: change slice input
    fn slice_words(self, slice: Vec<usize>) -> HashSet<String> {
        let mut current_nodes = vec![(0, &self)];
        let mut words: HashSet<String> = HashSet::new();

        for (index, slice_entry) in slice.iter().enumerate() {
            if *slice_entry < OPEN {
                // If we have a letter in this spot, then we need to get a new set of nodes
                let mut next_nodes = Vec::with_capacity(current_nodes.len());

                for node in &mut current_nodes {
                    if let Some(child) = node.1.children[*slice_entry as usize].as_deref() {
                        next_nodes.push((node.0, child));
                    }
                }

                current_nodes = next_nodes;
            } else {
                current_nodes = vec![];
            }

            current_nodes.push((index + 1, &self));

            // Now see if we have any words
            for (_, node) in current_nodes.iter().rev() {
                if let Some(word_info) = &node.word_info {
                    let word_to_add = word_info.word.to_string();

                    // First, remove from our set any words that are contained in this word
                    words.retain(|already_collected| !word_to_add.contains(already_collected));

                    // Add word to our set
                    words.insert(word_to_add);
                }
            }
        }

        words
    }

    // TODO: change slice input
    fn slice_coverage(self, slice: Vec<usize>) -> Vec<bool> {
        let mut current_nodes = vec![(0, &self)];
        let mut coverage: Vec<bool> = Vec::with_capacity(slice.len());

        for (index, slice_entry) in slice.iter().enumerate() {
            if *slice_entry < OPEN {
                // If we have a letter in this spot, then we need to get a new set of nodes
                let mut next_nodes = Vec::with_capacity(current_nodes.len());

                for node in &mut current_nodes {
                    if let Some(child) = node.1.children[*slice_entry as usize].as_deref() {
                        next_nodes.push((node.0, child));
                    }
                }

                current_nodes = next_nodes;
            } else {
                current_nodes = vec![];
            }

            current_nodes.push((index + 1, &self));

            // Now see if we have any words
            for (word_start, node) in current_nodes.iter().rev() {
                if let Some(word_info) = &node.word_info {
                    coverage[*word_start..*word_start + word_info.path.len()].fill(true);
                }
            }
        }

        coverage
    }
}
