use std::collections::HashSet;

use crate::consts::OPEN;
use crate::util::string_to_ints;

struct WordInfo {
    path: Vec<u8>,
    word: String,
    word_rank: f32,
}

const EMPTY: Option<Box<TrieNode>> = None;

const MIN_WORD: usize = 3;
const MAX_WORD: usize = 10;

pub struct TrieNode {
    children: [Option<Box<TrieNode>>; 26],
    word_info: Option<WordInfo>,
}

impl TrieNode {
    fn new() -> Self {
        TrieNode {
            children: [EMPTY; 26],
            word_info: None,
        }
    }

    pub fn build(minimum_word_length: usize, maximum_word_length: usize) -> Result<Self, String> {
        let mut root = TrieNode::new();
        // Load and filter words
        let words = include_str!("coca.txt");
        let filtered_words: Vec<&str> = words
            .split("\n")
            .map(|chunk| chunk.trim())
            .filter(|word| (word.len() > minimum_word_length) & (word.len() < maximum_word_length))
            .collect();
        // Add all of our words
        let num_words = filtered_words.len();
        for (rank_num, word) in filtered_words.iter().enumerate() {
            let word_rank = 1.0 - (rank_num as f32 / (num_words - 1) as f32);
            root.add(word.to_string(), word_rank)?;
        }
        Ok(root)
    }

    fn _add_one(&mut self, word_list: &Vec<u8>, word: &String, word_rank: f32) {
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

    fn get_top_k_valid_words(
        self,
        slice: Vec<u8>,
        blocked: Vec<bool>,
        indices: (Vec<usize>, Vec<usize>),
        target_index: usize,
        scoring_function: fn((Vec<usize>, Vec<usize>), f32, f32) -> f32,
        k: u16,
        max_words_to_check: u16,
    ) -> Vec<(f32, Vec<u8>, (Vec<usize>, Vec<usize>))> {
        Vec::new()
    }

    fn slice_words(self, slice: Vec<u8>) -> HashSet<String> {
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

    fn slice_coverage(self, slice: Vec<u8>) -> Vec<bool> {
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
