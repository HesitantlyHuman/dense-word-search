use std::collections::HashSet;

use crate::trie::consts::{MAX_WORD, MIN_WORD, NUM_LETTERS};
use crate::trie::iters::{TargetedValidWordIterator, WordIterator};
use crate::trie::types::WordInfo;
use crate::util::string_to_ints;

const EMPTY: Option<Box<TrieNode>> = None;

pub struct TrieNode {
    pub children: [Option<Box<TrieNode>>; NUM_LETTERS],
    pub word_info: Option<WordInfo>,
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
            .filter(|word| (word.len() >= MIN_WORD) & (word.len() <= MAX_WORD))
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
        &'a self,
        slice: &'a [usize],
        blocked: &'a [bool],
        target_index: usize,
    ) -> impl Iterator<Item = (&'a WordInfo, (usize, usize))> + 'a {
        debug_assert!(slice.len() == blocked.len());
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

        TargetedValidWordIterator::new(self, slice, target_index, start, end)
    }

    pub fn get_words<'a>(
        &'a self,
        slice: &'a [usize],
    ) -> impl Iterator<Item = (&'a WordInfo, (usize, usize))> + 'a {
        WordIterator::new(self, slice, 0, slice.len())
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

    pub fn slice_words<'a>(&'a self, slice: &'a [usize]) -> HashSet<String> {
        let mut final_set = HashSet::new();

        let mut current_start = 0;
        let mut current_stop = 0;
        let mut current_word: Option<&String> = None;

        for (word_info, (start, stop)) in self.get_words(slice) {
            if start != current_start {
                // We have moved forward, the current word (longest from
                // previous start) must be included in the final set
                if let Some(word) = current_word {
                    final_set.insert(word.clone());
                }

                current_word = None;
                current_start = start;
            }

            if stop > current_stop {
                current_word = Some(&word_info.word);
                current_stop = stop;
            }
        }

        if let Some(word) = current_word {
            final_set.insert(word.clone());
        }

        final_set
    }

    // TODO: Maybe a different return type is more efficient?
    pub fn slice_coverage<'a>(self, slice: &'a [usize]) -> Vec<bool> {
        let mut coverage: Vec<bool> = vec![false; slice.len()];

        for (_, (start, stop)) in self.get_words(slice) {
            coverage[start..stop].fill(true);
        }

        coverage
    }
}
