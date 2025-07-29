use std::collections::HashSet;

use crate::consts::{OPEN, WORD_SEARCH_HEIGHT, WORD_SEARCH_SIZE, WORD_SEARCH_WIDTH};
use crate::grid::Grid;
use crate::trie::TrieNode;
use crate::util;

struct WordSearch {
    word_grid: Grid<char, WORD_SEARCH_SIZE, WORD_SEARCH_HEIGHT, WORD_SEARCH_WIDTH>,
    words: HashSet<String>,
}

impl WordSearch {
    pub fn new(
        word_grid: Grid<usize, WORD_SEARCH_SIZE, WORD_SEARCH_HEIGHT, WORD_SEARCH_WIDTH>,
        words: HashSet<String>,
    ) -> Result<Self, String> {
        let word_grid = word_grid.try_map(util::int_to_char)?;
        Ok(WordSearch {
            word_grid: word_grid,
            words: words,
        })
    }
}

fn score_word(
    coverage: Vec<bool>,
    open_slots: Vec<bool>,
    word_indices: (Vec<usize>, Vec<usize>),
    word_rank: f32,
    random_value: f32,
) -> f32 {
    0.0
}

fn grid_coverage(
    word_grid: Grid<usize, WORD_SEARCH_SIZE, WORD_SEARCH_HEIGHT, WORD_SEARCH_WIDTH>,
    trie: TrieNode,
) -> Grid<bool, WORD_SEARCH_SIZE, WORD_SEARCH_HEIGHT, WORD_SEARCH_WIDTH> {
    let mut coverage_grid =
        Grid::<bool, WORD_SEARCH_SIZE, WORD_SEARCH_HEIGHT, WORD_SEARCH_WIDTH>::new();

    coverage_grid
}

fn seed_grid(
    word_grid: Grid<usize, WORD_SEARCH_SIZE, WORD_SEARCH_HEIGHT, WORD_SEARCH_WIDTH>,
    density: f32,
) -> Grid<usize, WORD_SEARCH_SIZE, WORD_SEARCH_HEIGHT, WORD_SEARCH_WIDTH> {
    let unfilled_indices = word_grid
        .items()
        .filter_map(|((row, col), value)| {
            if *value == OPEN {
                Some((row, col))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let target_num_filled = (WORD_SEARCH_SIZE as f32 * density).round() as i32;
    let num_to_fill = target_num_filled - (WORD_SEARCH_SIZE as i32 - unfilled_indices.len() as i32);

    if num_to_fill < 0 {
        return word_grid;
    }

    let num_to_fill = num_to_fill as usize;
    let mut new_grid = word_grid.clone();

    // TODO: generate `num_to_fill` random integers in the range 0..unfilled_indices.len() without replacement
    // Then select those integers from unfilled_indices and give them random values

    word_grid
}

fn solve(
    word_grid: Grid<usize, WORD_SEARCH_SIZE, WORD_SEARCH_HEIGHT, WORD_SEARCH_WIDTH>,
    blocked: Grid<bool, WORD_SEARCH_SIZE, WORD_SEARCH_HEIGHT, WORD_SEARCH_WIDTH>,
    trie: TrieNode,
) -> Option<WordSearch> {
    let seeded_grid = seed_grid(word_grid, 0.05);
    let initial_coverage = grid_coverage(seeded_grid, trie);

    let grid = Grid::<usize, WORD_SEARCH_SIZE, WORD_SEARCH_HEIGHT, WORD_SEARCH_WIDTH>::new();
    let words: HashSet<String> = HashSet::new();
    WordSearch::new(grid, words).ok()
}
