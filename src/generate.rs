use std::collections::HashSet;

use crate::consts::{OPEN, WORD_SEARCH_HEIGHT, WORD_SEARCH_SIZE, WORD_SEARCH_WIDTH};
use crate::grid::Grid;
use crate::trie::TrieNode;
use crate::util;

pub struct WordSearch<const N: usize, const ROWS: usize, const COLS: usize> {
    word_grid: Grid<char, N, ROWS, COLS>,
    words: HashSet<String>,
}

impl<const N: usize, const ROWS: usize, const COLS: usize> WordSearch<N, ROWS, COLS> {
    pub fn new(
        word_grid: Grid<usize, N, ROWS, COLS>,
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

fn grid_coverage<const N: usize, const ROWS: usize, const COLS: usize>(
    word_grid: &Grid<usize, N, ROWS, COLS>,
    trie: &TrieNode,
) -> Grid<bool, N, ROWS, COLS> {
    let mut coverage_grid = Grid::<bool, N, ROWS, COLS>::new();

    for slice in Grid::<usize, N, ROWS, COLS>::slices() {
        let (rows, cols) = slice;
        let data = word_grid
            .get_bulk(rows, cols)
            .copied()
            .collect::<Vec<usize>>();
        let slice_coverage = trie.slice_coverage(&data);
        let new_coverage = coverage_grid
            .get_bulk(rows, cols)
            .zip(&slice_coverage)
            .map(|(current, next)| *current || *next)
            .collect::<Vec<bool>>();
        coverage_grid.set_bulk(rows, cols, &new_coverage);
    }

    coverage_grid
}

fn grid_words<const N: usize, const ROWS: usize, const COLS: usize>(
    word_grid: &Grid<usize, N, ROWS, COLS>,
    trie: &TrieNode,
) -> HashSet<String> {
    let mut words = HashSet::new();

    for slice in Grid::<usize, N, ROWS, COLS>::slices() {
        let (rows, cols) = slice;
        let data = word_grid
            .get_bulk(rows, cols)
            .copied()
            .collect::<Vec<usize>>();
        let slice_words = trie.slice_words(&data);
        words.extend(slice_words);
    }

    words
}

fn fill_grid_backtracking<const N: usize, const ROWS: usize, const COLS: usize>(
    word_grid: &Grid<usize, N, ROWS, COLS>,
    blocked: &Grid<bool, N, ROWS, COLS>,
    trie: &TrieNode,
) -> Option<Grid<usize, N, ROWS, COLS>> {
    None
}

fn seed_grid<const N: usize, const ROWS: usize, const COLS: usize>(
    word_grid: Grid<usize, N, ROWS, COLS>,
    density: f32,
) -> Grid<usize, N, ROWS, COLS> {
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
    let target_num_filled = (N as f32 * density).round() as i32;
    let num_to_fill = target_num_filled - (N as i32 - unfilled_indices.len() as i32);

    if num_to_fill < 0 {
        return word_grid;
    }

    let num_to_fill = num_to_fill as usize;
    let mut new_grid = word_grid.clone();

    // TODO: generate `num_to_fill` random integers in the range 0..unfilled_indices.len() without replacement
    // Then select those integers from unfilled_indices and give them random values

    word_grid
}

pub fn solve<const N: usize, const ROWS: usize, const COLS: usize>(
    word_grid: Grid<usize, N, ROWS, COLS>,
    blocked: Grid<bool, N, ROWS, COLS>,
    trie: TrieNode,
) -> Option<WordSearch<N, ROWS, COLS>> {
    let seeded_grid = seed_grid(word_grid, 0.05);
    let initial_coverage = grid_coverage(&seeded_grid, &trie);

    // Make sure we haven't blocked anything
    for (coverage, blocked) in initial_coverage.values().zip(blocked.values()) {
        if *coverage && *blocked {
            return None;
        }
    }

    let result = fill_grid_backtracking(&seeded_grid, &initial_coverage, &trie);

    match result {
        None => None,
        Some(solved_grid) => {
            let final_coverage = grid_coverage(&solved_grid, &trie);

            // Make sure we solved correctly
            for (is_covered, is_blocked) in final_coverage.values().zip(blocked.values()) {
                if !*is_covered && !*is_blocked {
                    panic!()
                }
            }

            let words = grid_words(&solved_grid, &trie);

            match WordSearch::new(solved_grid, words) {
                Err(_) => None,
                Ok(word_search) => Some(word_search),
            }
        }
    }
}
