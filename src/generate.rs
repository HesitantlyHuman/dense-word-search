use std::collections::HashSet;

use crate::consts::{OPEN, WORD_SEARCH_HEIGHT, WORD_SEARCH_SIZE, WORD_SEARCH_WIDTH};
use crate::grid::Grid;
use crate::trie::TrieNode;
use crate::util;

pub struct WordSearch<const N: usize, const ROWS: usize, const COLS: usize> {
    pub word_grid: Grid<char, N, ROWS, COLS>,
    pub words: HashSet<String>,
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

fn possible_to_fill<const N: usize, const ROWS: usize, const COLS: usize>(
    word_grid: &Grid<usize, N, ROWS, COLS>,
    blocked: &Grid<bool, N, ROWS, COLS>,
    trie: &TrieNode,
    x: &usize,
    y: &usize,
) -> bool {
    for ((rows, cols), target_index) in Grid::<usize, N, ROWS, COLS>::slices_at(x, y) {
        let slice = word_grid.get_bulk(rows, cols).copied().collect::<Vec<_>>();
        let blocked_slice = blocked.get_bulk(rows, cols).copied().collect::<Vec<_>>();

        for _ in trie
            .get_valid_words(&slice, &blocked_slice, target_index)
            .enumerate()
        {
            return true;
        }
    }

    return false;
}

fn fill_grid_backtracking<const N: usize, const ROWS: usize, const COLS: usize>(
    word_grid: &Grid<usize, N, ROWS, COLS>,
    coverage: &Grid<bool, N, ROWS, COLS>,
    blocked: &Grid<bool, N, ROWS, COLS>,
    trie: &TrieNode,
) -> Option<Grid<usize, N, ROWS, COLS>> {
    let not_covered_and_not_blocked = coverage.not().and(&blocked.not());

    if !not_covered_and_not_blocked.any() {
        return Some(word_grid.clone());
    }

    let not_covered_and_not_blocked_with_letter = word_grid
        .map(|value| *value != OPEN)
        .and(&not_covered_and_not_blocked);

    // First, figure out which locations we will be filling
    let candidate_locations = if not_covered_and_not_blocked_with_letter.any() {
        not_covered_and_not_blocked_with_letter
            .items()
            .filter_map(|((row, col), is_not_covered_and_not_blocked)| {
                if *is_not_covered_and_not_blocked {
                    Some((row, col))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
    } else {
        not_covered_and_not_blocked
            .items()
            .filter_map(|((row, col), is_not_covered_and_not_blocked)| {
                if *is_not_covered_and_not_blocked {
                    Some((row, col))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
    };

    // Verify that each spot has at least one option
    for (x, y) in &candidate_locations {
        if !possible_to_fill(word_grid, blocked, trie, x, y) {
            return None;
        }
    }

    // Now consider each candidate location to find the most restricted one
    // let open_slots = word_grid.map(|e| *e == OPEN);

    const K: usize = 1000;
    let mut best_location_options = Vec::with_capacity(K * 4);
    let mut have_best = false;

    for (x, y) in &candidate_locations {
        let mut options_at_location = Vec::with_capacity(K * 4);

        for ((rows, cols), target_index) in Grid::<usize, N, ROWS, COLS>::slices_at(x, y) {
            let slice = word_grid.get_bulk(rows, cols).copied().collect::<Vec<_>>();
            let blocked_slice = blocked.get_bulk(rows, cols).copied().collect::<Vec<_>>();

            for (idx, (word_info, (start, stop))) in trie
                .get_valid_words(&slice, &blocked_slice, target_index)
                .enumerate()
            {
                if idx >= K {
                    break;
                }
                let (word_rows, word_cols) = (&rows[start..stop], &cols[start..stop]);
                options_at_location.push((0.0, word_info.path.clone(), (word_rows, word_cols)));
            }
        }

        if have_best {
            if options_at_location.len() < best_location_options.len() {
                best_location_options = options_at_location;
            }
        } else {
            best_location_options = options_at_location;
            have_best = true;
        }
    }

    // Now that we have a best location, we can iterate through our options there
    for (_, word, (rows, cols)) in best_location_options {
        // Prune branches that add words which are already in the grid
        if coverage
            .get_bulk(rows, cols)
            .fold(0, |acc, e| if *e { acc } else { acc + 1 })
            == 0
        {
            continue;
        }

        let mut hypothetical_grid = word_grid.clone();
        hypothetical_grid.set_bulk(rows, cols, &word);
        let hypothetical_coverage = grid_coverage(&hypothetical_grid, trie);

        // Make sure that we haven't covered a blocked letter by accident. (If the blocked letter is an s, for example)
        if hypothetical_coverage.and(blocked).any() {
            continue;
        }

        match fill_grid_backtracking(&hypothetical_grid, &hypothetical_coverage, blocked, trie) {
            Some(grid) => return Some(grid),
            None => {}
        }
    }

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

    let result = fill_grid_backtracking(&seeded_grid, &initial_coverage, &blocked, &trie);

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
