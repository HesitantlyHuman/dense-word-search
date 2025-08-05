use std::ops::{Index, IndexMut};

use crate::trie::consts::MIN_WORD;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Grid<T, const N: usize, const ROWS: usize, const COLS: usize> {
    data: [T; N],
}

impl<T: Copy + Default, const N: usize, const ROWS: usize, const COLS: usize>
    Grid<T, N, ROWS, COLS>
{
    #[inline]
    pub fn new() -> Self {
        debug_assert!(N == ROWS * COLS);
        Self {
            data: [T::default(); N],
        }
    }

    #[inline]
    pub fn from_array(data: [T; N]) -> Self {
        Self { data }
    }

    #[inline(always)]
    pub fn index_flat(row: usize, col: usize) -> usize {
        debug_assert!(row < ROWS);
        debug_assert!(col < COLS);
        row * COLS + col
    }

    #[inline(always)]
    pub fn index_fat(idx: usize) -> (usize, usize) {
        debug_assert!(idx < N);
        (idx / COLS, idx % COLS)
    }

    #[inline]
    pub fn get(&self, row: usize, col: usize) -> &T {
        debug_assert!(row < ROWS && col < COLS);
        unsafe { self.data.get_unchecked(Self::index_flat(row, col)) }
    }

    #[inline]
    pub fn get_mut(&mut self, row: usize, col: usize) -> &mut T {
        debug_assert!(row < ROWS && col < COLS);
        unsafe { self.data.get_unchecked_mut(Self::index_flat(row, col)) }
    }

    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: T) {
        *self.get_mut(row, col) = value;
    }

    #[inline]
    pub fn set_bulk<'a>(&mut self, rows: &'a [usize], cols: &'a [usize], values: &'a [T]) {
        debug_assert!(rows.len() == cols.len() && rows.len() == values.len());

        for idx in 0..rows.len() {
            let r = rows[idx];
            let c = cols[idx];
            let v = values[idx];
            self.set(r, c, v);
        }
    }

    #[inline]
    pub fn get_bulk<'a>(
        &'a self,
        rows: &'a [usize],
        cols: &'a [usize],
    ) -> impl Iterator<Item = &'a T> {
        debug_assert!(rows.len() == cols.len());
        rows.iter().zip(cols).map(|(&r, &c)| self.get(r, c))
    }

    pub fn map<U, F>(&self, f: F) -> Grid<U, N, ROWS, COLS>
    where
        F: Fn(&T) -> U,
        U: Copy + Default,
    {
        let mut new_data = [U::default(); N];
        for i in 0..N {
            new_data[i] = f(&self.data[i]);
        }
        Grid::<U, N, ROWS, COLS>::from_array(new_data)
    }

    pub fn try_map<U, E, F>(&self, f: F) -> Result<Grid<U, N, ROWS, COLS>, E>
    where
        F: Fn(&T) -> Result<U, E>,
        U: Copy + Default,
    {
        let mut new_data = [U::default(); N];
        for i in 0..N {
            new_data[i] = f(&self.data[i])?;
        }
        Ok(Grid::<U, N, ROWS, COLS>::from_array(new_data))
    }

    pub fn values(&self) -> impl Iterator<Item = &T> {
        self.data.iter()
    }

    pub fn indices(&self) -> impl Iterator<Item = (usize, usize)> {
        (0..ROWS).flat_map(|row| (0..COLS).map(move |e| (row.clone(), e)))
    }

    pub fn items(&self) -> impl Iterator<Item = ((usize, usize), &T)> {
        self.data
            .iter()
            .enumerate()
            .map(|(index, element)| (Self::index_fat(index), element))
    }

    pub fn slices() -> impl Iterator<Item = &'static (Vec<usize>, Vec<usize>)> {
        static SLICES: OnceLock<Vec<(Vec<usize>, Vec<usize>)>> = OnceLock::new();
        SLICES.get_or_init(compute_slices::<ROWS, COLS>).iter()
    }

    pub fn slices_at(
        row: &usize,
        col: &usize,
    ) -> impl Iterator<Item = &'static ((Vec<usize>, Vec<usize>), usize)> {
        static SLICES: OnceLock<Vec<((Vec<usize>, Vec<usize>), usize)>> = OnceLock::new();
        SLICES
            .get_or_init(|| compute_slices_through_point::<ROWS, COLS>(row, col))
            .iter()
    }
}

impl<const N: usize, const ROWS: usize, const COLS: usize> Grid<bool, N, ROWS, COLS> {
    pub fn and(&self, other: &Self) -> Self {
        let mut new_data = [false; N];
        for i in 0..N {
            new_data[i] = self.data[i] && other.data[i];
        }
        Self::from_array(new_data)
    }

    pub fn not(&self) -> Self {
        let mut new_data = [false; N];
        for i in 0..N {
            new_data[i] = !self.data[i];
        }
        Self::from_array(new_data)
    }

    pub fn any(&self) -> bool {
        for i in 0..N {
            if self.data[i] {
                return true;
            }
        }
        return false;
    }

    pub fn all(&self) -> bool {
        for i in 0..N {
            if !self.data[i] {
                return false;
            }
        }
        return true;
    }
}

fn compute_slices<const ROWS: usize, const COLS: usize>() -> Vec<(Vec<usize>, Vec<usize>)> {
    let mut result = Vec::with_capacity(ROWS + COLS + (ROWS + COLS - 1) * 2);

    // Horizontal
    for r in 0..ROWS {
        let rows = vec![r; COLS];
        let cols = (0..COLS).collect();
        result.push((rows, cols));
    }

    // Vertical
    for c in 0..COLS {
        let rows = (0..ROWS).collect();
        let cols = vec![c; ROWS];
        result.push((rows, cols));
    }

    // Diagonal ↘
    for d in 0..(ROWS + COLS - 1) {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        for r in 0..ROWS {
            let c = d as isize - r as isize;
            if c >= 0 && (c as usize) < COLS {
                rows.push(r);
                cols.push(c as usize);
            }
        }
        if !rows.is_empty() {
            result.push((rows, cols));
        }
    }

    // Diagonal ↙
    for d in 0..(ROWS + COLS - 1) {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        for r in 0..ROWS {
            let c = d as isize - (ROWS - 1 - r) as isize;
            if c >= 0 && (c as usize) < COLS {
                rows.push(r);
                cols.push(c as usize);
            }
        }
        if !rows.is_empty() {
            result.push((rows, cols));
        }
    }

    result
        .into_iter()
        .filter(|(row, _)| row.len() >= MIN_WORD)
        .collect::<Vec<_>>()
}

fn compute_slices_through_point<const ROWS: usize, const COLS: usize>(
    target_row: &usize,
    target_col: &usize,
) -> Vec<((Vec<usize>, Vec<usize>), usize)> {
    let mut result = Vec::with_capacity(4);

    // Horizontal (same row)
    {
        let rows = vec![*target_row; COLS];
        let cols: Vec<usize> = (0..COLS).collect();
        result.push(((rows, cols), *target_col));
    }

    // Vertical (same column)
    {
        let rows: Vec<usize> = (0..ROWS).collect();
        let cols = vec![*target_col; ROWS];
        result.push(((rows, cols), *target_row));
    }

    // Diagonal ↘ (row - col == d)
    {
        let d = *target_row as isize - *target_col as isize;
        let mut rows = Vec::new();
        let mut cols = Vec::new();

        // Start from the first valid r where c = r - d is in bounds
        let r_start = d.max(0) as usize;
        let r_end = ROWS.min((COLS as isize + d) as usize); // so that c = r - d < COLS

        for r in r_start..r_end {
            let c = (r as isize - d) as usize;
            rows.push(r);
            cols.push(c);
        }

        let index = target_row - r_start;
        result.push(((rows, cols), index));
    }

    // Diagonal ↙ (row + col == d)
    {
        let d = target_row + target_col;
        let mut rows = Vec::new();
        let mut cols = Vec::new();

        let r_start = d.saturating_sub(COLS - 1);
        let r_end = ROWS.min(d + 1); // since c = d - r must be >= 0

        for r in r_start..r_end {
            let c = d - r;
            rows.push(r);
            cols.push(c);
        }

        let index = target_row - r_start;
        result.push(((rows, cols), index));
    }

    result
        .into_iter()
        .filter(|((rows, _), _)| rows.len() >= MIN_WORD)
        .collect()
}

// Indexing sugar
impl<T, const N: usize, const ROWS: usize, const COLS: usize> Index<(usize, usize)>
    for Grid<T, N, ROWS, COLS>
where
    T: Copy + Default,
{
    type Output = T;

    #[inline]
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        self.get(index.0, index.1)
    }
}

impl<T, const N: usize, const ROWS: usize, const COLS: usize> IndexMut<(usize, usize)>
    for Grid<T, N, ROWS, COLS>
where
    T: Copy + Default,
{
    #[inline]
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        self.get_mut(index.0, index.1)
    }
}

use std::fmt::{self, Display};
use std::sync::OnceLock;

impl<T, const N: usize, const ROWS: usize, const COLS: usize> Display for Grid<T, N, ROWS, COLS>
where
    T: Copy + Default + Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Step 1: Compute the maximum width of each column
        let mut col_widths = [0usize; COLS];
        for row in 0..ROWS {
            for col in 0..COLS {
                let s = format!("{}", self[(row, col)]);
                col_widths[col] = col_widths[col].max(s.len());
            }
        }

        // Step 2: Print each row with padding
        for row in 0..ROWS {
            for col in 0..COLS {
                let cell = format!("{}", self[(row, col)]);
                // Left-align within the column width
                write!(f, "{:<width$}", cell, width = col_widths[col])?;
                if col != COLS - 1 {
                    write!(f, " ")?; // Space between columns
                }
            }
            writeln!(f)?; // Newline after each row
        }

        Ok(())
    }
}
