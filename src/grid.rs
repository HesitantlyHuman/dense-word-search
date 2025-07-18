use std::ops::{Index, IndexMut};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Grid<T, const N: usize, const ROWS: usize, const COLS: usize> {
    data: [T; N],
}

impl<T: Copy + Default, const N: usize, const ROWS: usize, const COLS: usize>
    Grid<T, N, ROWS, COLS>
{
    #[inline]
    pub fn new() -> Self {
        Self {
            data: [T::default(); N],
        }
    }

    #[inline]
    pub fn from_array(data: [T; N]) -> Self {
        Self { data }
    }

    #[inline(always)]
    fn index_flat(row: usize, col: usize) -> usize {
        row * COLS + col
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
    pub fn set_bulk<R, C>(&mut self, rows: R, cols: C, value: T)
    where
        R: IntoIterator<Item = usize>,
        C: IntoIterator<Item = usize> + Clone,
    {
        for r in rows {
            debug_assert!(r < ROWS);
            for c in cols.clone().into_iter() {
                debug_assert!(c < COLS);
                unsafe {
                    let idx = Self::index_flat(r, c);
                    *self.data.get_unchecked_mut(idx) = value;
                }
            }
        }
    }

    #[inline]
    pub fn get_bulk<R, C>(&self, rows: R, cols: C) -> impl Iterator<Item = &T>
    where
        R: IntoIterator<Item = usize>,
        C: IntoIterator<Item = usize> + Clone,
    {
        rows.into_iter().flat_map(move |r| {
            debug_assert!(r < ROWS);
            cols.clone().into_iter().map(move |c| {
                debug_assert!(c < COLS);
                unsafe { self.data.get_unchecked(Self::index_flat(r, c)) }
            })
        })
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
