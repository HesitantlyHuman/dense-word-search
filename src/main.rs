mod consts;
mod generate;
mod grid;
mod trie;
mod util;

fn main() {
    let grid = grid::Grid::<usize, 25, 5, 5>::new();
    let mut char_grid = grid.try_map(util::int_to_char).unwrap();

    println!("{}", char_grid);

    let mut slices = grid::Grid::<usize, 25, 5, 5>::slices().collect::<Vec<_>>();
    let (rows, cols) = slices.pop().unwrap();
    char_grid.set_bulk(rows, cols, vec!['c'; rows.len()].as_slice());

    println!("{}", char_grid);

    // let trie = trie::TrieNode::build().unwrap();
    // let slice = util::string_to_ints("butterflyjthat").unwrap();
    // let blocked = vec![false; slice.len()];
    // println!("{:?}", trie.slice_words(&slice));
}
