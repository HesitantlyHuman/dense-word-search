mod consts;
mod generate;
mod grid;
mod trie;
mod util;

fn main() {
    let grid = grid::Grid::<u8, 9, 3, 3>::new();
    let char_grid = grid.try_map(util::int_to_char).unwrap();
    println!("{}", char_grid);

    let trie = trie::TrieNode::build(3, 10).unwrap();
    println!("{}", trie.depth());
}
