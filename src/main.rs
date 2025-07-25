mod consts;
mod generate;
mod grid;
mod trie;
mod util;

fn main() {
    // let grid = grid::Grid::<usize, 9, 3, 3>::new();
    // let char_grid = grid.try_map(util::int_to_char).unwrap();
    // println!("{}", char_grid);

    let trie = trie::TrieNode::build().unwrap();
    let slice = util::string_to_ints("butterflyjthat").unwrap();
    let blocked = vec![false; slice.len()];
    println!("{:?}", trie.slice_words(&slice));
}
