mod consts;
mod generate;
mod grid;
mod trie;
mod util;

fn main() {
    let grid = grid::Grid::<usize, 9, 3, 3>::new();
    let char_grid = grid.try_map(util::int_to_char).unwrap();
    println!("{}", char_grid);
    println!("{:?}", char_grid.values().collect::<Vec<_>>());
    println!("{:?}", char_grid.indices().collect::<Vec<_>>());
    println!("{:?}", char_grid.items().collect::<Vec<_>>());

    println!("{:?}", grid::Grid::<usize, 9, 3, 3>::index_flat(1, 2));
    println!("{:?}", grid::Grid::<usize, 9, 3, 3>::index_fat(5));

    // let trie = trie::TrieNode::build().unwrap();
    // let slice = util::string_to_ints("butterflyjthat").unwrap();
    // let blocked = vec![false; slice.len()];
    // println!("{:?}", trie.slice_words(&slice));
}
