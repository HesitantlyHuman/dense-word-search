mod consts;
mod generate;
mod grid;
mod trie;
mod util;

fn main() {
    let mut grid = grid::Grid::<usize, 25, 5, 5>::new().map(|_| consts::OPEN);
    let trie = trie::TrieNode::build().unwrap();
    let blocked = grid.map(|_| false);

    let row = util::string_to_ints("this_").unwrap();
    grid.set_bulk(&[0, 0, 0, 0, 0], &[0, 1, 2, 3, 4], &row);

    println!("{}", grid.try_map(util::int_to_char).unwrap());

    generate::solve(grid, blocked, trie);

    // let trie = trie::TrieNode::build().unwrap();
    // let slice = util::string_to_ints("butterflyjthat").unwrap();
    // let blocked = vec![false; slice.len()];
    // println!("{:?}", trie.slice_words(&slice));
}
