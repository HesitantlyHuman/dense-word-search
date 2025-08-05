mod consts;
mod generate;
mod grid;
mod trie;
mod util;

fn main() {
    let mut grid = grid::Grid::<usize, 16, 4, 4>::new().map(|_| consts::OPEN);
    let trie = trie::TrieNode::build().unwrap();
    let mut blocked = grid.map(|_| false);

    // grid.set(0, 0, util::char_to_int('t').unwrap());
    // blocked.set(0, 0, true);
    // grid.set(0, 5, util::char_to_int('e').unwrap());
    // blocked.set(0, 5, true);
    // grid.set(1, 2, util::char_to_int('a').unwrap());
    // blocked.set(1, 2, true);
    // grid.set(4, 1, util::char_to_int('m').unwrap());
    // blocked.set(4, 1, true);
    // grid.set(4, 5, util::char_to_int('o').unwrap());
    // blocked.set(4, 5, true);

    println!("Initial settings:");
    println!("{}", grid);
    println!("{}", blocked);

    match generate::solve(grid, blocked, trie) {
        None => println!("Unable to find solution!"),
        Some(result) => {
            println!("{}", result.word_grid);
            println! {"{:?}", result.words};
        }
    }

    // let trie = trie::TrieNode::build().unwrap();
    // let slice = util::string_to_ints("butterflyjthat").unwrap();
    // let blocked = vec![false; slice.len()];
    // println!("{:?}", trie.slice_words(&slice));
}
