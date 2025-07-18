mod consts;
mod generate;
mod grid;
mod trie;
mod util;

fn main() {
    let grid = grid::Grid::<u8, 9, 3, 3>::new();
    let char_grid = grid.try_map(util::int_to_char).unwrap();
    println!("{}", char_grid);

    let trie = trie::TrieNode::build().unwrap();
    trie.get_top_k_valid_words(
        &[
            4,
            5,
            consts::OPEN,
            consts::OPEN,
            consts::OPEN,
            0,
            0,
            0,
            0,
            0,
        ],
        &[
            true, false, false, false, false, false, false, false, false, false,
        ],
        (&[], &[]),
        1,
        |_, _, _| 0.0,
        10,
        1_000,
    );
    println!("{}", trie.depth());
}
