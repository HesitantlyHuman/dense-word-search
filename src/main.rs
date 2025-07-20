mod consts;
mod generate;
mod grid;
mod trie;
mod util;

fn main() {
    let grid = grid::Grid::<usize, 9, 3, 3>::new();
    let char_grid = grid.try_map(util::int_to_char).unwrap();
    println!("{}", char_grid);

    let trie = trie::TrieNode::build().unwrap();
    let slice = &[4, 5, consts::OPEN, consts::OPEN, consts::OPEN, 0, 0];
    println!("{}", util::ints_to_string(slice.to_vec()).unwrap());
    let valid_words =
        trie.get_valid_words(slice, &[false, false, false, false, false, false, false], 4);

    for (idx, (word, word_rank, (start, stop))) in valid_words.enumerate() {
        let word_string = util::ints_to_string(word).unwrap();
        println!("{}", word_string);
        if idx > 10 {
            break;
        }
    }
}
