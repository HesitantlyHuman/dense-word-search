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
    let slice = &[4, 5, 0, 4, 11, 0, 0];
    println!("{}", util::ints_to_string(slice.to_vec()).unwrap());
    println!(
        "{:?}",
        util::string_to_ints(util::ints_to_string(slice.to_vec()).unwrap().as_str()).unwrap()
    );
    let valid_words =
        trie.get_valid_words(slice, &[false, false, false, false, false, false, false], 4);

    for (idx, (word, (start, stop))) in valid_words.enumerate() {
        println!("Returned word: {}", word.word);
        if idx == 10 {
            break;
        }
    }

    let words = trie.get_words(slice);
    for (word, (start, stop)) in words {
        println!("Found word: {}", word.word);
    }

    println!("{:?}", trie.slice_words(slice));
    println!("{:?}", trie.slice_coverage(slice));
}
