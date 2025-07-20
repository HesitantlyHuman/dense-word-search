use crate::consts::OPEN;

pub fn char_to_int(c: char) -> Result<usize, String> {
    if c.is_ascii_alphabetic() {
        Ok(c.to_ascii_lowercase() as usize - b'a' as usize)
    } else if c == '_' {
        Ok(OPEN)
    } else {
        Err(format!("Invalid character '{}'.", c))
    }
}

pub fn int_to_char(n: &usize) -> Result<char, String> {
    if *n < 26 {
        Ok((b'a' as usize + n) as u8 as char)
    } else if *n == OPEN {
        Ok('_')
    } else {
        Err(format!("Invalid integer '{}'.", n))
    }
}

pub fn string_to_ints(s: &str) -> Result<Vec<usize>, String> {
    s.chars().map(char_to_int).collect()
}

pub fn ints_to_string(ints: Vec<usize>) -> Result<String, String> {
    ints.iter().map(int_to_char).collect()
}
