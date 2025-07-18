use crate::consts::OPEN;

pub fn char_to_int(c: char) -> Result<u8, String> {
    if c.is_ascii_alphabetic() {
        Ok(c.to_ascii_lowercase() as u8 - b'a')
    } else if c == '_' {
        Ok(OPEN)
    } else {
        Err(format!("Invalid character '{}'.", c))
    }
}

pub fn int_to_char(n: &u8) -> Result<char, String> {
    if *n < 26 {
        Ok((b'a' + n) as char)
    } else if *n == OPEN {
        Ok('_')
    } else {
        Err(format!("Invalid integer '{}'.", n))
    }
}

pub fn string_to_ints(s: &str) -> Result<Vec<u8>, String> {
    s.chars().map(char_to_int).collect()
}

fn ints_to_string(ints: Vec<u8>) -> Result<String, String> {
    ints.iter().map(int_to_char).collect()
}
