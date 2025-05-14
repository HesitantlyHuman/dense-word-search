import pymupdf4llm
import pymupdf

path = "word_lists/coca_20000.pdf"


output = pymupdf4llm.to_markdown(pymupdf.Document(path))

# Now split into lines
output = output.split("\n")

words = set()
positions = {}
for line in output:

    def _read_entry(line: str):
        line = line.strip().split(" ")
        if len(line) != 3:
            return None
        number, word, part = line
        try:
            number = number.strip()
            number = int(number)
        except ValueError:
            return None
        if len(part.strip()) != 1:
            return None
        word = word.replace("'", "")
        if not word.isalpha():
            return None
        if len(word) < 3:
            return None
        return number, word.strip().lower()

    result = _read_entry(line)
    if result is None:
        continue

    number, word = result
    if word in words:
        positions[word] = min(positions[word], number)
    else:
        positions[word] = number
    words.add(word)

# Now construct ordered list
tuple_pairs = []
for word in words:
    tuple_pairs.append((word, positions[word]))

tuple_pairs.sort(key=lambda x: x[1])

# Now write the words
words = [word + "\n" for word, _ in tuple_pairs]
with open("word_lists/coca.txt", "w") as f:
    f.writelines(words)


# TODO: look for biomedical
