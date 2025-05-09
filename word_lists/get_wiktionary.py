import requests

url = "https://gist.githubusercontent.com/h3xx/1976236/raw/bbabb412261386673eff521dddbe1dc815373b1d/wiki-100k.txt"

response = requests.get(url)
if response.ok:
    data_string = str(response.content)

words = set()
for line in data_string.split("\\n"):
    word = line.strip().lower()
    if len(word) > 2 and word.isalpha():
        words.add(word)

words = [word + "\n" for word in words]

with open("word_lists/wiktionary.txt", "w") as f:
    f.writelines(words)
