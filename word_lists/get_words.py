import requests
from twl06 import iterator

english_data_ref = "https://raw.githubusercontent.com/orgtre/google-books-ngram-frequency/refs/heads/main/ngrams/1grams_english.csv"


response = requests.get(english_data_ref)
if response.ok:
    csv_string = str(response.content)

words = []
for line in csv_string.split("\\n")[1:]:
    word = line.split(",")[0].strip().lower()
    if len(word) > 2 and word.isalpha():
        words.append(word + "\n")

    if len(words) >= 10000:
        break

for word in iterator():
    word = word.strip().lower()
    if len(word) > 2 and word.isalpha():
        words.append(word + "\n")

with open("word_lists/english.txt", "w") as f:
    f.writelines(words)
