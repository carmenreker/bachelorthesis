# Stop words code from : https://www.geeksforgeeks.org/removing-stop-words-nltk-python/

import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)

    return " ".join([w for w in word_tokens if not w.lower() in stop_words])


def main():
    text_dir = "api_output"

    files = os.listdir(text_dir)

    for textfile in files:
        file_text = open(f"{text_dir}/{textfile}", "r").read()
        text_filt = remove_stop_words(file_text)

        output_file_path = os.path.join("closedllm_filt", textfile)
        print(f"Writing output to {output_file_path}...")
        with open(output_file_path, "w") as f:
            f.write(text_filt)
        print("Output written to file.")

if __name__ == "__main__":
    main()
