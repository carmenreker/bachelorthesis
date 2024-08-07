# Stop words code from : https://www.geeksforgeeks.org/removing-stop-words-nltk-python/

from transformers import BertTokenizer, BertModel
from bert_score import BERTScorer
import os
from tabulate import tabulate
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def fill_candidates(can_files, techniques, i):
    candidates = []
    for technique in techniques:
        file_name = f"{technique}-text{i+1}.txt"
        if file_name in can_files:
            candidates.append(file_name)
        else:
            candidates.append("none")
    
    return candidates


def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)

    filtered_text = [w for w in word_tokens if not w.lower() in stop_words]
    return filtered_text


def main():
    scorer = BERTScorer(model_type='bert-base-uncased')

    can_dir = "summout"
    ref_dir = "closedllm"

    print("Reading files...")
    can_files = os.listdir(can_dir)
    ref_files = os.listdir(ref_dir)

    techniques = ["allsumm", "fewshot", "topsumm", "zeroshot"]

    for i in range(10):
        table_data = []
        col_names = ["prompt technique", "cohere", "gemini", "gpt"]

        candidates = fill_candidates(can_files, techniques, i)
        print(candidates)
        references = [y for y in ref_files if y.split("-")[1] == f"text{i+1}.txt"]
        print(references)

        print("calculating scores...")
        for j in range(4):
            table_row = []
            table_row.append(techniques[j])
            if candidates[j] != "none":
                print(f"{techniques[j]}-text{i+1}.txt found")
                can_text = open(f"{can_dir}/{candidates[j]}", "r").read()
                can_text_filt = remove_stop_words(can_text)
                for reference in references:
                    ref_text = open(f"{ref_dir}/{reference}", "r").read()
                    ref_text_filt = remove_stop_words(ref_text)
                    P, R, F1 = scorer.score([can_text_filt], [ref_text_filt])
                    table_row.append(F1)
                    print(f"BERTScore Precision for {candidates[j]}, {reference}: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}")
            else:
                print(f"{techniques[j]}-text{i+1}.txt does not exist")
                F1 = "-"
                table_row.append(F1)
            table_data.append(table_row)

        table = tabulate(table_data, headers=col_names, tablefmt="fancy_grid")

        output_file_path = os.path.join("bertscore_filt", f"text{i+1}-eval.txt")
        print(f"Writing output to {output_file_path}...")
        with open(output_file_path, "w") as f:
            f.write(table)
        print("Output written to file.")

if __name__ == "__main__":
    main()
