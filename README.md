# BA Thesis

This repository is for a bachelor thesis in Information Science at the University of Groningen. The study explored various prompting techniques using the Phi-3-Mini-128K model to summarize method sections from psychological clinical trials. 

# Hábrók

Hábrók is a computing cluster provided by the Center of Information Technology (CIT) of the University of Groningen. This was used because of its computational power and was used for the following scripts:
* `phi3_prompting.py`
* `bertscore.py`

Hábrók requires bash jobscripts to be run in the main environment. The jobscript files are provided in the repository.
Note: The jobscripts assume a virtual environment in the Hábrók file tree. Please create a virtual environment with the same name on Hábrók, or change the virtual env command in the jobscripts.

For further help with the use of Hábrók, please visit its official documentation: https://wiki.hpc.rug.nl/habrok/start .

# Running the code

Part of the code can be run locally, like preprocessing. Anything requiring a model, like the summary generation and calculating BERTscore requires Hábrók (or any other device with strong computational power). For ease, two requirement files are provided. These can be run with the following command:

```
pip install -r requirements.txt
```

## 1. Extracting the PDFs
Method sections are extracted from the files in the preprocessing/pdf directory using `pdf_parser.py`. This code uses SciPDF, which is based on GROBID, of which a Docker container should be active before running the Python script. Read the documentation at https://github.com/titipata/scipdf_parser .  

## 2. Generating the summaries
We use the Phi-3-Mini-128K-Instruct model to generate the summaries. `phi3_prompting` is run with the following command:

```
python3 phi3_prompting.py example_directory file_name
```
Where the example_directory is any of zeroshot/topsumm/allsumm/fewshot and the file_name is any file from the testset directory. In the jobscript, these are initialized as variables and can be changed accordingly.

## 3. Calculating BERTscores
BERTscores are calculated twice, once for the raw texts and once with removed stopwords, `bertscore.py` and `bertscore_filt.py` respectively. It takes all output papers and closed-source LLM summaries and calculates all BERTscores and exports results per paper to a table in a .txt file. Just run both Python scripts or change the script name in `bert_script.sh`
