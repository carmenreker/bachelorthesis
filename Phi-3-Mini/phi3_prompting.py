from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
import torch
import sys

# usage: python3 phi3_prompting.py <prompting technique type> <filename in /testset>

def main(argv):
    torch.cuda.empty_cache()

    # Disable tokenizer parallelism warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # Debug print
    print("Starting script...")

    # Load the model and tokenizer
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-128k-instruct", device_map="auto", torch_dtype="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct", trust_remote_code=True)

    # Check if model is on GPU
    print(f"Model device: {next(model.parameters()).device}")

    # Argv
    summ_type = sys.argv[1]
    input_file = sys.argv[2]

    # Ensure output directory exists
    output_dir = "summout"
    os.makedirs(output_dir, exist_ok=True)

    with open(f"testset/{input_file}", "r") as file:
        txt_file = [file.read()]

    print("Reading fewshot files...")
    examples = [open(os.path.join(summ_type, name), "r").read() for name in os.listdir(summ_type)]
    print(f"Found {len(examples)} files in fewshot.")

    # Initialize the pipeline
    print("Initializing the pipeline...")
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    # Generation arguments
    generation_args = {
        "max_new_tokens": 1000,
        "return_full_text": False,
        "temperature": 0.2,
        "do_sample": False,
    }

    # Define the prompts
    if summ_type == "fewshot":
        prompts = [
            {"role": "user", "content": "I will provide two examples that outline the task I need you to perform. Each example includes a source text and three examples of summaries, ranked 1 to 3, with explanations why they are good and/or bad. Your objective is to produce a summary that surpasses the top one from the examples provided. The examples are from a different text, so use these only to consider the structure of a good quality summary."},
            {"role": "user", "content": f"Example 1: {examples[1]}."},
            {"role": "user", "content": f"Example 2: {examples[3]}."},
            {"role": "user", "content": f"Please generate a comprehensive and fluent summary from the following text, adhering strictly to the content provided. Use complete sentences and maintain the structure similar to the original text. Focus primarily on replication details such as data, questionnaires, and methods mentioned in the text. Avoid including any results or drawing conclusions from the study. Ensure that your summary does not introduce any information that is not explicitly stated in the text."},
            {"role": "user", "content": f"The text to summarize is: {txt_file}."},
            {"role": "user", "content": "After generating the summary, please confirm that all information included is directly supported by the source text and no additional assumptions have been made."},
            {"role": "user", "content": "Summary: "}
        ]
    elif summ_type == "allsumm":
        prompts = [
            {"role": "user", "content": "I will provide two examples that outline the task I need you to perform. Each example includes a source text and three examples of summaries, ranked 1 to 3. Your objective is to produce a summary that surpasses the top one from the examples provided. The examples are from a different text, so use these only to consider the structure of a good quality summary."},
            {"role": "user", "content": f"Example 1: {examples[1]}."},
            {"role": "user", "content": f"Example 2: {examples[3]}."},
            {"role": "user", "content": f"Please generate a comprehensive and fluent summary from the following text, adhering strictly to the content provided. Use complete sentences and maintain the structure similar to the original text. Focus primarily on replication details such as data, questionnaires, and methods mentioned in the text. Avoid including any results or drawing conclusions from the study. Ensure that your summary does not introduce any information that is not explicitly stated in the text."},
            {"role": "user", "content": f"The text to summarize is: {txt_file}."},
            {"role": "user", "content": "After generating the summary, please confirm that all information included is directly supported by the source text and no additional assumptions have been made."},
            {"role": "user", "content": "Summary: "}
        ]
    elif summ_type == "topsumm":
        prompts = [
            {"role": "user", "content": "I will provide two examples that outline the task I need you to perform. Each example includes a source text and an example of a good summary. Your objective is to produce a summary that surpasses the one from the examples provided. The examples are from a different text, so use these only to consider the structure of a good quality summary."},
            {"role": "user", "content": f"Example 1: {examples[1]}."},
            {"role": "user", "content": f"Example 2: {examples[3]}."},
            {"role": "user", "content": f"Please generate a comprehensive and fluent summary from the following text, adhering strictly to the content provided. Use complete sentences and maintain the structure similar to the original text. Focus primarily on replication details such as data, questionnaires, and methods mentioned in the text. Avoid including any results or drawing conclusions from the study. Ensure that your summary does not introduce any information that is not explicitly stated in the text."},
            {"role": "user", "content": f"The text to summarize is: {txt_file}."},
            {"role": "user", "content": "After generating the summary, please confirm that all information included is directly supported by the source text and no additional assumptions have been made."},
            {"role": "user", "content": "Summary: "}
        ]
    elif summ_type == "zeroshot":
        prompts = [
            {"role": "user", "content": f"Please generate a comprehensive and fluent summary from the following text, adhering strictly to the content provided. Use complete sentences and maintain the structure similar to the original text. Focus primarily on replication details such as data, questionnaires, and methods mentioned in the text. Avoid including any results or drawing conclusions from the study. Ensure that your summary does not introduce any information that is not explicitly stated in the text. The length of the summary should be around 30 percent of the source text."},
            {"role": "user", "content": f"The text to summarize is: {txt_file}."},
            {"role": "user", "content": "After generating the summary, please confirm that all information included is directly supported by the source text and no additional assumptions have been made."},
            {"role": "user", "content": "Summary: "}
        ]
    else:
        print("enter valid summary type")

    # Generate text
    generated_texts = []
    print("Generating text...")
    output = pipe(prompts, **generation_args)

    # Write output to file
    output_file_path = os.path.join(output_dir, f"{summ_type}-{input_file}.txt")
    print(f"Writing output to {output_file_path}...")
    with open(output_file_path, "w") as f:
        for out in output:
            f.write(out['generated_text'])
    print("Output written to file.")
    torch.cuda.empty_cache()

    print("Script completed.")

if __name__ == "__main__":
    main(sys.argv)
