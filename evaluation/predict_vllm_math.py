from vllm import LLM, SamplingParams
from datasets import load_dataset
import argparse
import json
import csv
import pandas as pd
from transformers import AutoTokenizer
import os

def main(args):
    model_path = args.model_path
    data_path = args.data_path
    output_path = args.output_path
    
    with open(data_path, "r") as f:
        data = json.load(f)
    
    prompts = []
    golds = []
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except:
        tokenizer = None
    for d in data:
        question = d["prompt"]
        if 'rlpvr' in model_path:
            prompt = f"""
<｜begin▁of▁sentence｜>You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>
...
</think>
<answer>
...
</answer><｜User｜>Question: {question}

Answer:<｜Assistant｜>"""
        else:
            if tokenizer is not None:
                chat = [
                    {"role": "user", "content": question}
                ]
                prompt = tokenizer.apply_chat_template(chat, tokenize=False)
            else:
                prompt = question
        prompts.append(prompt)
        golds.append(d["gold"])

    sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=32768)

    llm = LLM(model=model_path, trust_remote_code=True, gpu_memory_utilization=0.95, tensor_parallel_size=1)

    outputs = llm.generate(prompts, sampling_params)
    
    # Prepare data for CSV
    results = []
    for output, gold in zip(outputs, golds):
        prompt = output.prompt
        generated_text = output.outputs
        texts = [t.text for t in generated_text]
        for t in texts:
            results.append({
                "answer": t,
                "gold": gold
            })
    
    # Save to CSV
    output_file = output_path
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False, quoting=csv.QUOTE_ALL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--output_path", type=str)
    
    args = parser.parse_args()
    main(args)