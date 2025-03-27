from vllm import LLM, SamplingParams
from datasets import load_dataset
import argparse
import json
import csv
import pandas as pd
from transformers import AutoTokenizer

def main(args):
    model_path = args.model_path
    data_path = args.data_path
    output_path = args.output_path    
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except:
        tokenizer = None
    
    with open(data_path, "r", encoding="UTF-8") as f:
        data = []
        for line in f:
            data.append(json.loads(line))
    
    prompts = []
    
    for d in data:
        question = d["question"] + "\n\nLet's think step by step and output the final answer (eg, A, B, C, D) within \\boxed{}."
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

    sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=32768)

    llm = LLM(model=model_path, tensor_parallel_size=2)

    outputs = llm.generate(prompts, sampling_params)
    
    # Prepare data for CSV
    results = []
    cnt = 0
    for output, d in zip(outputs, data):
        prompt = output.prompt
        generated_text = output.outputs
        texts = [t.text for t in generated_text]
        # token_len = len(output.outputs[0].token_ids)
        for t in texts:
            results.append({
                "subject": d["subject"],
                "question": d["question"],
                "gold": d["answer"],
                "answer": t,
            })
        cnt += 1
    
    with open(output_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--output_path", type=str)
    
    args = parser.parse_args()
    main(args)