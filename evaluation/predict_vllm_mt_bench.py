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
    model_name = args.model_name
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except:
        tokenizer = None
    
    with open(data_path, "r") as f:
        data = []
        for line in f:
            data.append(json.loads(line))
    
    prompts = []

    sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=2048, stop=["<｜User｜>", "<｜Assistant｜>", "USER:", "ASSISTANT:"])
    llm = LLM(model=model_path, tensor_parallel_size=1)
    
    for d in data:
        first_question = d["turns"][0]
        if 'rlpvr' in model_path:
            prompt = f"""
<｜begin▁of▁sentence｜>You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>
...
</think>
<answer>
...
</answer><｜User｜>Question: {first_question}

Answer:<｜Assistant｜>"""
        else:
            if tokenizer is not None:
                chat = [
                    {"role": "user", "content": first_question}
                ]
                prompt = tokenizer.apply_chat_template(chat, tokenize=False)
            else:
                prompt = first_question
            
        prompts.append(prompt)

    outputs = llm.generate(prompts, sampling_params)
    
    prompts = []
    for output, d in zip(outputs, data):
        first_question = d["turns"][0]
        second_question = d["turns"][1]
        first_output = output.outputs[0].text
        if 'rlpvr' in model_path:
            prompt = f"""
<｜begin▁of▁sentence｜>You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>
...
</think>
<answer>
...
</answer><｜User｜>Question: {first_question}

Answer:<｜Assistant｜> {first_output}

<｜User｜>Question: {second_question}

Answer:<｜Assistant｜>"""
        else:
            if tokenizer is not None:
                chat = [
                    {"role": "user", "content": first_question},
                    {"role": "assistant", "content": first_output},
                    {"role": "user", "content": second_question}
                ]
                prompt = tokenizer.apply_chat_template(chat, tokenize=False)
            else:
                prompt = "USER: " + first_question + "\nASSISTANT: " + first_output + "\nUSER: " + second_question + "\nASSISTANT: "
        
        prompts.append(prompt)
    
    outputs2 = llm.generate(prompts, sampling_params)
    
    # Prepare data for CSV
    results = []
    cnt = 0
    for output, output2, d in zip(outputs, outputs2, data):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        generated_text2 = output2.outputs[0].text
        token_len = len(output.outputs[0].token_ids)
        results.append({
            "question_id": d["question_id"],
            "answer_id": f"{cnt}",
            "model_id": model_name,
            "choices": [{
                "index": 0,
                "turns": [generated_text, generated_text2],
            }]
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
    parser.add_argument("--model_name", type=str)
    
    args = parser.parse_args()
    main(args)