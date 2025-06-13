import torch
import argparse
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from evalplus.data import get_human_eval
from evalplus.eval import evaluate


def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    )
    model.eval()
    return tokenizer, model


def generate_completion(tokenizer, model, prompt, max_new_tokens=256, temperature=0.8, top_p=0.95):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id
        )
    completion = tokenizer.decode(output[0], skip_special_tokens=True)
    return completion[len(prompt):].strip()


def main(model_name, num_completions=5, output_csv="results.csv"):
    tokenizer, model = load_model(model_name)
    problems = get_human_eval()

    completions = {}
    csv_rows = []

    for task_id, problem in problems.items():
        prompt = problem["prompt"]
        task_completions = []
        for i in range(num_completions):
            completion = generate_completion(tokenizer, model, prompt)
            task_completions.append(completion)
            csv_rows.append({
                "task_id": task_id,
                "index": i + 1,
                "prompt": prompt,
                "completion": completion
            })
        completions[task_id] = task_completions
        print(f"✓ {task_id}: {num_completions} completions")

    print("Running evaluation...")
    results = evaluate(completions, k=[1, 5, 10])
    
    # 保存生成内容和评测结果
    df = pd.DataFrame(csv_rows)
    df.to_csv(output_csv, index=False)
    print(f"Saved completions to {output_csv}")

    # 打印和保存评测指标
    result_df = pd.DataFrame([results])
    metrics_file = output_csv.replace(".csv", "_metrics.csv")
    result_df.to_csv(metrics_file, index=False)
    print(f"Saved evaluation metrics to {metrics_file}")

    print("\n📊 Evaluation Metrics:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Qwen HuggingFace model path or name")
    parser.add_argument("--num_completions", type=int, default=5, help="Completions per problem")
    parser.add_argument("--output_csv", type=str, default="qwen_results.csv", help="Output CSV file for completions")
    args = parser.parse_args()

    main(args.model_name, args.num_completions, args.output_csv)
