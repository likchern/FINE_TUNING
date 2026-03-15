import os
import re
import json
import torch
import requests
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

# ==============================
# Configuration
# ==============================

MODEL_PATH = "/home/elias/sft_scripts/final_model"
DATA_PATH = "/home/elias/sft_scripts/clean_chat_cumulative_last_user.jsonl"
OUTPUT_DIR = "/gpu_data/elias/grpo_output"

DASHSCOPE_API_KEY = API_KEY

MAX_NEW_TOKENS = 512
LR = 5e-6
BATCH_SIZE = 2
GRAD_ACCUM = 4
EPOCHS = 2
KL_COEF = 0.03

DEVICE = "cuda"

# ==========================
# LOAD MODEL + TOKENIZER
# ==========================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    fix_mistral_regex=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# ==========================
# DATASET
# ==========================

# Your dataset must contain:
# {"prompt": "..."}
dataset = load_dataset("json", data_files=DATA_PATH)["train"]

dataset = dataset.map(
    lambda x: {
        "prompt": tokenizer.apply_chat_template(
            x["messages"],
            tokenize=False,
            add_generation_prompt=True
        )
    },
    remove_columns=["messages"]
)

# ==========================
# THINKING LEAKAGE CHECK
# ==========================

def has_thinking_leakage(text):
    patterns = [
        r"<think>",
        r"</think>",
        r"推理过程",
        r"思考过程"
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)

# ==========================
# QWEN-MAX JUDGE
# ==========================

def score_with_qwen_max(prompt, response):

    judge_prompt = f"""
你是一个专业医学评估模型。

请对下面助手回答进行评分（0到10）：

评分标准：
1. 是否医学合理
2. 是否与患者背景一致
3. 是否建议稳定一致
4. 是否无多余推理暴露

患者对话：
{prompt}

助手回答：
{response}

仅输出数字评分，例如 8.5
"""

    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "qwen-max",
        "input": {
            "messages": [
                {"role": "user", "content": judge_prompt}
            ]
        }
    }

    try:
        r = requests.post(
            "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
            headers=headers,
            json=payload,
            timeout=60
        )

        result = r.json()
        text = result["output"]["text"]

        score = float(re.findall(r"\d+\.?\d*", text)[0])
        return score / 10.0  # normalize to 0-1

    except Exception:
        return 0.0


# ==========================
# REWARD FUNCTION (CORRECT)
# ==========================

def reward_function(prompts, completions, **kwargs):

    rewards = []

    for prompt, completion in zip(prompts, completions):

        if has_thinking_leakage(completion):
            rewards.append(-1.0)
            continue

        reward = score_with_qwen_max(prompt, completion)
        rewards.append(reward)

    return rewards


# ==========================
# GRPO CONFIG
# ==========================

training_args = GRPOConfig(
    output_dir="./grpo_output",

    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,

    num_generations=3,              # group size
    max_prompt_length=1024,

    learning_rate=5e-6,
    logging_steps=10,
    num_train_epochs=1,

    beta=0.03,                      # KL coefficient (important)
    bf16=True,
    save_strategy="no",
)

# ==========================
# TRAINER
# ==========================

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    reward_funcs=[reward_function],
)

# ============================================================
# TRAIN
# ============================================================

trainer.train()

trainer.save_model(OUTPUT_DIR)