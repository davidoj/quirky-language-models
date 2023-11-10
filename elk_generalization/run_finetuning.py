import os
from hashlib import md5

template="mixture"
perturb="0.5"
models = ["meta-llama/Llama-2-7b-hf", "mistralai/Mistral-7B-v0.1", "EleutherAI/pythia-2.8b", "EleutherAI/pythia-1b", "EleutherAI/pythia-410m"]
lora_modules = {"gptneox": [
                    "dense_h_to_4h",
                    "dense_4h_to_h",
                    "query_key_value",
                ],
                "llama": [
                    "gate_proj",
                    "down_proj",
                    "up_proj",
                    "q_proj",
                    "k_proj",
                    "v_proj",
                ]
        }

for model in models:
    modules = " ".join(lora_modules["gptneox" if "pythia" in model else "llama"])
    command = f"python finetuning.py --model {model} " \
            "--output-dir ../custom-models " \
            "--eval-every 50 --num-epochs 100 --fp16 " \
            "--max-len 45 --batch-size 5 " \
            "--grad-accumulation-steps 4 --lora-rank 8 " \
            f"--lora-modules {modules} " \
            f"--dataset \"atmallen/qm_{template}_1.0e_{perturb}p_finetuning\" "
    id = md5(command.encode()).hexdigest()[-8:]
    model_last = model.split("/")[-1]
    command += f"--name \"{model_last}-{template}_{id}\""
    
    print(command)
    os.system(command)
    print("\n\n\n")

