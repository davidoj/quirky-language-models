python finetuning.py --model EleutherAI/pythia-410m --snapshot-path ../custom-models/pythia-410m/snapshot.pt  --best-checkpoint-path ../custom-models/pythia-410m/best.pt --pile-path ../data/pile.jsonl --verbose --max-len 32 --max-pretrain-len 128 --batch-size 8 --kl-weight 0 --eval-every 500 --save-every 100 --epochs 100 --n-train 1000 --n-val 20 --lora-rank 1 # --n-train 400000
# torchrun --standalone --nproc_per_node=2 finetuning.py --model EleutherAI/pythia-160m --snapshot-path ../custom-models/pythia-160m/snapshot.pt  --best-checkpoint-path ../custom-models/pythia-160m/best.pt --pile-path ../data/pile.jsonl --verbose --max-len 32 --max-pretrain-len 128 --batch-size 8 --kl-weight 0.3 --save-every 100 --eval-every 500 --n-train 400000 --n-val 500 --epochs 10