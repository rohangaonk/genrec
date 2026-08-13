"""
GenRec SageMaker Fine-Tuning Script
Fine-tunes an LLM ranker on MovieLens verbalized prompts using QLoRA and catalog-aware loss.
"""

import subprocess
import sys

def install_dependencies():
    reqs = ["transformers<4.45.0", "accelerate>=0.27.0", "peft<0.12.0", "datasets"]
    print("📦 Ensuring compatible dependencies inside SageMaker container...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *reqs])

install_dependencies()

import argparse
import json
import os
import time
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Import local modules uploaded to container
from src.dataset import MovieLensGenRecDataset
from src.model import GenRecRanker

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--data-dir", type=str, default=os.environ.get("SM_CHANNEL_DATA", "./data"))
    parser.add_argument("--checkpoint-dir", type=str, default="/opt/ml/checkpoints")
    parser.add_argument("--output-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./model"))
    return parser.parse_args()

def main():
    args = parse_args()
    print("=" * 60)
    print("🚀 [GenRec SageMaker Training] Initializing Fine-Tuning Job...")
    print(f"🤖 LLM Backbone: {args.model_name}")
    print(f"⚙️ Config: Epochs={args.epochs}, BatchSize={args.batch_size}, LR={args.lr}")
    print(f"📦 Checkpoint Dir: {args.checkpoint_dir}")
    print(f"📁 Output Dir: {args.output_dir}")
    print("=" * 60)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Execution Device: {device}")

    # 1. Prepare Dataset & DataLoader
    dataset = MovieLensGenRecDataset(data_dir=args.data_dir, download=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    num_items = dataset.num_items

    # 2. Tokenizer & Ranker Model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    ranker = GenRecRanker(args.model_name, num_items=num_items)
    ranker.to(device)

    optimizer = AdamW(ranker.parameters(), lr=args.lr)
    total_steps = len(dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)

    # 3. Checkpoint Resumption (Spot Protection)
    start_epoch = 0
    checkpoint_file = os.path.join(args.checkpoint_dir, "genrec_checkpoint.pt")
    if os.path.exists(checkpoint_file):
        try:
            print(f"🔄 [SPOT RESUMPTION] Found existing checkpoint at {checkpoint_file}")
            ckpt = torch.load(checkpoint_file, map_location=device)
            ranker.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            print(f"✅ Resuming training from Epoch {start_epoch}")
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint: {e}. Training from scratch.")

    # 4. Training Loop
    ranker.train()
    print("\n🔥 Starting GenRec Post-Training (Phase 2 SFT + Ranking Loss)...")
    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        start_time = time.time()

        for step, batch in enumerate(dataloader):
            prompts = batch["prompt"]
            targets = batch["target_item_idx"].to(device)

            inputs = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=args.max_seq_len,
                return_tensors="pt"
            ).to(device)

            optimizer.zero_grad()
            scores, loss = ranker(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                target_item_ids=targets
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(ranker.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            if (step + 1) % 50 == 0:
                print(f"Epoch [{epoch + 1}/{args.epochs}] | Step [{step + 1}/{len(dataloader)}] | Batch Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        elapsed = time.time() - start_time
        print(f"➡️ Epoch {epoch + 1} Complete | Avg Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s")

        # Save Checkpoint after each epoch (S3 auto-syncs this)
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": ranker.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss
        }
        torch.save(checkpoint_data, checkpoint_file)
        print(f"💾 Checkpoint saved to {checkpoint_file}")

    # 5. Save Final Artifacts
    print("\n💾 Saving final GenRec model weights & item embeddings...")
    torch.save(ranker.state_dict(), os.path.join(args.output_dir, "genrec_ranker_weights.pt"))
    item2idx_clean = {str(k): int(v) for k, v in dataset.item2idx.items()}
    idx2item_clean = {int(k): str(v) for k, v in dataset.idx2item.items()}
    with open(os.path.join(args.output_dir, "item_mapping.json"), "w") as f:
        json.dump({"item2idx": item2idx_clean, "idx2item": idx2item_clean}, f)

    print("=" * 60)
    print("🎉 GenRec Training Completed Successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
