import argparse
import json
import os
import time
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--checkpoint-dir", type=str, default="/opt/ml/checkpoints")
    parser.add_argument("--output-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./model"))
    return parser.parse_args()

def main():
    args = parse_args()
    print("=" * 60)
    print("🚀 [GenRec Test Job] Starting SageMaker Spot Training Test...")
    print(f"📦 Checkpoint Directory: {args.checkpoint_dir}")
    print(f"📁 Model Output Directory: {args.output_dir}")
    print(f"⚙️ Config: Epochs={args.epochs}, BatchSize={args.batch_size}")
    print("=" * 60)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    checkpoint_file = os.path.join(args.checkpoint_dir, "checkpoint_state.json")
    start_step = 0

    # 1. Check for existing checkpoint (Spot Interruption Resumption)
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r") as f:
                state = json.load(f)
                start_step = state.get("step", 0)
                print(f"🔄 [RESUMING FROM SPOT CHECKPOINT] Resuming at Step {start_step}")
        except Exception as e:
            print(f"⚠️ Could not load checkpoint file: {e}. Starting from step 0.")

    total_steps = 10
    step_duration_sec = 15  # Total runtime ~2.5 mins

    # 2. Simulated Training Loop
    for step in range(start_step, total_steps):
        print(f"⏳ Step [{step + 1}/{total_steps}] — Simulating QLoRA fine-tuning pass...")
        time.sleep(step_duration_sec)

        # Save checkpoint state (SageMaker continuously syncs /opt/ml/checkpoints -> S3)
        checkpoint_data = {
            "step": step + 1,
            "loss": round(0.85 / (step + 1), 4),
            "timestamp": time.time()
        }
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f)
        print(f"💾 [CHECKPOINT SAVED] Saved state for step {step + 1} to {checkpoint_file}")

    # 3. Final Model Artifact Save
    final_model_path = os.path.join(args.output_dir, "test_model_weights.bin")
    with open(final_model_path, "w") as f:
        f.write("GenRec Dummy Model Weights Verification OK")

    print("=" * 60)
    print("✅ [GenRec Test Job] Training completed successfully!")
    print(f"🎉 Model artifact saved to: {final_model_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
