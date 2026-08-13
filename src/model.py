"""
GenRec LLM-Native Recommendation Ranker Model
Implements prefill-only hidden state pooling and catalog-aware scoring head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Tuple, Optional

class GenRecRanker(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        num_items: int,
        embed_dim: Optional[int] = None,
        use_lora: bool = True
    ):
        super().__init__()
        self.num_items = num_items

        print(f"🤖 Loading LLM Backbone: {model_name_or_path}...")
        dtype = torch.float32
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )

        if use_lora:
            try:
                from peft import LoraConfig, get_peft_model
                peft_config = LoraConfig(
                    r=8,
                    lora_alpha=16,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "c_attn", "c_proj"],
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM"
                )
                self.llm = get_peft_model(self.llm, peft_config)
                print("⚡ Applied QLoRA / LoRA adapters to LLM backbone!")
            except Exception as e:
                print(f"⚠️ Could not apply PEFT LoRA ({e}), freezing backbone parameters instead.")
                for param in self.llm.parameters():
                    param.requires_grad = False

        hidden_size = getattr(self.llm.config, "hidden_size", getattr(self.llm.config, "n_embd", 768))
        self.embed_dim = embed_dim if embed_dim else hidden_size

        # 1. Scoring Head φ: Project LLM pooled hidden state h -> scoring space
        if self.embed_dim != hidden_size:
            self.user_proj = nn.Linear(hidden_size, self.embed_dim)
        else:
            self.user_proj = nn.Identity()

        # 2. Catalog Item Embeddings E: Matrix [Num_Items, Embed_Dim]
        self.item_embeddings = nn.Embedding(num_items, self.embed_dim)
        nn.init.normal_(self.item_embeddings.weight, std=0.02)
        self.temperature = 0.07

    def extract_user_representation(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Runs PREFILL-ONLY pass through LLM backbone and extracts the last token's hidden state.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]

        Returns:
            Pooled user representation vector h of shape [batch_size, embed_dim]
        """
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        # Extract last hidden state: [batch_size, seq_len, hidden_size]
        last_hidden_state = outputs.hidden_states[-1]

        # Extract vector corresponding to the LAST valid non-padding token
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = input_ids.shape[0]
        
        pooled_h = last_hidden_state[torch.arange(batch_size, device=input_ids.device), sequence_lengths]

        # Project to scoring dimension
        user_vector = self.user_proj(pooled_h)
        return user_vector

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_item_ids: Optional[torch.Tensor] = None,
        candidate_item_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass computing catalog-aware dot-product scores and Cross-Entropy ranking loss.

        Args:
            input_ids: Prompt token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            target_item_ids: Ground truth target item index [batch_size]
            candidate_item_ids: Optional subset of candidate item indices [batch_size, num_candidates]

        Returns:
            Tuple of (scores, loss)
        """
        # 1. Extract pooled user vector h via Prefill-Only LLM pass
        user_vector = self.extract_user_representation(input_ids, attention_mask) # [batch_size, embed_dim]

        # Normalize and compute in FP32 precision to prevent FP16 logit overflow and NaN loss
        user_vector_norm = F.normalize(user_vector.float(), p=2, dim=-1)

        if candidate_item_ids is not None:
            cand_embeds = F.normalize(self.item_embeddings(candidate_item_ids).float(), p=2, dim=-1)
            scores = torch.bmm(cand_embeds, user_vector_norm.unsqueeze(-1)).squeeze(-1) / self.temperature
        else:
            all_item_embeds = F.normalize(self.item_embeddings.weight.float(), p=2, dim=-1)
            scores = torch.matmul(user_vector_norm, all_item_embeds.T) / self.temperature

        loss = None
        if target_item_ids is not None:
            loss = F.cross_entropy(scores, target_item_ids)

        return scores, loss


if __name__ == "__main__":
    # Local self-test using a tiny 0.5B model to verify prefill pass and loss calculation
    print("🧪 Running local model architecture test with GPT-2...")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    ranker = GenRecRanker(model_name, num_items=100)

    prompts = [
        "Context: User watching TV. Watch History: The Matrix. Predict target.",
        "Context: User on Mobile. Watch History: Inception. Predict target."
    ]
    inputs = tokenizer(prompts, padding=True, return_tensors="pt")

    targets = torch.tensor([12, 45]) # Target item indices
    scores, loss = ranker(inputs["input_ids"], inputs["attention_mask"], target_item_ids=targets)

    print(f"✅ Prefill-Only Forward Pass Successful!")
    print(f"📊 Output Scores Shape: {scores.shape} (Batch=2, Catalog=100)")
    print(f"📉 Cross-Entropy Ranking Loss: {loss.item():.4f}")
