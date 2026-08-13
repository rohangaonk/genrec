# GenRec — Our Implementation Plan

> **Source**: [Netflix Tech Blog — GenRec: Towards LLM-Native Recommendation](https://netflixtechblog.com/genrec-towards-llm-native-recommendation-at-netflix-f20be6f643e3)
> **Goal**: Learn by implementing a simplified version of Netflix's GenRec system on AWS.
> **Constraint**: Cost-conscious, learning-first, full user control over deployments.

---

## What Is GenRec? (Plain English)

Traditional recommendation systems at Netflix used **hand-crafted features** fed into gradient-boosted trees or deep neural networks. GenRec replaces that with an **LLM-native approach**: take a large language model, fine-tune it on Netflix-specific data, and use it to *rank* what to show a user next — instead of generating text.

The key insight: **LLMs already understand content semantics, user preferences, and context from pre-training**. Rather than fighting that and building features manually, Netflix leverages it.

---

## GenRec Architecture (What Netflix Does)

```
User History + Context + Item Metadata
          │
          ▼
    [ Verbalizer V ]         ← Converts structured data → natural language prompt
          │
          ▼
    [ Foundation LLM ]       ← Netflix-adapted decoder-only model (Phase 1)
          │
    Pooled hidden state h
          │
          ▼
    [ Scoring Head φ ]       ← dot(h, item_embedding_i) for each item in catalog
          │
          ▼
    Softmax → Ranked List π  ← Top-K recommendations
```

### Training: Two Phases

| Phase | What | Why |
|-------|------|-----|
| **Phase 1** | Continue pre-training on Netflix's own corpus (content metadata, user signals, etc.) | Adapt the LLM to "speak Netflix" — improves MRR by 10–20% |
| **Phase 2** | Fine-tune with ranking-specific loss (cross-entropy over catalog) + LM loss + reward-weighted loss | Teach the model to rank, not just understand — adds 35–50% MRR gain |

### Training Objectives (Multi-loss)
1. **Ranking Loss**: Cross-entropy — teach the model which items the user actually engaged with.
2. **LM Loss**: Next-token-prediction — retain general language understanding, enable explanation generation.
3. **Reward-Weighted Loss**: Scale the ranking loss by reward scores from separate reward models that proxy for long-term satisfaction.

### Serving / Inference
- **Prefill-Only Inference**: No autoregressive decoding. The prompt is processed once, a hidden state is extracted, and scored against all candidate items in parallel. Huge cost win.
- **vLLM + Triton**: Netflix's serving stack.
- **Prefix Caching**: Prompts are structured so shared user-history prefixes can be cached across requests.
- **Context Compaction**: Prompts compressed to 1/3 of original tokens with negligible quality loss.

---

## Our Version: What We Are Building

We will build a simplified but faithful implementation. Netflix runs this at massive scale with proprietary LLMs; we'll do it with open-source models (e.g., Llama 3.1 or Mistral) on a small movie dataset.

### Dataset Options

| Option | Dataset | Size | Notes |
|--------|---------|------|-------|
| **A** | MovieLens 1M | 1M ratings, 6K users, 4K movies | Industry standard, small, fast |
| **B** | MovieLens 25M | 25M ratings, 162K users, 62K movies | Richer, slower |
| **C** | Amazon Product Reviews (Movies & TV) | ~1.7M reviews | Text-rich, closer to Netflix's use case |

> **💬 Discussion**: Option A is best to start. We can verify the pipeline works, then scale to C for richer metadata (closer to what Netflix does with content descriptions).

---

## Implementation Phases

### Phase 0 — Project Setup (1–2 hrs)
- [ ] CDK project init
- [ ] S3 bucket for data/artifacts
- [ ] IAM roles (least-privilege)
- [ ] Basic CI (optional)

**AWS Resources needed**:
- S3 bucket: ~$0.02/month for a few GB
- IAM roles: Free

---

### Phase 1 — Data & Verbalizer (2–4 hrs)

**What Netflix does**: A `Verbalizer V` converts structured user history + item metadata into a natural-language prompt.

**What we'll do**: Write a Python function that converts MovieLens data into prompts like:

```
User has watched: "The Matrix" (Sci-Fi, 1999), "Inception" (Sci-Fi, 2010), "Interstellar" (Sci-Fi, 2014).
Context: Evening, mobile device.
Rate the following for this user: "Dune" (Sci-Fi, 2021)
```

**AWS Resources**:
- SageMaker Processing Job (for large-scale preprocessing): ~$0.50/run on `ml.t3.medium`
- OR just run locally — preferred for learning phase.

---

### Phase 2 — Foundation Model (LLM Backbone)

#### Options & Tradeoffs

| Option | Model | Params | Approach | Approx Cost to Fine-tune |
|--------|-------|--------|----------|--------------------------|
| **A** | Llama 3.1 8B | 8B | SageMaker + QLoRA (4-bit) | ~$2–5/hr on `ml.g5.2xlarge` |
| **B** | Mistral 7B | 7B | SageMaker + QLoRA | ~$2–5/hr on `ml.g5.2xlarge` |
| **C** | GPT-2 Medium | 355M | Full fine-tune possible, CPU-friendly | ~$0.10/hr on `ml.c5.2xlarge` |
| **D** | Skip Phase 1** | — | Use pre-trained HuggingFace model as-is | $0 extra |

> **💬 Discussion Point 1**: Do we want to actually do Phase 1 domain adaptation (Netflix's "adapt to our corpus") or skip straight to Phase 2 ranking fine-tuning? 
>
> - **Skip Phase 1** → Simpler, cheaper, still learns the core ranker idea.
> - **Do Phase 1** → More authentic to the paper, but costs more and takes longer (we'd need a corpus).
>
> **My recommendation**: Skip Phase 1 initially. Use a pre-trained 7B or 8B model as the backbone. We can revisit.

> **💬 Discussion Point 2**: Model size. 
> - GPT-2 Medium is tiny and trains cheaply, but won't showcase LLM's semantic understanding well.
> - Llama 3.1 8B is a good balance — modern, capable, and fits in 24GB VRAM with QLoRA.
> - **Recommendation**: Start with Llama 3.1 8B + QLoRA on a single `g5.2xlarge` spot instance.

---

### Phase 3 — Scoring Head & Training (Core of GenRec)

This is the heart of the project. We:
1. Load a pre-trained LLM.
2. Attach a **scoring head** (small MLP or dot-product) on top of the last hidden state.
3. Train with the multi-objective loss.

**Architecture in code:**
```python
class GenRecModel(nn.Module):
    def __init__(self, llm_backbone, num_items, embed_dim):
        self.backbone = llm_backbone          # Frozen or LoRA-adapted LLM
        self.item_embeddings = nn.Embedding(num_items, embed_dim)
        self.scoring_head = nn.Linear(embed_dim, embed_dim)  # or MLP

    def forward(self, input_ids, attention_mask, item_ids):
        h = self.backbone(input_ids, attention_mask).last_hidden_state[:, -1, :]
        e_i = self.item_embeddings(item_ids)
        scores = torch.einsum('bd,bnd->bn', self.scoring_head(h), e_i)
        return scores
```

**Training**:
```
total_loss = ranking_loss(cross_entropy) + λ₁ × lm_loss + λ₂ × reward_weighted_loss
```

**AWS Resources**:
- SageMaker Training Job on `ml.g5.2xlarge` Spot: ~$1.01/hr (spot), ~$2.03/hr (on-demand)
- Training time estimate: 2–4 hrs for MovieLens 1M
- **Estimated cost per training run: ~$2–8**

---

### Phase 4 — Context Engineering

**What Netflix does**: Treat the context window as a "feature budget."
- Keep high-signal events (long plays, thumbs up).
- Compress binge-watching patterns.
- Drop low-signal events (short hovers).
- Target: 1/3 of original tokens.

**What we'll do**: Implement heuristic compaction rules in the Verbalizer:
- Keep top-N rated movies.
- Summarize genre clusters ("User watches lots of Sci-Fi").
- Drop movies rated < 2 stars.

**Cost**: This is pure Python logic — no AWS cost.

---

### Phase 5 — Inference Serving

**What Netflix does**: vLLM + Triton on GPU, prefill-only mode.

#### Our Options

| Option | Service | Cost | Notes |
|--------|---------|------|-------|
| **A** | SageMaker Real-Time Endpoint (g5.2xlarge) | ~$1.40/hr | Always-on, expensive |
| **B** | SageMaker Serverless Inference | ~$0.0002/sec compute | Pay per request, cold starts |
| **C** | Lambda + EFS (model on EFS) | $0.20/M requests | Only for tiny models (GPT-2) |
| **D** | Local inference (dev only) | $0 | No AWS needed |

> **💬 Discussion Point 3**: For learning, serving is the most expensive phase. 
> - Option D (local) lets us verify the system works without cost.
> - Option B (SageMaker Serverless) is the right answer for a demo endpoint without always-on costs.
> - **Recommendation**: Start with local inference. Add SageMaker Serverless endpoint once we're happy with quality.

---

### Phase 6 — Evaluation

**Netflix's metrics**:
- **MRR (Mean Reciprocal Rank)**: How high does the true positive item rank?
- **Online A/B test**: Homepage engagement, long-term satisfaction.

**Our metrics** (offline only):
- MRR on a held-out test set.
- NDCG@10 (Normalized Discounted Cumulative Gain).
- Compare vs. a simple popularity baseline and a collaborative filtering baseline.

---

## Open Questions / Design Decisions

These are things we need to discuss before building. I've put my recommendation but want your input:

| # | Question | My Recommendation | Decision |
|---|----------|-------------------|----------|
| 1 | Which dataset to start with? | MovieLens 1M (fastest) | ❓ |
| 2 | Skip Phase 1 domain adaptation? | Yes, skip for now | ❓ |
| 3 | Which model backbone? | Llama 3.1 8B + QLoRA | ❓ |
| 4 | Reward model: build one or fake it? | Use a simple heuristic (rating value) | ❓ |
| 5 | Serving: local only or SageMaker endpoint? | Local first, then serverless | ❓ |
| 6 | Add a simple UI demo? | Optional — Flask or Streamlit | ❓ |

---

## Approximate Total Cost Estimate

For the full project, assuming we are frugal:

| Phase | Resource | Cost |
|-------|----------|------|
| Setup | S3, IAM | ~$0 |
| Data prep | Local or SageMaker Processing (1 run) | ~$0–1 |
| Training (3 runs) | `g5.2xlarge` Spot @ $1.01/hr × 4hr × 3 | ~$12 |
| Serving (demo) | SageMaker Serverless (light usage) | ~$1–5 |
| **Total** | | **~$15–20 for full project** |

> This is a rough estimate. We will do an exact cost check before every deploy.

---

## What We Are NOT Doing (And Why)

| Netflix Feature | Why We're Skipping |
|----------------|-------------------|
| Phase 1 domain-specific pre-training on proprietary corpus | No such corpus; would cost $$$$ on GPU clusters |
| Full catalog scoring (millions of items) | Computationally prohibitive; we'll use a candidate set of ~100 items |
| Online A/B testing | No live users |
| Reward models for long-term satisfaction | Complex; we'll proxy with raw ratings |
| vLLM + Triton serving stack | Use HuggingFace's `text-generation-inference` or vLLM locally |

---

## Suggested First Steps

1. ✅ Create this plan (done)
2. ✅ Create SKILL.md (done)
3. 🔜 Discuss open questions above and make decisions
4. 🔜 Initialize CDK project in `genrec/`
5. 🔜 Download MovieLens 1M and write the Verbalizer
6. 🔜 Prototype the scoring head + training loop locally (no AWS yet)

---

*Last updated: 2026-08-10. Evolves as we build.*
