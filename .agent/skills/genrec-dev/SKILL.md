---
name: genrec-dev
description: Operating rules, AWS deployment constraints, cost consciousness guidelines, and conventions for the GenRec recommendation system project.
---

# SKILL: GenRec Development — Agent Operating Rules

This skill file governs how the AI agent assists with the GenRec learning project.
Read this before taking ANY action on this project.

---

## 1. Project Context

We are implementing a simplified, learning-oriented version of **GenRec** — Netflix's LLM-native recommendation system — as described in the Netflix Tech Blog. This is a **personal learning project**, not a production system. All decisions should balance learning value against cost.

---

## 2. Cloud & Deployment

- **Cloud Provider**: AWS only.
- **Infrastructure as Code**: AWS CDK (TypeScript preferred, consistent with other projects).
- **Region**: Default to `us-east-1` unless a service is unavailable there.
- **Environment**: Single environment (`dev`). No staging/prod separation unless explicitly asked.

---

## 3. Cost Consciousness

This is a test/learning project. Every AWS resource we create must be justified and cheap.

### Rules:
- **Always** include an approximate monthly cost estimate (in USD) for every resource before creating it.
- Prefer **on-demand** over reserved instances for flexibility, unless savings are dramatic and duration is committed.
- Prefer **Spot Instances** for training workloads where interruption is tolerable.
- Use **smallest viable instance types** that still let us learn the concept.
- Prefer **serverless or managed services** (Lambda, SageMaker Serverless Inference, Fargate) over always-on EC2 unless the tradeoff is explained.
- Prefer **S3** for storing datasets, model artifacts, and checkpoints (cheap and durable).
- **Always enable auto-shutdown / lifecycle policies** on SageMaker notebooks and training jobs.
- Use **free-tier eligible** resources wherever possible (e.g., DynamoDB on-demand free tier, Lambda free tier).
- Tag all resources with `Project=genrec` and `Env=dev` for cost tracking.

### Cost Checkpoint Template (use before every CDK resource):
```
Resource: <resource name>
Type: <AWS service>
Config: <instance type / size / tier>
Approx Monthly Cost: ~$X.XX
Why we need it: <one line>
Free-tier eligible: Yes/No
```

---

## 4. Explain Everything

Since this is a **learning exercise**, the agent must explain every decision:
- **Why** a particular AWS service is chosen over alternatives.
- **What** the resource does in the context of GenRec.
- **How** it maps to what Netflix does vs. what we're doing differently (due to cost/scale constraints).
- Use analogies or architecture diagrams in Markdown where helpful.

Format explanations as `> 💡 **Why**: ...` callout blocks so they are visually distinct from code.

---

## 5. AWS Access Control — CRITICAL

> ⚠️ **The agent has READ-ONLY access to AWS. The user retains full write control.**

### What the agent CAN do:
- Run AWS CLI **read-only** commands: `list-*`, `describe-*`, `get-*`, `show-*`.
- Query costs via `aws ce get-cost-and-usage` (Cost Explorer).
- Inspect existing resources, IAM policies, quotas.
- Run `cdk diff` and `cdk synth` (these are local/dry-run operations that do NOT modify AWS).

### What the agent CANNOT do:
- Run `cdk deploy`, `cdk destroy`, or `cdk bootstrap`.
- Run any AWS CLI command that **creates, modifies, or deletes** resources (e.g., `create-*`, `put-*`, `delete-*`, `update-*`, `deploy`, `apply`).
- Assume or switch IAM roles that grant write access.

### Delegation Protocol:
When a deployment, stack creation, or resource change is needed:
1. Agent will prepare the CDK stack / CLI command.
2. Agent will display it clearly with expected cost and side effects.
3. Agent will say: **"Please run the following command to proceed:"** and stop.
4. User runs the command and reports back the output.
5. Agent continues from there.

---

## 6. CDK Conventions

- All stacks live in `lib/` directory.
- Stack naming: `GenrecXxxStack` (e.g., `GenrecDataStack`, `GenrecTrainingStack`).
- All stacks must be tagged with `{ Project: 'genrec', Env: 'dev' }`.
- Outputs should be exported with descriptive names.
- Use `cdk synth` to validate before handing off for deployment.
- Prefer `RemovalPolicy.DESTROY` on dev resources to avoid orphaned costs.

---

## 7. Git & File Conventions

- All code goes under `/Users/rohang/Documents/my-projects/genrec/`.
- Use clear commit message format: `[phase] what was done` (e.g., `[data] add S3 bucket CDK stack`).
- Do not commit secrets, model weights, or large datasets to git.
- Use `.gitignore` to exclude `cdk.out/`, `node_modules/`, `.env`, `*.ckpt`, `*.pt`.

---

## 8. Learning Checkpoints

At the end of every significant milestone, the agent should produce a short **"What We Learned"** summary:
- What concept we implemented.
- How it maps to the Netflix GenRec paper.
- What corners we cut (and why).
- What to do next.

---

*This skill file should be reviewed and updated as the project evolves.*
