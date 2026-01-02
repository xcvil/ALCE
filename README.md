# Citation-Aware DPO Dataset Builder (ALCE-DPO)

This repo is an **engineering-focused extension** of the ALCE benchmark (Gao et al., EMNLP 2023) for turning *citation-grounded generation runs* into **preference pairs** suitable for **TRL DPO** training.

The goal of this project is to demonstrate an end-to-end practical **evaluation → data → training** pipeline, with attention to **data contracts**, **reproducibility**, and **robust automation**.

---

## TL;DR (what you can do here)

- **Generate multiple candidates** for the same ALCE sample IDs (different configs / seeds / decoding).
- **Export per-sample automatic metrics** (especially citation recall/precision) with `eval.py`.
- **Build TRL-ready DPO pairs** (`prompt/chosen/rejected`) with `make_dpo_dataset.py`.

---

## Core pipeline

```mermaid
flowchart LR
  Run["run.py (candidate generation)"] --> ResultJson["result/*.json (args + data)"]
  ResultJson -->|eval.py --export_scored_candidates| ScoredJsonl["*.scored_candidates.jsonl"]
  ScoredJsonl --> Builder["make_dpo_dataset.py"]
  Builder --> DPO["dpo_train.jsonl (TRL: prompt/chosen/rejected)"]
```

---

## Key engineering decisions

### Stable grouping: `(dataset, id, prompt_hash)`
Candidates are grouped by `(dataset, id, sha1(prompt))` so that `chosen` and `rejected` always share **exactly the same prompt**, even if two runs accidentally differ in prompt text.

### Prompt normalization: rebuild a **shot=0** prompt
Candidate runs can differ in demonstrations (shot, seed). To make DPO pairs well-defined, the builder reconstructs a consistent **shot=0** prompt using `args.prompt_file` + `args.ndoc` and the item’s `docs` / `question`.

### Composable artifacts
Per-item score files (`*.scored_candidates.jsonl`) are cacheable. If missing, the dataset builder can auto-generate them by calling `eval.py`.

---

## Quickstart (end-to-end)

### 0) Install
You need `torch`, `transformers`, `accelerate`, `sentencepiece`, `nltk`, `rouge-score`, `tqdm` (plus optional `mauve-text` if you use `--mauve`).

### 1) Generate candidates
Run multiple configs to create multiple candidate result JSONs:

```bash
python run.py --config configs/asqa/asqa_gpt5_shot2_ndoc10_gtr_default.yaml
python run.py --config configs/asqa/asqa_gpt5_shot2_ndoc10_gtr_summary.yaml
python run.py --config configs/asqa/asqa_gpt5_shot2_ndoc10_gtr_extraction.yaml
```

Each run writes a JSON under `result/` with top-level `{args, data}`.

### 2) Export per-sample metrics (optional precompute)
You can precompute the per-item score JSONL explicitly:

For ASQA, use the following command
```bash
python eval.py --f {path/to/result/file} --citations --qa --mauve --export_scored_candidates
```

For QAMPARI, use the following command
```bash
python eval.py --f {path/to/result/file} --citations --export_scored_candidates
```

For ELI5, use the following command
```bash
python eval.py --f {path/to/result/file} --citations --claims_nli --mauve --export_scored_candidates
```

### 3) Build the TRL DPO dataset
Point the builder at your candidate result files/directories:

```bash
python make_dpo_dataset.py \
  --inputs result_asqa result_qampari result_eli5 \
  --output dpo_train.jsonl \
  --w_correctness 1 --w_citation_rec 1 --w_citation_prec 1
```

If `<result>.scored_candidates.jsonl` is missing, `make_dpo_dataset.py` will call `eval.py` to generate it.

---

## Tools & outputs

### Per-sample score JSONL (`*.scored_candidates.jsonl`)
One line per dataset item:

```json
{"id":"...","candidate_tag":"...","citation_rec":12.3,"citation_prec":45.6,"correctness":78.9}
```

### TRL DPO JSONL (`dpo_train.jsonl`)
One line per preference pair:

```json
{"prompt":"...","chosen":"...","rejected":"...","id":"...","dataset":"..."}
```

---

## Relevant files

- `run.py`: candidate generation (API via Portkey/OpenAI or local HF)
- `eval.py`: ALCE evaluation + **per-sample export**
- `make_dpo_dataset.py`: **preference dataset builder**
- `configs/asqa/`: curated ASQA configs
- `tools/run_all_configs.py`: batch runner with logs

---

## Practical notes

- AutoAIS-style citation scoring is compute-heavy (large T5 NLI model). Use a GPU and consider limiting citations via `--at_most_citations`.
- The DPO dataset produced here is based on **automatic metrics**, not human preferences.

---

## Attribution / citation

This work builds on ALCE and its evaluation methodology:

- Paper: `https://arxiv.org/abs/2305.14627`
- Upstream repo: `https://github.com/princeton-nlp/ALCE`
- Upstream-style README preserved in this repo: `README_ALCE.md`

```bibtex
@inproceedings{gao2023enabling,
  title={Enabling Large Language Models to Generate Text with Citations},
  author={Gao, Tianyu and Yen, Howard and Yu, Jiatong and Chen, Danqi},
  year={2023},
  booktitle={Empirical Methods in Natural Language Processing (EMNLP)},
}
```


