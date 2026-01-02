import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from utils import make_demo


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _is_glob_pattern(p: str) -> bool:
    return any(ch in p for ch in ["*", "?", "["])


def _iter_input_files(inputs: List[str]) -> List[str]:
    paths: List[str] = []
    for inp in inputs:
        if _is_glob_pattern(inp):
            paths.extend(glob.glob(inp))
        else:
            paths.append(inp)

    out: List[str] = []
    seen = set()

    def _maybe_add(fp: str):
        fp = os.path.abspath(fp)
        if fp in seen:
            return
        seen.add(fp)
        out.append(fp)

    for p in paths:
        p = os.path.abspath(p)
        if os.path.isdir(p):
            for fp in glob.glob(os.path.join(p, "*.json")):
                base = os.path.basename(fp)
                # Exclusions: aggregate score, per-sample jsonl, small debug runs
                if base.endswith(".score"):
                    continue
                if ".small" in base:
                    continue
                _maybe_add(fp)
        elif os.path.isfile(p):
            _maybe_add(p)
        else:
            # Ignore missing paths; user may provide broad globs
            continue

    return sorted(out)


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _resolve_repo_root() -> str:
    # Assume this script lives in repo root.
    return os.path.dirname(os.path.abspath(__file__))


def _resolve_prompt_file(repo_root: str, prompt_file: str) -> str:
    # In run.py configs, prompt_file is usually relative to repo root (e.g. prompts/qampari_default.json)
    pf = prompt_file
    if not os.path.isabs(pf):
        pf = os.path.join(repo_root, pf)
    return os.path.abspath(pf)


def _dataset_from_args(result_args: dict, fallback_path: str) -> str:
    ds = (result_args or {}).get("dataset_name", None)
    if ds:
        return str(ds)
    # Fallback: prefix before first '-' in filename
    base = os.path.basename(fallback_path)
    return base.split("-")[0] if "-" in base else "unknown"


def _candidate_tag_from_score_rows(score_rows: List[dict], result_path: str) -> str:
    if score_rows:
        tag = score_rows[0].get("candidate_tag", None)
        if tag:
            return str(tag)
    return os.path.splitext(os.path.basename(result_path))[0]


def _ensure_scored_candidates_jsonl(
    *,
    repo_root: str,
    result_path: str,
    dataset_name: str,
    at_most_citations: int,
    cot: bool,
    verbose: bool,
) -> str:
    score_path = result_path + ".scored_candidates.jsonl"
    if os.path.exists(score_path):
        return score_path

    # Auto-generate via eval.py (requires the user to run this script in an env with torch/transformers).
    eval_py = os.path.join(repo_root, "eval.py")
    cmd = [
        sys.executable,
        eval_py,
        "--f",
        result_path,
        "--citations",
        "--export_scored_candidates",
        score_path,
        "--at_most_citations",
        str(at_most_citations),
        "--export_extra_metrics",
    ]

    ds = dataset_name.lower()
    if ds == "asqa":
        cmd.append("--qa")
    elif ds == "eli5":
        cmd.append("--claims_nli")
    elif ds == "qampari":
        if cot:
            cmd.append("--cot")

    if verbose:
        print("[make_dpo_dataset] Generating per-item scores:", " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True)

    if not os.path.exists(score_path):
        raise RuntimeError(f"Expected eval.py to create `{score_path}` but it was not found.")
    return score_path


@dataclass
class Candidate:
    dataset: str
    sample_id: str
    prompt: str
    prompt_hash: str
    output: str
    candidate_tag: str
    metrics: dict
    score: float


def _clean_output_for_training(text: str) -> str:
    # Match eval.py’s cleanup so scores correspond to the exported correctness/citation metrics.
    t = (text or "").replace("<|im_end|>", "").strip()
    t = t.split("\n")[0].strip()
    return t


def _build_shot0_prompt(
    *,
    prompt_data: dict,
    item: dict,
    ndoc: int,
    use_shorter: Optional[str],
) -> str:
    return make_demo(
        item,
        prompt=prompt_data["demo_prompt"],
        ndoc=ndoc,
        doc_prompt=prompt_data.get("doc_prompt", ""),
        instruction=prompt_data.get("instruction", ""),
        use_shorter=use_shorter,
        test=True,
    )


def _extract_required_metrics(score_row: dict) -> Tuple[float, float, float]:
    # Required by the scoring formula
    try:
        cit_rec = float(score_row.get("citation_rec", 0.0))
        cit_prec = float(score_row.get("citation_prec", 0.0))
        corr = float(score_row.get("correctness", 0.0))
    except Exception:
        cit_rec, cit_prec, corr = 0.0, 0.0, 0.0
    return cit_rec, cit_prec, corr


def _score_candidate(
    *,
    cit_rec: float,
    cit_prec: float,
    correctness: float,
    w_citation_rec: float,
    w_citation_prec: float,
    w_correctness: float,
) -> float:
    return (
        w_correctness * correctness
        + w_citation_rec * cit_rec
        + w_citation_prec * cit_prec
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        required=True,
        help="Result JSON files/dirs/globs to include as candidate runs.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dpo_train.jsonl",
        help="Output path for TRL DPO JSONL.",
    )
    parser.add_argument("--w_correctness", type=float, default=1.0)
    parser.add_argument("--w_citation_rec", type=float, default=1.0)
    parser.add_argument("--w_citation_prec", type=float, default=1.0)
    parser.add_argument("--min_score_gap", type=float, default=0.0)
    parser.add_argument("--at_most_citations", type=int, default=3)
    parser.add_argument("--cot", action="store_true", help="Forward --cot to eval.py for QAMPARI when generating missing score files.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    repo_root = _resolve_repo_root()
    result_files = _iter_input_files(args.inputs)
    if args.verbose:
        print(f"[make_dpo_dataset] Found {len(result_files)} candidate result files.", file=sys.stderr)

    candidates: List[Candidate] = []
    missing_score_files = 0

    # Cache prompt files to avoid re-reading for each item
    prompt_cache: Dict[str, dict] = {}

    for result_path in result_files:
        try:
            result_obj = _load_json(result_path)
        except Exception:
            continue

        if not isinstance(result_obj, dict) or "data" not in result_obj:
            continue

        result_args = result_obj.get("args", {}) or {}
        dataset = _dataset_from_args(result_args, result_path)

        # Ensure per-item score JSONL
        score_path = result_path + ".scored_candidates.jsonl"
        if not os.path.exists(score_path):
            # if .scored_candidates.jsonl is not found, generate it
            missing_score_files += 1
            score_path = _ensure_scored_candidates_jsonl(
                repo_root=repo_root,
                result_path=result_path,
                dataset_name=dataset,
                at_most_citations=args.at_most_citations,
                cot=args.cot,
                verbose=args.verbose,
            )

        score_rows = _load_jsonl(score_path)
        score_by_id: Dict[str, dict] = {}
        for row in score_rows:
            sid = row.get("id", None)
            if sid is None:
                continue
            score_by_id[str(sid)] = row
        candidate_tag = _candidate_tag_from_score_rows(score_rows, result_path)

        # Prompt file (shot=0 reconstruction)
        prompt_file = result_args.get("prompt_file", None)
        if not prompt_file:
            raise ValueError(f"Missing args.prompt_file in `{result_path}`; cannot rebuild shot=0 prompt.")
        prompt_file_path = _resolve_prompt_file(repo_root, str(prompt_file))
        if prompt_file_path not in prompt_cache:
            prompt_cache[prompt_file_path] = _load_json(prompt_file_path)
        prompt_data = prompt_cache[prompt_file_path]

        ndoc = int(result_args.get("ndoc", 0) or 0)
        use_shorter = result_args.get("use_shorter", None)

        for item_idx, item in enumerate(result_obj.get("data", [])):
            # ASQA uses `sample_id` instead of `id` at the item level.
            group_id = item.get("id", None)
            if group_id is None:
                group_id = item.get("sample_id", None)
            if group_id is None:
                group_id = item_idx
            group_id = str(group_id)

            lookup_keys: List[str] = []
            for k in (item.get("id", None), item.get("sample_id", None), item_idx):
                if k is None:
                    continue
                lookup_keys.append(str(k))
            matched_key = next((k for k in lookup_keys if k in score_by_id), None)
            if matched_key is None:
                # Score file missing this id (or it was exported with a different id scheme) → skip
                continue
            score_row = score_by_id[matched_key]

            prompt = _build_shot0_prompt(prompt_data=prompt_data, item=item, ndoc=ndoc, use_shorter=use_shorter)
            prompt_hash = _sha1(prompt)

            output_text = _clean_output_for_training(item.get("output", ""))

            cit_rec, cit_prec, corr = _extract_required_metrics(score_row)
            total_score = _score_candidate(
                cit_rec=cit_rec,
                cit_prec=cit_prec,
                correctness=corr,
                w_citation_rec=args.w_citation_rec,
                w_citation_prec=args.w_citation_prec,
                w_correctness=args.w_correctness,
            )

            candidates.append(
                Candidate(
                    dataset=dataset,
                    sample_id=group_id,
                    prompt=prompt,
                    prompt_hash=prompt_hash,
                    output=output_text,
                    candidate_tag=candidate_tag,
                    metrics=score_row,
                    score=total_score,
                )
            )

    if args.verbose and missing_score_files > 0:
        print(f"[make_dpo_dataset] Auto-generated scores for {missing_score_files} files.", file=sys.stderr)

    # Group by (dataset, id, prompt_hash)
    groups: Dict[Tuple[str, str, str], List[Candidate]] = {}
    for c in candidates:
        key = (c.dataset, c.sample_id, c.prompt_hash)
        groups.setdefault(key, []).append(c)

    total_groups = len(groups)
    used = 0
    skipped_no_pair = 0
    skipped_gap = 0

    out_path = args.output
    if not os.path.isabs(out_path):
        out_path = os.path.join(repo_root, out_path)
    out_path = os.path.abspath(out_path)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as wf:
        for (dataset, sid, _ph), cand_list in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
            if len(cand_list) < 2:
                skipped_no_pair += 1
                continue
            cand_list_sorted = sorted(cand_list, key=lambda c: c.score, reverse=True)
            chosen = cand_list_sorted[0]
            rejected = cand_list_sorted[-1]
            if (chosen.score - rejected.score) < args.min_score_gap:
                skipped_gap += 1
                continue

            row = {
                "prompt": chosen.prompt,
                "chosen": chosen.output,
                "rejected": rejected.output,
                # Helpful metadata (safe for TRL; ignored unless you use it)
                "id": sid,
                "dataset": dataset,
                "chosen_tag": chosen.candidate_tag,
                "rejected_tag": rejected.candidate_tag,
                "chosen_score": chosen.score,
                "rejected_score": rejected.score,
                "chosen_metrics": {
                    "citation_rec": chosen.metrics.get("citation_rec"),
                    "citation_prec": chosen.metrics.get("citation_prec"),
                    "correctness": chosen.metrics.get("correctness"),
                },
                "rejected_metrics": {
                    "citation_rec": rejected.metrics.get("citation_rec"),
                    "citation_prec": rejected.metrics.get("citation_prec"),
                    "correctness": rejected.metrics.get("correctness"),
                },
            }
            wf.write(json.dumps(row, ensure_ascii=False) + "\n")
            used += 1

    print(
        json.dumps(
            {
                "output": out_path,
                "num_candidates": len(candidates),
                "num_groups": total_groups,
                "num_pairs_written": used,
                "skipped_no_pair": skipped_no_pair,
                "skipped_gap": skipped_gap,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    """
    python make_dpo_dataset.py \
  --inputs /home/zhengx46/project/ALCE/result_asqa \
  --output /home/zhengx46/project/ALCE/dpo_train_asqa.jsonl \
  --w_correctness 1 --w_citation_rec 1 --w_citation_prec 1 \
  --min_score_gap 100.0
  """
    main()


