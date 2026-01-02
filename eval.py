import argparse
import collections
import json
import os
import re
import string
import torch
import copy

from nltk import sent_tokenize
import numpy as np
from rouge_score import rouge_scorer, scoring
from tqdm import tqdm
import sys
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline
)
from datasets import Dataset

from utils import normalize_answer, get_max_memory, remove_citations

QA_MODEL="gaotianyu1350/roberta-large-squad"
AUTOAIS_MODEL="google/t5_xxl_true_nli_mixture"

global autoais_model, autoais_tokenizer
autoais_model, autoais_tokenizer = None, None


def compute_f1(a_gold, a_pred):
    """Compute F1 score between two strings."""

    def _get_tokens(s):
        if not s:
            return []
        return normalize_answer(s).split()

    gold_toks = _get_tokens(a_gold)
    pred_toks = _get_tokens(a_pred)

    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())

    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def compute_exact(a_gold, a_pred):
    """Check whether two strings are equal up to normalization."""

    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def exact_presence(short_answers, context):
    """Verify if any of the answers is present in the given context.
    Args:
        short_answers: list of short answers to look for in the context
        context: a paragraph to search for short answers
    Returns:
        true if any of the short answers is present in the context
    """

    n_short_answers = [normalize_answer(sa) for sa in short_answers]
    n_context = normalize_answer(context)

    for ans in n_short_answers:
        if ans in n_context:
            return True

    return False


def compute_rouge(data):
    """Main function for rouge scoring.
    If two references are provided,
    the best score is chosen for each instance.
    Args:
        data: requires field `output` and `answer` (or `annotations` for ASQA)
        metrics: list of evaluation metrics
    Returns:
        dictionary representation of rouge scores
    """
    def _rouge_calculation(hypotheses,
                        references1,
                        references2=[],
                        metrics=['rougeLsum']):

        if references2 == []:
            references2 = references1

        scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
        aggregator = scoring.BootstrapAggregator()

        for i in range(len(hypotheses)):
            scores1 = scorer.score(references1[i], hypotheses[i])
            scores2 = scorer.score(references2[i], hypotheses[i])
            if scores1['rougeLsum'].fmeasure > scores2['rougeLsum'].fmeasure:
                aggregator.add_scores(scores1)
            else:
                aggregator.add_scores(scores2)

        scores = {m: [] for m in metrics}

        for m in metrics:
            fmeasure = aggregator.aggregate()[m].mid.fmeasure
            scores[m].append(fmeasure)

        for m in scores:
            scores[m] = 100 * sum(scores[m]) / len(scores[m])

        return scores

    hypotheses = {}
    references1 = {}
    references2 = {}

    for idx, item in enumerate(data):
        hypotheses[idx] = item["output"]
        if "annotations" in item and item['annotations'] is not None: # For ASQA
            references1[idx] = item["annotations"][0]["long_answer"]
            references2[idx] = item["annotations"][1]["long_answer"]
        else:
            references1[idx] = item["answer"]
            references2[idx] = item["answer"]

    h, r1, r2 = [], [], []

    for key in references1:
        h.append(hypotheses[key])
        r1.append(references1[key])

        if references2 is not None:
            r2.append(references2[key])

    h = ['\n'.join(sent_tokenize(text.lower())) for text in h]
    r1 = ['\n'.join(sent_tokenize(text.lower())) for text in r1]
    r2 = ['\n'.join(sent_tokenize(text.lower())) for text in r2]
    scores = _rouge_calculation(h, r1, r2)

    return scores['rougeLsum']


def compute_rouge_per_item(data):
    """
    Compute per-item ROUGE-Lsum f-measure (0-100), selecting the better of two references if provided.
    Note: this differs from `compute_rouge()` which uses BootstrapAggregator for corpus-level scoring.
    """
    scorer = rouge_scorer.RougeScorer(["rougeLsum"], use_stemmer=True)
    per_item = []
    for item in data:
        hypothesis = '\n'.join(sent_tokenize(item.get("output", "").lower()))
        if "annotations" in item and item["annotations"] is not None:
            ref1 = item["annotations"][0]["long_answer"]
            ref2 = item["annotations"][1]["long_answer"]
        else:
            ref1 = item.get("answer", "")
            ref2 = item.get("answer", "")
        ref1 = '\n'.join(sent_tokenize(str(ref1).lower()))
        ref2 = '\n'.join(sent_tokenize(str(ref2).lower()))
        score1 = scorer.score(ref1, hypothesis)["rougeLsum"].fmeasure
        score2 = scorer.score(ref2, hypothesis)["rougeLsum"].fmeasure
        per_item.append(100 * (score1 if score1 > score2 else score2))
    return per_item


def compute_str_em(data):
    """Compute STR-EM metric (only for ASQA)
    Args:
        data: requires field `qa_pairs/short_answers` and `output`
    Returns:
        STR-EM and STR-EM-HIT ()
    """

    if 'qa_pairs' not in data[0] or data[0]['qa_pairs'] is None:
        return 0, 0

    acc = []
    hit = []

    for item in data:
        loc_acc = []
        for qa_pair in item['qa_pairs']:
            loc_acc.append(exact_presence(qa_pair['short_answers'], item["output"]))
        acc.append(np.mean(loc_acc))
        hit.append( int(np.mean(loc_acc) == 1) )

    return 100 * np.mean(acc), 100 * np.mean(hit)


def compute_str_em_per_item(data):
    """Compute per-item STR-EM/STR-HIT (0-100)."""
    per_item = []
    for item in data:
        qa_pairs = item.get("qa_pairs", None)
        if not qa_pairs:
            per_item.append({"str_em": 0.0, "str_hit": 0.0})
            continue
        loc_acc = [exact_presence(qa_pair["short_answers"], item.get("output", "")) for qa_pair in qa_pairs]
        score = float(np.mean(loc_acc)) if len(loc_acc) > 0 else 0.0
        per_item.append({"str_em": 100 * score, "str_hit": 100 * float(score == 1.0)})
    return per_item


def compute_len(data):
    """Compute average length of predictions."""

    res, cntr = 0, 0
    for item in data:
        res += len(item["output"].split())
        cntr += 1
    return res / cntr


def compute_qa(data, return_per_item=False):
    """Compute QA-based accuracy.
    Args:
        data: requires filed `qa_pairs/short_answers` and `output`
    Returns:
        QA metrics (QA-EM, QA-F1, QA-Hit)
    """

    if 'qa_pairs' not in data[0] or data[0]['qa_pairs'] is None:
        logger.warn("Warning: no QA pairs found in data")
        empty = {
            'QA-EM': 0,
            'QA-F1': 0,
            'QA-Hit': 0,
        }
        if return_per_item:
            per_item = [{"QA-EM": 0.0, "QA-F1": 0.0, "QA-Hit": 0.0} for _ in range(len(data))]
            return empty, per_item
        return empty

    # Load model
    logger.info("Loading the RoBERTa-large SQuAD model for QA-based accuracy...")
    qa_pipeline = pipeline("question-answering", model=QA_MODEL, device=0, batch_size=8)
    logger.info("Done")

    # Collect all QA inputs for batched processing
    logger.info("Computing the QA-based accuracy...")
    all_inputs = []
    item_indices = []  # Track which item each input belongs to
    qa_pair_indices = []  # Track which qa_pair within the item

    for item_idx, item in enumerate(data):
        context = item['output'] if len(item['output']) > 0 else " "
        for qa_idx, qa_pair in enumerate(item['qa_pairs']):
            all_inputs.append({
                "question": qa_pair['question'],
                "context": context
            })
            item_indices.append(item_idx)
            qa_pair_indices.append(qa_idx)

    # Run batched inference using Dataset for efficiency
    if len(all_inputs) > 0:
        dataset = Dataset.from_list(all_inputs)
        all_results = []
        for out in tqdm(qa_pipeline(dataset, handle_impossible_answer=True), total=len(all_inputs)):
            all_results.append(out)
    else:
        all_results = []

    # Aggregate results back to per-item metrics
    em, f1, bins = [], [], []
    per_item = []

    # Initialize per-item accumulators
    item_stats = {i: {"loc_em": 0, "loc_f1": 0, "loc_counter": 0} for i in range(len(data))}

    for result_idx, res in enumerate(all_results):
        item_idx = item_indices[result_idx]
        qa_idx = qa_pair_indices[result_idx]
        answers = data[item_idx]["qa_pairs"][qa_idx]["short_answers"]
        prediction = res["answer"]

        item_stats[item_idx]["loc_em"] += max([compute_exact(a, prediction) for a in answers])
        item_stats[item_idx]["loc_f1"] += max([compute_f1(a, prediction) for a in answers])
        item_stats[item_idx]["loc_counter"] += 1

    for item_idx in range(len(data)):
        stats = item_stats[item_idx]
        loc_counter = stats["loc_counter"]
        loc_em = stats["loc_em"]
        loc_f1 = stats["loc_f1"]

        item_em = loc_em / loc_counter if loc_counter > 0 else 0
        item_f1 = loc_f1 / loc_counter if loc_counter > 0 else 0
        item_hit = int(loc_em == loc_counter) if loc_counter > 0 else 0
        em.append(item_em)
        f1.append(item_f1)
        bins.append(item_hit)
        if return_per_item:
            per_item.append({"QA-EM": 100 * item_em, "QA-F1": 100 * item_f1, "QA-Hit": 100 * item_hit})

    agg = {
        'QA-EM': 100 * np.mean(em),
        'QA-F1': 100 * np.mean(f1),
        'QA-Hit': 100 * np.mean(bins)
    }
    if return_per_item:
        return agg, per_item
    return agg


def compute_mauve(data):
    """Compute Mauve score."""

    logger.info("Computing MAUVE...")
    human_data = []
    model_data = []
    for item in data:
        # Remove ending punctuations
        # Remove any new lines
        # Truncate by 100 words
        human_data.append(' '.join((item['question'] + " " + item['answer'].strip()).split()[:100]).rstrip(string.punctuation))
        model_data.append(' '.join((item['question'] + " " + item['output'].strip()).split()[:100]).rstrip(string.punctuation))

    import mauve
    out = mauve.compute_mauve(
        p_text=human_data,
        q_text=model_data,
        device_id=0,
        max_text_length=512,
        verbose=True,
        batch_size=8,
        featurize_model_name="gpt2-large"
    )
    return out.mauve * 100


def _run_nli_autoais(passage, claim):
    """
    Run inference for assessing AIS between a premise and hypothesis.
    Adapted from https://github.com/google-research-datasets/Attributed-QA/blob/main/evaluation.py
    """
    global autoais_model, autoais_tokenizer
    input_text = "premise: {} hypothesis: {}".format(passage, claim)
    input_ids = autoais_tokenizer(input_text, return_tensors="pt").input_ids.to(autoais_model.device)
    with torch.inference_mode():
        outputs = autoais_model.generate(input_ids, max_new_tokens=10)
    result = autoais_tokenizer.decode(outputs[0], skip_special_tokens=True)
    inference = 1 if result == "1" else 0
    return inference


def compute_claims(data, return_per_item=False):
    global autoais_model, autoais_tokenizer
    if autoais_model is None:
        logger.info("Loading AutoAIS model...")
        autoais_model = AutoModelForSeq2SeqLM.from_pretrained(AUTOAIS_MODEL, dtype=torch.bfloat16, max_memory=get_max_memory(), device_map="auto")
        autoais_tokenizer = AutoTokenizer.from_pretrained(AUTOAIS_MODEL, use_fast=False)

    logger.info("Computing claims...")
    scores = []
    per_item = []
    for item in tqdm(data):
        normalized_output = remove_citations(item['output'])
        entail = 0
        claims = item.get("claims", None) or []
        for claim in claims:
            entail += _run_nli_autoais(normalized_output, claim)
        item_score = entail / len(claims) if len(claims) > 0 else 0
        scores.append(item_score)
        if return_per_item:
            per_item.append({"claims_nli": 100 * item_score})
    agg = 100 * np.mean(scores) if len(scores) > 0 else 0
    if return_per_item:
        return agg, per_item
    return agg


def compute_autoais(data,
                    decontext=False,
                    concat=False,
                    qampari=False,
                    at_most_citations=None,
                    return_per_item=False,):
    """
    Compute AutoAIS score.

    Args:
        data: requires field `output` and `docs`
              - docs should be a list of items with fields `title` and `text` (or `phrase` and `sent` for QA-extracted docs)
        citation: check citations and use the corresponding references.
        decontext: decontextualize the output
    """

    global autoais_model, autoais_tokenizer
    if autoais_model is None:
        logger.info("Loading AutoAIS model...")
        autoais_model = AutoModelForSeq2SeqLM.from_pretrained(AUTOAIS_MODEL, dtype=torch.bfloat16, max_memory=get_max_memory(), device_map="auto")
        autoais_tokenizer = AutoTokenizer.from_pretrained(AUTOAIS_MODEL, use_fast=False)

    logger.info(f"Running AutoAIS...")

    def _format_document(doc):
        """Format document for AutoAIS."""

        if "sent" in doc:
            # QA-extracted docs
            return "Title: %s\n%s" % (doc['title'], doc['sent'])
        else:
            return "Title: %s\n%s" % (doc['title'], doc['text'])

    ais_scores = []
    ais_scores_prec = []
    per_item_citation_rec = [0.0 for _ in range(len(data))]
    per_item_citation_prec = [0.0 for _ in range(len(data))]

    sent_total = 0
    sent_mcite = 0
    sent_mcite_support = 0
    sent_mcite_overcite = 0
    autoais_log = []
    for item_idx, item in enumerate(tqdm(data)):
        # Get sentences by using NLTK
        if qampari:
            sents = [item['question'] + " " + x.strip() for x in item['output'].rstrip().rstrip(".").rstrip(",").split(",")]
        else:
            sents = sent_tokenize(item['output'])
        if len(sents) == 0:
            # Keep aggregate behavior (skip empty outputs) but still emit a per-item placeholder.
            per_item_citation_rec[item_idx] = 0.0
            per_item_citation_prec[item_idx] = 0.0
            continue

        target_sents = [remove_citations(sent).strip() for sent in sents]

        entail = 0
        entail_prec = 0
        total_citations = 0
        for sent_id, sent in enumerate(sents):
            target_sent = target_sents[sent_id] # Citation removed and (if opted for) decontextualized
            joint_entail = -1 # Undecided

            # Find references
            ref = [int(r[1:])-1 for r in re.findall(r"\[\d+", sent)] # In text citation id starts from 1
            logger.info(f"For `{sent}`, find citations {ref}")
            if len(ref) == 0:
                # No citations
                joint_entail = 0
            elif any([ref_id >= len(item['docs']) for ref_id in ref]):
                # Citations out of range
                joint_entail = 0
            else:
                if at_most_citations is not None:
                    ref = ref[:at_most_citations]
                total_citations += len(ref)
                joint_passage = '\n'.join([_format_document(item['docs'][psgs_id]) for psgs_id in ref])

            # If not directly rejected by citation format error, calculate the recall score
            if joint_entail == -1: 
                joint_entail = _run_nli_autoais(joint_passage, target_sent)
                autoais_log.append({
                    "question": item['question'],
                    "output": item['output'],
                    "claim": sent,
                    "passage": [joint_passage],
                    "model_type": "NLI",
                    "model_output": joint_entail,
                })

            entail += joint_entail
            if len(ref) > 1:
                sent_mcite += 1

            # calculate the precision score if applicable
            if joint_entail and len(ref) > 1:
                sent_mcite_support += 1
                # Precision check: did the model cite any unnecessary documents?
                for psgs_id in ref:
                    # condition A
                    passage = _format_document(item['docs'][psgs_id]) 
                    nli_result = _run_nli_autoais(passage, target_sent)

                    # condition B
                    if not nli_result:
                        subset_exclude = copy.deepcopy(ref)
                        subset_exclude.remove(psgs_id)
                        passage = '\n'.join([_format_document(item['docs'][pid]) for pid in subset_exclude])
                        nli_result = _run_nli_autoais(passage, target_sent)
                        if nli_result: # psgs_id is not necessary
                            flag = 0
                            sent_mcite_overcite += 1 
                        else:
                            entail_prec += 1
                    else:
                        entail_prec += 1
            else:
                entail_prec += joint_entail 

        sent_total += len(sents)
        item_citation_rec = entail / len(sents)
        item_citation_prec = entail_prec / total_citations if total_citations > 0 else 0
        ais_scores.append(item_citation_rec)
        ais_scores_prec.append(item_citation_prec) # len(sents))
        per_item_citation_rec[item_idx] = 100 * item_citation_rec
        per_item_citation_prec[item_idx] = 100 * item_citation_prec

    if sent_mcite > 0 and sent_mcite_support > 0:
        print("Among all sentences, %.2f%% have multiple citations, among which %.2f%% are supported by the joint set, among which %.2f%% overcite." % (
            100 * sent_mcite / sent_total, 
            100 * sent_mcite_support / sent_mcite, 
            100 * sent_mcite_overcite / sent_mcite_support
        ))

    agg = {
        "citation_rec": 100 * np.mean(ais_scores),
        "citation_prec": 100 * np.mean(ais_scores_prec),
    }
    if return_per_item:
        per_item = [
            {"citation_rec": per_item_citation_rec[i], "citation_prec": per_item_citation_prec[i]}
            for i in range(len(data))
        ]
        return agg, per_item
    return agg


def compute_qampari_f1(data, cot=False):
    prec = []
    rec = []
    rec_top5 = []
    f1 = []
    f1_top5 = []

    num_preds = []
    for item in data:
        if cot:
            if ":" in item['output']:
                o = ':'.join(item['output'].split(":")[1:]) # try to separate the COT part and the answer list part.
            else:
                o = ""
        else:
            o = item['output']
        preds = [normalize_answer(x.strip()) for x in o.rstrip().rstrip(".").rstrip(",").split(",")]
        preds = [p for p in preds if len(p) > 0] # delete empty answers
        num_preds.append(len(preds))
        answers = [[normalize_answer(x) for x in ans] for ans in item['answers']]
        flat_answers = [item for sublist in answers for item in sublist]
        
        prec.append(sum([p in flat_answers for p in preds]) / len(preds) if len(preds) > 0 else 0)
        rec.append(sum([any([x in preds for x in a]) for a in answers]) / len(answers))
        rec_top5.append(min(5, sum([any([x in preds for x in a]) for a in answers])) / min(5, len(answers)))
        if (prec[-1] + rec[-1]) == 0:
            f1.append(0)
        else:
            f1.append(2 * prec[-1] * rec[-1] / (prec[-1] + rec[-1]))
        if (prec[-1] + rec_top5[-1]) == 0:
            f1_top5.append(0) 
        else:
            f1_top5.append(2 * prec[-1] * rec_top5[-1] / (prec[-1] + rec_top5[-1]))

    return {
        "num_preds": np.mean(num_preds),
        "qampari_prec": 100 * np.mean(prec),
        "qampari_rec": 100 * np.mean(rec),
        "qampari_rec_top5": 100 * np.mean(rec_top5),
        "qampari_f1": 100 * np.mean(f1),
        "qampari_f1_top5": 100 * np.mean(f1_top5),
    }


def compute_qampari_f1_per_item(data, cot=False):
    """Compute per-item QAMPARI precision/recall/F1 (0-100) and num_preds."""
    per_item = []
    for item in data:
        if cot:
            if ":" in item.get("output", ""):
                o = ':'.join(item["output"].split(":")[1:])
            else:
                o = ""
        else:
            o = item.get("output", "")

        preds = [normalize_answer(x.strip()) for x in o.rstrip().rstrip(".").rstrip(",").split(",")]
        preds = [p for p in preds if len(p) > 0]
        answers = [[normalize_answer(x) for x in ans] for ans in item.get("answers", [])]
        flat_answers = [x for sublist in answers for x in sublist]

        p = sum([pred in flat_answers for pred in preds]) / len(preds) if len(preds) > 0 else 0
        r = sum([any([x in preds for x in a]) for a in answers]) / len(answers) if len(answers) > 0 else 0
        r_top5 = (
            min(5, sum([any([x in preds for x in a]) for a in answers])) / min(5, len(answers))
            if len(answers) > 0
            else 0
        )
        f1 = 0 if (p + r) == 0 else 2 * p * r / (p + r)
        f1_top5 = 0 if (p + r_top5) == 0 else 2 * p * r_top5 / (p + r_top5)

        per_item.append(
            {
                "num_preds": float(len(preds)),
                "qampari_prec": 100 * p,
                "qampari_rec": 100 * r,
                "qampari_rec_top5": 100 * r_top5,
                "qampari_f1": 100 * f1,
                "qampari_f1_top5": 100 * f1_top5,
            }
        )
    return per_item


def compute_len_per_item(data):
    """Compute per-item output length in words."""
    return [float(len(item.get("output", "").split())) for item in data]


def _resolve_correctness_metric(args, qampari, normalized_data):
    """
    Resolve correctness metric name based on args + dataset, aligned with README recipes.
    Returns one of the explicit metric identifiers accepted by `--correctness_metric`.
    """
    if args.correctness_metric != "auto":
        return args.correctness_metric

    if qampari:
        return "qampari_f1"

    has_qa_pairs = any(("qa_pairs" in item and item["qa_pairs"] is not None) for item in normalized_data)
    if has_qa_pairs:
        return "asqa_qa_f1" if args.qa else "asqa_str_em"

    if args.claims_nli:
        return "eli5_claims_nli"

    return "rougeLsum"


def _get_candidate_tag(args):
    if getattr(args, "candidate_tag", None):
        return args.candidate_tag
    return os.path.splitext(os.path.basename(args.f))[0]


def _get_item_id(item, fallback):
    """Prefer stable item identifiers when available."""
    if isinstance(item, dict):
        for key in ("id", "sample_id"):
            if key in item and item[key] is not None:
                return item[key]
    return fallback

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", type=str, required=True, help="Output file. Should have field `question`, `output`, (ROUGE) `answer`, \
                        (accuracy) `qa_pairs`, (AIS) `docs`")
    parser.add_argument("--no_rouge", action="store_true", help="Do not evaluate ROUGE score")
    parser.add_argument("--qa", action="store_true", help="Use the QA model")
    parser.add_argument("--mauve", action="store_true", help="Use the mauve score model")
    parser.add_argument("--citations", action="store_true", help="Evaluation with citation")
    parser.add_argument("--at_most_citations", type=int, default=3, help="At most take this many documents (mostly for precision)")
    parser.add_argument("--claims_nli", action="store_true", help="Use claims for ELI5")

    # QAMPARI
    parser.add_argument("--cot", action="store_true", help="For QAMPARI, try to find colon and separate the COT and answer listing")

    # Per-sample export (for DPO scoring)
    parser.add_argument(
        "--export_scored_candidates",
        type=str,
        nargs="?",
        const="__AUTO__",
        default=None,
        help="Export per-sample scores as JSONL. If provided without a path, defaults to `<f>.scored_candidates.jsonl`.",
    )
    parser.add_argument(
        "--candidate_tag",
        type=str,
        default=None,
        help="Optional constant tag written into each JSONL row. Defaults to result filename basename.",
    )
    parser.add_argument(
        "--correctness_metric",
        type=str,
        default="auto",
        choices=[
            "auto",
            "qampari_f1",
            "qampari_f1_top5",
            "asqa_str_em",
            "asqa_qa_f1",
            "eli5_claims_nli",
            "rougeLsum",
        ],
        help="Per-sample correctness metric to export (default: auto, aligned with README recipes).",
    )
    parser.add_argument(
        "--export_extra_metrics",
        action="store_true",
        help="If set, export extra per-sample metrics (when available/enabled) in addition to the required fields.",
    )

    args = parser.parse_args()

    export_requested = args.export_scored_candidates is not None
    export_path = None
    if export_requested:
        export_path = args.export_scored_candidates
        if export_path == "__AUTO__":
            export_path = args.f + ".scored_candidates.jsonl"
        if not args.citations:
            raise ValueError("--export_scored_candidates requires --citations to compute per-sample citation scores.")
    elif args.export_extra_metrics:
        raise ValueError("--export_extra_metrics requires --export_scored_candidates.")

    with open(args.f) as f:
        data_with_config = json.load(f)
    data = data_with_config['data'] 

    if "qampari" in args.f:
        args.no_rouge = True
        args.qa = False
        args.mauve = False
        args.decontext = False
        qampari = True
    else:
        qampari = False

    # Truncate by newline and remove on the fly search result
    logger.warning("We remove all the pre/appended space/newlines and we truncate the answer by the first newline.")
    logger.warning("We replace any on the fly search result to standard bracket citation format.")
    for i in range(len(data)):
        data[i]['output'] = data[i]['output'].strip().split("\n")[0]
        data[i]['output'] = data[i]['output'].replace("<|im_end|>", "")


    # Remove all citations for all non-AutoAIS evaluation
    normalized_data = copy.deepcopy(data)
    for i in range(len(normalized_data)):
        normalized_data[i]['output'] = remove_citations(normalized_data[i]['output'])

    result = {}
    result['length'] = compute_len(normalized_data)
    result['str_em'], result['str_hit'] = compute_str_em(normalized_data)

    # Optional per-item metric holders (aligned by index into `data` / `normalized_data`)
    per_item_str = None
    per_item_qampari = None
    per_item_rouge = None
    per_item_qa = None
    per_item_claims = None
    per_item_citations = None
    per_item_length = None

    if export_requested:
        per_item_str = compute_str_em_per_item(normalized_data)
        per_item_length = compute_len_per_item(normalized_data)

    if qampari:
        result.update(compute_qampari_f1(normalized_data, cot=args.cot))
        if export_requested:
            per_item_qampari = compute_qampari_f1_per_item(normalized_data, cot=args.cot)
    if not args.no_rouge:
        result['rougeLsum'] = compute_rouge(normalized_data)
    if args.qa:
        qa_ret = compute_qa(normalized_data, return_per_item=export_requested)
        if isinstance(qa_ret, tuple):
            qa_agg, per_item_qa = qa_ret
            result.update(qa_agg)
        else:
            result.update(qa_ret)
    if args.mauve:
        result['mauve'] = compute_mauve(normalized_data)
    if args.citations: 
        autoais_ret = compute_autoais(
            data,
            qampari=qampari,
            at_most_citations=args.at_most_citations,
            return_per_item=export_requested,
        )
        if isinstance(autoais_ret, tuple):
            autoais_agg, per_item_citations = autoais_ret
            result.update(autoais_agg)
        else:
            result.update(autoais_ret)
    if args.claims_nli:
        claims_ret = compute_claims(
            normalized_data,
            return_per_item=export_requested,
        )
        if isinstance(claims_ret, tuple):
            claims_agg, per_item_claims = claims_ret
            result["claims_nli"] = claims_agg
        else:
            result["claims_nli"] = claims_ret

    if export_requested:
        candidate_tag = _get_candidate_tag(args)
        correctness_metric = _resolve_correctness_metric(args, qampari=qampari, normalized_data=normalized_data)

        # Resolve per-item correctness values
        correctness = [0.0 for _ in range(len(normalized_data))]
        if correctness_metric == "qampari_f1":
            if not qampari or per_item_qampari is None:
                raise ValueError("correctness_metric=qampari_f1 requires a QAMPARI result file.")
            correctness = [x["qampari_f1"] for x in per_item_qampari]
        elif correctness_metric == "qampari_f1_top5":
            if not qampari or per_item_qampari is None:
                raise ValueError("correctness_metric=qampari_f1_top5 requires a QAMPARI result file.")
            correctness = [x["qampari_f1_top5"] for x in per_item_qampari]
        elif correctness_metric == "asqa_str_em":
            if per_item_str is None:
                per_item_str = compute_str_em_per_item(normalized_data)
            correctness = [x["str_em"] for x in per_item_str]
        elif correctness_metric == "asqa_qa_f1":
            if not args.qa:
                raise ValueError("correctness_metric=asqa_qa_f1 requires --qa.")
            if per_item_qa is None:
                qa_agg, per_item_qa = compute_qa(normalized_data, return_per_item=True)
                result.update(qa_agg)
            correctness = [x["QA-F1"] for x in per_item_qa]
        elif correctness_metric == "eli5_claims_nli":
            if not args.claims_nli:
                raise ValueError("correctness_metric=eli5_claims_nli requires --claims_nli.")
            if per_item_claims is None:
                claims_agg, per_item_claims = compute_claims(normalized_data, return_per_item=True)
                result["claims_nli"] = claims_agg
            correctness = [x["claims_nli"] for x in per_item_claims]
        elif correctness_metric == "rougeLsum":
            if per_item_rouge is None:
                per_item_rouge = compute_rouge_per_item(normalized_data)
            correctness = per_item_rouge
        else:
            raise ValueError(f"Unknown correctness_metric: {correctness_metric}")

        if per_item_citations is None:
            raise RuntimeError("Per-item citations were not computed; ensure --citations is set.")

        with open(export_path, "w", encoding="utf-8") as wf:
            for idx, item in enumerate(data):
                row = {
                    "id": _get_item_id(item, idx),
                    "candidate_tag": candidate_tag,
                    "citation_rec": per_item_citations[idx]["citation_rec"],
                    "citation_prec": per_item_citations[idx]["citation_prec"],
                    "correctness": correctness[idx],
                }

                if args.export_extra_metrics:
                    if per_item_length is None:
                        per_item_length = compute_len_per_item(normalized_data)
                    row["length"] = per_item_length[idx]
                    if per_item_str is not None:
                        row.update(per_item_str[idx])
                    if per_item_rouge is None:
                        per_item_rouge = compute_rouge_per_item(normalized_data)
                    if per_item_rouge is not None:
                        row["rougeLsum"] = per_item_rouge[idx]
                    if per_item_qampari is not None:
                        row.update(per_item_qampari[idx])
                    if per_item_qa is not None:
                        row.update(per_item_qa[idx])
                    if per_item_claims is not None:
                        row.update(per_item_claims[idx])

                wf.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(result)
    with open(args.f + ".score", "w") as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    main()
