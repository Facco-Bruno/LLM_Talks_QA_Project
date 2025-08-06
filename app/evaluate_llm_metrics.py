import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from fuzzywuzzy import fuzz
import os

INPUT_CSV = "evaluation/llm_evaluation_results.csv"
OUTPUT_CSV = "evaluation/llm_evaluation_metrics.csv"

def compute_bleu(reference, candidate):
    smooth = SmoothingFunction().method1
    ref_tokens = [reference.lower().split()]
    cand_tokens = candidate.lower().split()
    return sentence_bleu(ref_tokens, cand_tokens, smoothing_function=smooth)

def compute_rouge_l(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    score = scorer.score(reference, candidate)
    return score['rougeL'].fmeasure

def compute_fuzzy(reference, candidate):
    return fuzz.token_sort_ratio(reference, candidate) / 100

def evaluate_metrics():
    df = pd.read_csv(INPUT_CSV)

    bleu_scores = []
    rouge_scores = []
    fuzzy_scores = []

    for _, row in df.iterrows():
        ref = str(row["expected_answer"])
        pred = str(row["generated_answer"])

        bleu = compute_bleu(ref, pred)
        rouge = compute_rouge_l(ref, pred)
        fuzzy = compute_fuzzy(ref, pred)

        bleu_scores.append(bleu)
        rouge_scores.append(rouge)
        fuzzy_scores.append(fuzzy)

    df["BLEU"] = bleu_scores
    df["ROUGE-L"] = rouge_scores
    df["Fuzzy"] = fuzzy_scores

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Métricas salvas em: {OUTPUT_CSV}")

if __name__ == "__main__":
    evaluate_metrics()
