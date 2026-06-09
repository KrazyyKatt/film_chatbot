"""
evaluation.py — Film Chatbot Evaluation
Metrics: ROUGE-1, ROUGE-2, ROUGE-L, BLEU
"""

import json
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

from rag import initialize_rag

# ── Test questions with reference answers ─────────────────────
TEST_CASES = [
    {
        "question": "Who directed Inception and what is it about?",
        "reference": "Inception was directed by Christopher Nolan. It is a science fiction film about a thief who enters people's dreams to steal secrets from their subconscious."
    },
    {
        "question": "What is the plot of Interstellar?",
        "reference": "Interstellar is a film directed by Christopher Nolan about a group of astronauts who travel through a wormhole near Saturn to find a new home for humanity as Earth becomes uninhabitable."
    },
    {
        "question": "What films did Quentin Tarantino direct?",
        "reference": "Quentin Tarantino directed films such as Pulp Fiction, Reservoir Dogs, Kill Bill, Inglourious Basterds, Django Unchained, The Hateful Eight, and Once Upon a Time in Hollywood."
    },
    {
        "question": "What is The Godfather about?",
        "reference": "The Godfather is a crime film directed by Francis Ford Coppola about the powerful Corleone mafia family, focusing on patriarch Vito Corleone and his son Michael who takes over the family business."
    },
    {
        "question": "Tell me about Christopher Nolan as a director.",
        "reference": "Christopher Nolan is a British-American filmmaker known for directing complex, mind-bending films such as Memento, The Dark Knight trilogy, Inception, Interstellar, Dunkirk, Tenet, and Oppenheimer."
    },
]


def compute_rouge(prediction: str, reference: str) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return {
        "rouge1_f": round(scores["rouge1"].fmeasure, 4),
        "rouge2_f": round(scores["rouge2"].fmeasure, 4),
        "rougeL_f": round(scores["rougeL"].fmeasure, 4),
    }


def compute_bleu(prediction: str, reference: str) -> float:
    ref_tokens = nltk.word_tokenize(reference.lower())
    pred_tokens = nltk.word_tokenize(prediction.lower())
    smoothie = SmoothingFunction().method4
    score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)
    return round(score, 4)


def run_evaluation():
    print("=" * 60)
    print("FILM CHATBOT — EVALUATION REPORT")
    print("=" * 60)

    chain = initialize_rag()
    results = []

    for i, tc in enumerate(TEST_CASES, 1):
        print(f"\n[{i}/{len(TEST_CASES)}] {tc['question']}")
        print("-" * 50)

        result = chain.invoke({"query": tc["question"]})
        prediction = result["result"].strip()

        rouge = compute_rouge(prediction, tc["reference"])
        bleu = compute_bleu(prediction, tc["reference"])

        print(f"ANSWER: {prediction[:300]}...")
        print(f"ROUGE-1: {rouge['rouge1_f']} | ROUGE-2: {rouge['rouge2_f']} | ROUGE-L: {rouge['rougeL_f']} | BLEU: {bleu}")

        results.append({
            "question": tc["question"],
            "reference": tc["reference"],
            "prediction": prediction,
            "rouge1": rouge["rouge1_f"],
            "rouge2": rouge["rouge2_f"],
            "rougeL": rouge["rougeL_f"],
            "bleu": bleu,
        })

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    avg_r1 = sum(r["rouge1"] for r in results) / len(results)
    avg_r2 = sum(r["rouge2"] for r in results) / len(results)
    avg_rl = sum(r["rougeL"] for r in results) / len(results)
    avg_bl = sum(r["bleu"] for r in results) / len(results)
    print(f"Avg ROUGE-1 : {avg_r1:.4f}")
    print(f"Avg ROUGE-2 : {avg_r2:.4f}")
    print(f"Avg ROUGE-L : {avg_rl:.4f}")
    print(f"Avg BLEU    : {avg_bl:.4f}")

    # Save to JSON
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nResults saved to evaluation_results.json")

    return results


if __name__ == "__main__":
    run_evaluation()