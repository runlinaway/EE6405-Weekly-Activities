from rouge_score import rouge_scorer
from nltk import skipgrams
import matplotlib.pyplot as plt
import numpy as np

# Define the reference (gold-standard) summary
reference = """The quick brown fox swiftly jumped over the lazy dog, who lay resting in the sunlit meadow. 
As the fox landed, it dashed toward the dense woods, disappearing into the thick foliage. 
Meanwhile, the dog lazily stretched, unbothered by the commotion. 
The surrounding birds chirped in harmony as if nothing had happened."""

# Define the predicted (machine-generated) summary
predicted = """A fast-moving fox leaped over a sleepy dog resting in the sunlit field. 
Upon landing, the fox ran toward the nearby forest, vanishing into the trees. 
The dog remained still, stretching occasionally, showing no concern for the fox’s presence. 
In the background, birds continued singing peacefully."""

# Initialize the ROUGE scorer (Only ROUGE-N and ROUGE-L)
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Compute ROUGE-N and ROUGE-L scores
scores = scorer.score(reference, predicted)

# Function to compute ROUGE-S using skip-bigrams
def rouge_s(reference, candidate):
    def skip_bigrams(sentence):
        words = sentence.split()
        return set(skipgrams(words, 2, 1))  # Bigram with max skip distance of 1

    cand_bigrams = skip_bigrams(candidate)
    max_f1_score = 0

    ref_bigrams = skip_bigrams(reference)
    common_bigrams = ref_bigrams.intersection(cand_bigrams)
    precision = len(common_bigrams) / len(cand_bigrams) if cand_bigrams else 0
    recall = len(common_bigrams) / len(ref_bigrams) if ref_bigrams else 0  
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

    return precision, recall, f1_score

# Compute ROUGE-S
rouge_s_precision, rouge_s_recall, rouge_s_f1 = rouge_s(reference, predicted)

# Print the results
print("ROUGE Scores:")
for metric, score in scores.items():
    print(f"{metric.upper()}: Precision={score.precision:.4f}, Recall={score.recall:.4f}, F1-Score={score.fmeasure:.4f}")
print(f"ROUGE-S: Precision={rouge_s_precision:.4f}, Recall={rouge_s_recall:.4f}, F1-Score={rouge_s_f1:.4f}")

# Extract values for visualization
rouge_metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'ROUGE-S']
precision_scores = [scores['rouge1'].precision, scores['rouge2'].precision, scores['rougeL'].precision, rouge_s_precision]
recall_scores = [scores['rouge1'].recall, scores['rouge2'].recall, scores['rougeL'].recall, rouge_s_recall]
f1_scores = [scores['rouge1'].fmeasure, scores['rouge2'].fmeasure, scores['rougeL'].fmeasure, rouge_s_f1]

# Set width of bars
bar_width = 0.2  
index = np.arange(len(rouge_metrics))

# Create the bar chart
plt.figure(figsize=(9,5))
plt.bar(index, precision_scores, bar_width, label='Precision', alpha=0.7)
plt.bar(index + bar_width, recall_scores, bar_width, label='Recall', alpha=0.7)
plt.bar(index + 2 * bar_width, f1_scores, bar_width, label='F1-Score', alpha=0.7)

# Labels and formatting
plt.xlabel("ROUGE Metrics")
plt.ylabel("Scores")
plt.title("ROUGE Score Comparison (ROUGE-N, ROUGE-L, ROUGE-S)")
plt.xticks(index + bar_width, rouge_metrics)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Show the plot
plt.show()

# Analysis: Which metric performs best?
metrics_with_rouge_s = {
    'rouge1': scores['rouge1'].fmeasure,
    'rouge2': scores['rouge2'].fmeasure,
    'rougeL': scores['rougeL'].fmeasure,
    'rougeS': rouge_s_f1
}
best_metric = max(metrics_with_rouge_s, key=metrics_with_rouge_s.get)
print(f"\nBest Performing ROUGE Metric: {best_metric.upper()} based on F1-score.")

# Explanation
print("\nAnalysis:")
print("- ROUGE-1 focuses on word overlap and is typically high.")
print("- ROUGE-2 measures bigram overlap, showing how well phrases are retained.")
print("- ROUGE-L evaluates longest common subsequences, reflecting sentence fluency.")
print("- ROUGE-S considers **skipped bigrams**, helping capture sentence meaning even with word reordering.")

print("\nIn this case, the best-performing metric is", best_metric.upper(), "which suggests the predicted summary excels in this area.")



