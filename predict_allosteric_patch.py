import numpy as np
import matplotlib.pyplot as plt
import torch
from shared.loader import model, tokenizer

# Mutated globulin-b sequence (TPVECS stretch present)
sequence = "MVHLTPVEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH"

# TPVEKS stretch
stretch = "TPVEKS"

# Find the TPVEKS stretch in the sequence
stretch_start = sequence.find(stretch)
if stretch_start == -1:
    raise ValueError(f"Stretch {stretch} not found in sequence.")
stretch_indices = list(range(stretch_start, stretch_start + len(stretch)))

# Identify indices for T, E, and K in the stretch
leg_letters = ['T', 'E', 'K']
leg_indices = [stretch_start + stretch.index(l) for l in leg_letters]

# Space-separate the sequence for ProtBERT
spaced_sequence = " ".join(list(sequence))

# Tokenize sequence
encoded = tokenizer.encode_plus(spaced_sequence, return_tensors='pt', add_special_tokens=True)
input_ids = encoded['input_ids']

# Predict residue-level functional scores
print("Predicting functional scores...")
with torch.no_grad():
    outputs = model(input_ids)
    # outputs[0] shape: (batch, seq_len, vocab_size)
    logits = outputs[0][0]  # (seq_len, vocab_size)
    # Remove special tokens ([CLS] and [SEP])
    logits = logits[1:-1]
    residue_scores = logits.mean(axis=1).numpy()

# Ensure residue_scores and sequence are the same length
min_len = min(len(residue_scores), len(sequence))
if len(residue_scores) != len(sequence):
    print(f"Warning: residue_scores length ({len(residue_scores)}) != sequence length ({len(sequence)}). Truncating to {min_len}.")
    residue_scores = residue_scores[:min_len]
    sequence = sequence[:min_len]
    # Also adjust leg_indices if any are out of bounds
    leg_indices = [idx for idx in leg_indices if idx < min_len]

# Now, residue_scores should align with the sequence
# Define the motif to cover
motif = "MVHLTPVEK"
motif_start = sequence.find(motif)
if motif_start == -1:
    raise ValueError(f"Motif {motif} not found in sequence.")
motif_indices = set(range(motif_start, motif_start + len(motif)))

# Find a patch with high scores that overlaps with the motif
window_size = 9  # Patch size (can adjust)
best_patch = None
best_patch_score = -np.inf
for i in range(len(sequence) - window_size + 1):
    patch_indices = set(range(i, i + window_size))
    # Check if patch overlaps with the motif
    if patch_indices & motif_indices:
        patch_score = residue_scores[i:i+window_size].mean()
        if patch_score > best_patch_score:
            best_patch_score = patch_score
            best_patch = (i, i + window_size)

if best_patch is None:
    raise ValueError(f"No patch found that overlaps with motif {motif}.")

patch_seq = sequence[best_patch[0]:best_patch[1]]
print(f"Predicted allosteric patch overlapping motif {motif} (indices {best_patch[0]}-{best_patch[1]-1}): {patch_seq}")

# Visualization
plt.figure(figsize=(14, 2))
plt.plot(residue_scores, label='Predicted functional score')
plt.axvspan(best_patch[0], best_patch[1]-1, color='orange', alpha=0.3, label='Predicted patch')
plt.axvspan(motif_start, motif_start+len(motif)-1, color='blue', alpha=0.2, label=f'Motif: {motif}')
plt.title(f'ProtBERT Functional Score and Predicted Patch Overlapping {motif}')
plt.xlabel('Residue Index')
plt.ylabel('Score')
plt.legend()
plt.tight_layout()
plt.savefig('allosteric_patch.png', dpi=300)
plt.show() 