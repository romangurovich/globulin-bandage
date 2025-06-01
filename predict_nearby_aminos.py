import itertools
import torch
from transformers import BertForMaskedLM, BertTokenizer

# Load ProtBERT model and tokenizer
model_name = "Rostlab/prot_bert"
tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
model = BertForMaskedLM.from_pretrained(model_name)
model.eval()

# A mutated hemoglobin beta-globin sequence
# 1. Convert sequence to array of single-letter strings
seq = list("MVHLTPVEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH")

# Specifically, the amino acid glutamic acid (E) is replaced with the amino acid valine at position 6 in beta-globin, written as Glu6Val (E6V)
# 2. Pick a target index (e.g., 6)
target_index = 6

# 3. Add 0 to 2 [MASK] tokens on either side
num_left_masks = 2  # Change as needed (0, 1, or 2)
num_right_masks = 2  # Change as needed (0, 1, or 2)

# Build the new sequence
new_seq = (
    seq[:target_index]
    + ["[MASK]"] * num_left_masks
    + [seq[target_index]]
    + ["[MASK]"] * num_right_masks
    + seq[target_index + 1 :]
)

# 4. Convert back to string separated by spaces
input_text = " ".join(new_seq)

top_k = 3  # Number of candidates per mask


# Tokenize and find mask positions
inputs = tokenizer(input_text, return_tensors="pt")
mask_token_id = tokenizer.mask_token_id
mask_positions = (inputs["input_ids"] == mask_token_id).nonzero(as_tuple=True)[1].tolist()

# Get top-k predictions for each mask
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits[0]
    topk_tokens = []
    for pos in mask_positions:
        probs = logits[pos].softmax(dim=-1)
        topk = torch.topk(probs, top_k)
        topk_tokens.append(topk.indices.tolist())

# Generate all combinations
combinations = list(itertools.product(*topk_tokens))

# Score each combination
scored_sequences = []
for combo in combinations:
    filled_ids = inputs["input_ids"].clone()
    for idx, pos in enumerate(mask_positions):
        filled_ids[0, pos] = combo[idx]
    with torch.no_grad():
        outputs = model(input_ids=filled_ids)
        logits = outputs.logits[0]
        # Sum log-probabilities for the chosen tokens at their positions
        log_probs = 0
        for idx, pos in enumerate(mask_positions):
            token_id = combo[idx]
            log_probs += torch.log_softmax(logits[pos], dim=-1)[token_id].item()
    sequence = tokenizer.decode(filled_ids[0], skip_special_tokens=True)
    scored_sequences.append((log_probs, sequence, combo))

# Sort by score (descending)
scored_sequences.sort(reverse=True, key=lambda x: x[0])

# Print top results
print("Top joint predictions:")
for score, seq, tokens in scored_sequences[:5]:
    token_strs = [tokenizer.decode([t]) for t in tokens]
    print(f"Score: {score:.4f} | Sequence: {seq} | Tokens: {token_strs}")
