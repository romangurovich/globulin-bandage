import itertools
import torch
from tqdm import tqdm  # Added for progress bar
from shared.loader import tokenizer, model
from shared.savers import save_scored_sequences

def predict_nearby_aminos(
    seq=list("MVHLTPVEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH"),
    target_index=6,
    num_left_masks=2,
    num_right_masks=2,
    top_k=3
):
    # 1. Convert sequence to array of single-letter strings
    # seq is already a list

    # 3. Add 0 to 2 [MASK] tokens on either side
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
    for combo in tqdm(combinations, desc="Scoring combinations"):  # Added tqdm progress bar
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

    save_scored_sequences(scored_sequences)
    return scored_sequences


def main():
    scored_sequences = predict_nearby_aminos()
    save_scored_sequences(scored_sequences)  # Save after computing

    print("Top joint predictions:")
    for score, seq, tokens in scored_sequences[:5]:
        token_strs = [tokenizer.decode([t]) for t in tokens]
        print(f"Score: {score:.4f} | Sequence: {seq} | Tokens: {token_strs}")

    return scored_sequences

if __name__ == "__main__":
    main()
