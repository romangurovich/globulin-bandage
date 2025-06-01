import sys
import os

# Import predict_nearby_aminos from predict_nearby_aminos
from predict_nearby_aminos import predict_nearby_aminos
# Import generate_ligands and tokenizer from predict_ligand
from predict_ligand import display_ligands
from shared.loader import tokenizer

def main():
    # A mutated hemoglobin beta-globin sequence
    seq = list("MVHLTPVEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH")
    target_index = 6
    num_left_masks = 2
    num_right_masks = 2
    top_k = 3
    scored_sequences = predict_nearby_aminos(seq, target_index, num_left_masks, num_right_masks, top_k)
    display_ligands(scored_sequences)

if __name__ == "__main__":
    main()
