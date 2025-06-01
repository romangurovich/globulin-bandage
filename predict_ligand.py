from shared.loader import tokenizer
import itertools

# List of all 20 standard amino acids (single-letter codes)
amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

def generate_ligands(start_aa, min_len=2, max_len=4):
    ligands = []
    for length in range(min_len, max_len + 1):
        # Generate all combinations for the rest of the ligand
        for combo in itertools.product(amino_acids, repeat=length-1):
            ligand = start_aa + ''.join(combo)
            ligands.append(ligand)
    return ligands

def display_ligands(scored_sequences):
    print("Top joint predictions and candidate ligands:")
    for score, seq, tokens in scored_sequences[:5]:
        token_strs = [tokenizer.decode([t]).replace(" ", "") for t in tokens]
        print(f"Score: {score:.4f} | Sequence: {seq} | Tokens: {token_strs}")
        for aa in token_strs:
            ligands = generate_ligands(aa, min_len=2, max_len=3)  # 2-3mers for brevity
            print(f"  Ligands starting with {aa}: {ligands[:5]} ...")  # Show first 5 for brevity

def main():
    # Example usage for your top joint predictions:
    scored_sequences = load_scored_sequences()
    display_ligands(scored_sequences)

if __name__ == "__main__":
    main()