import pickle

def save_scored_sequences(scored_sequences, filename="artifacts/scored_sequences.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(scored_sequences, f)

def load_scored_sequences(filename="artifacts/scored_sequences.pkl"):
    with open(filename, "rb") as f:
        return pickle.load(f)