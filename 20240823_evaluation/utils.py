import pickle

def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)
