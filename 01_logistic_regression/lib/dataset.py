from lib.config import DATA_DIR
import os.path
import pickle

with open(os.path.join(DATA_DIR, 'data.p'), 'rb') as f:
    RAW_DATA = pickle.load(f)
