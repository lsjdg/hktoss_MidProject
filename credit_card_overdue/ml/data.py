import pandas as pd
import pickle

def load_data():
    DATA_PATH = '../data/modified/'
    df = pd.read_csv(DATA_PATH)
    
