import pickle
import pandas as pd

def save_data(data_tosave, path, filename):
  pickle.dump(data_tosave, open(path+ '/'+filename+'.p', 'wb'))

def load_saved_data(path, filename):
  data_loaded = pickle.load(open(path+ '/'+filename+'.p', "rb" ))
  return data_loaded

def save_data_csv(data_tosave, path, filename, idx_exist=True):
  data_tosave.to_csv(path+ '/'+filename+'.csv', index=idx_exist, float_format="%.3f")