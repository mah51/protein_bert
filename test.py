from functions import HelperClass 
import pandas as pd
from sklearn.model_selection import train_test_split

help = HelperClass() # Initiate our helper class from functions.py

train_dict = help.get_dictionary(file_paths=['./data/no-pdb', './data/uniprot'])
print(f'There are {len(train_dict)} sequences') # If sequence_dump_go.pkl exists it will load dump. If not it will donwload sequences from uniprot.
train_set_df = pd.DataFrame.from_dict(train_dict, orient='index').drop_duplicates(subset=['seq'])

def extract_go_terms(row):
  return row['go_terms'].split("|")

train_set_df['extracted_go_terms'] = train_set_df.apply(lambda row: extract_go_terms(row), axis=1)

print(len(train_set_df))

train_set_df.loc[:, ['extracted_go_terms']].head()

train_set_seqs, valid_set_seqs, train_set_labels, valid_set_labels = train_test_split(train_set_df['seq'], train_set_df['extracted_go_terms'], test_size = 0.1)
