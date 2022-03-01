
import pickle
import pandas as pd
from IPython.display import display

from tensorflow import keras

from sklearn.model_selection import train_test_split

from proteinbert import OutputType, OutputSpec, FinetuningModelGenerator, load_pretrained_model, finetune, evaluate_by_len
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs

from functions import HelperClass

BENCHMARK_NAME = 'short_proteins'


help = HelperClass() # Initiate our helper class from functions.py

train_dict = help.get_dictionary(file_paths=['./data/no-pdb', './data/uniprot'])
print(f'There are {len(train_dict)} sequences') # If sequence_dump_go.pkl exists it will load dump. If not it will donwload sequences from uniprot.

all_terms = [] 


# Make an array of all go terms.
train_set = pd.DataFrame.from_dict(train_dict, orient='index').drop_duplicates(subset=['seq'])
print(train_set.head())


for i in train_set["go_terms"]: 
    for j in i.split("|"):
        all_terms.append(j)



#  Sequence with categorical labels (GO).
OUTPUT_TYPE = OutputType(True, 'categorical')
UNIQUE_LABELS = list(set(all_terms)) # Sets cannot contain duplicate values / ensures there are no repeat go terms.
OUTPUT_SPEC = OutputSpec(OUTPUT_TYPE, UNIQUE_LABELS)

def extract_go_terms(row):
  return row['go_terms'].split("|")

train_set['extracted_go_terms'] = train_set.apply(lambda row: extract_go_terms(row), axis=1)

# Loading the dataset

print(train_set.columns)

X_train, X_valid, y_train, y_valid = train_test_split(train_set['seq'], train_set['extracted_go_terms'], test_size = 0.1, random_state = 0)

print(f'{len(X_train)} training set records, {len(y_train)} validation set records.')


# Loading the pre-trained model and fine-tuning it on the loaded dataset
pretrained_model_generator, input_encoder = load_pretrained_model()

# get_model_with_hidden_layers_as_outputs gives the model output access to the hidden layers (on top of the output)
model_generator = FinetuningModelGenerator(pretrained_model_generator, OUTPUT_SPEC, pretraining_model_manipulation_function = \
        get_model_with_hidden_layers_as_outputs, dropout_rate = 0.5)

training_callbacks = [
    keras.callbacks.ReduceLROnPlateau(patience = 1, factor = 0.25, min_lr = 1e-05, verbose = 1),
    keras.callbacks.EarlyStopping(patience = 2, restore_best_weights = True),
]

finetune(model_generator, input_encoder, OUTPUT_SPEC, X_train, y_train, X_valid, y_valid, \
        seq_len = 150, batch_size = 32, max_epochs_per_stage = 40, lr = 1e-04, begin_with_frozen_pretrained_layers = True, \
        lr_with_frozen_pretrained_layers = 1e-02, n_final_epochs = 1, final_seq_len = 150, final_lr = 1e-05, callbacks = training_callbacks)


# Evaluating the performance on the test-set

results, confusion_matrix = evaluate_by_len(model_generator, input_encoder, OUTPUT_SPEC, X_train, y_train, \
        start_seq_len = 512, start_batch_size = 32)

print('Train-set performance:')
display(results)

print('Confusion matrix:')
display(confusion_matrix)

with open('confusion_matrix_dump.pkl', 'wb') as file:
  pickle.dump(confusion_matrix, file)
