import pandas as pd
from IPython.display import display
import sys
from tensorflow import keras

from sklearn.model_selection import train_test_split

from proteinbert import OutputType, OutputSpec, FinetuningModelGenerator, load_pretrained_model, finetune, evaluate_by_len, log
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs

from functions import HelperClass

help = HelperClass() # Initiate our helper class from functions.py

train_dict = help.get_dictionary(file_paths=['./data/no-pdb', './data/uniprot'])
print(f'There are {len(train_dict)} sequences') # If sequence_dump_go.pkl exists it will load dump. If not it will donwload sequences from uniprot.
train_set_df = pd.DataFrame.from_dict(train_dict, orient='index').drop_duplicates(subset=['seq'])

def extract_go_terms(row):
  return row['go_terms'].split("|")

train_set_df['label'] = train_set_df.apply(lambda row: extract_go_terms(row), axis=1)

all_terms = [] 

for i in train_set_df["go_terms"]: 
    for j in i.split("|"):
        all_terms.append(j)

BENCHMARK_NAME, output_type = ('go_terms', OutputType(True, 'categorical'))


settings = {
    
    'max_dataset_size': None,
    'max_epochs_per_stage': 1,
    'seq_len': 152,
    'batch_size': 2,
    'final_epoch_seq_len': 152,
    'initial_lr_with_frozen_pretrained_layers': 1e-02,
    'initial_lr_with_all_layers': 1e-04,
    'final_epoch_lr': 1e-05,
    'dropout_rate': 0.5,
    'training_callbacks': [
        keras.callbacks.ReduceLROnPlateau(patience = 1, factor = 0.25, min_lr = 1e-05, verbose = 1),
        keras.callbacks.EarlyStopping(patience = 2, restore_best_weights = True),
    ],
}

def run_benchmark(benchmark_name, pretraining_model_generator, input_encoder, pretraining_model_manipulation_function = None):
    
    print('========== %s ==========' % benchmark_name)  
    
    log('Output type: %s' % output_type)

    train_set_seqs, valid_set_seqs, train_set_labels, valid_set_labels = train_test_split(train_set_df['seq'], train_set_df['label'], test_size = 0.1)

    log(f'{len(train_set_seqs)} training set records, {len(valid_set_seqs)} validation set records, {0} test set records.')
    
    unique_labels = sorted(set(all_terms))
       
    log('%d unique lebels.' % len(unique_labels))
    
    output_spec = OutputSpec(output_type, unique_labels)
    
    model_generator = FinetuningModelGenerator(pretraining_model_generator, output_spec, pretraining_model_manipulation_function = pretraining_model_manipulation_function, dropout_rate = settings['dropout_rate'])

    finetune(model_generator, input_encoder, output_spec, train_set_seqs, train_set_labels, valid_set_seqs, valid_set_labels, \
            seq_len = settings['seq_len'], batch_size = settings['batch_size'], max_epochs_per_stage = settings['max_epochs_per_stage'], \
            lr = settings['initial_lr_with_all_layers'], begin_with_frozen_pretrained_layers = True, lr_with_frozen_pretrained_layers = \
            settings['initial_lr_with_frozen_pretrained_layers'], n_final_epochs = 0, final_seq_len = settings['final_epoch_seq_len'], \
            final_lr = settings['final_epoch_lr'], callbacks = settings['training_callbacks'])
    
    save_data = {
      "samples": [('Training-set', (train_set_seqs, train_set_labels)), ('Validation-set', (valid_set_seqs, valid_set_labels))],
      "model_generator": model_generator,
      "input_encoder": input_encoder,
      "output_spec": output_spec,
      "dataset": dataset,
      "start_seq_len" : settings['seq_len'],
    }

    for dataset_name, dataset in [('Training-set', (train_set_seqs, train_set_labels)), ('Validation-set', (valid_set_seqs, valid_set_labels))]:
        
        log('*** %s performance: ***' % dataset_name)
        results, confusion_matrix = evaluate_by_len(model_generator, input_encoder, output_spec, dataset[0], dataset[1], \
                start_seq_len = settings['seq_len'], start_batch_size = settings['batch_size'])
  
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            display(results)
        
        if confusion_matrix is not None:
            with pd.option_context('display.max_rows', 16, 'display.max_columns', 10):
                log('Confusion matrix:')
                display(confusion_matrix)
                
    return model_generator
print("before pretrained")
        
pretrained_model_generator, input_encoder = load_pretrained_model()
print("before run benchmark")
run_benchmark(BENCHMARK_NAME, pretrained_model_generator, input_encoder, pretraining_model_manipulation_function = \
            get_model_with_hidden_layers_as_outputs)
        
log('Done.')

