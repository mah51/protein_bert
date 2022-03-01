import os
from typing import Set
from Bio.SeqIO.FastaIO import SimpleFastaParser
import xml.etree.ElementTree as ET
import pickle
import numpy as np

class HelperClass:
  """Helper class"""
  
  def __init__(self):
    """Helper class constructor"""
    self.all_go_terms = None
    self.seq_dict = None
    self.char_to_idx_dict = None
    self.go_to_idx_dict = None

  def get_file_data(self, file_path: str) -> "list[str]":
    """
    Reads a file and returns the data in a list.
    """
    if file_path and os.path.isfile(file_path):
      with open(file_path, 'r') as file:
        return file.readlines();
    else:
      raise NameError('File not found!')

  def split_dataset(self, dataset, split_ratio=0.8):
    """Splits dataset into training and validation set"""
    # For now splitting is done manually to make sure there are training examples of each go_term.
    sorted_by_term = {}
    
    for i in dataset:
      for j in dataset[i]['go_terms']:
        if j in sorted_by_term:
          sorted_by_term[j].append(i)
        else:
          sorted_by_term[j] = [i]

    valid_set = {}
    train_set = dataset

    for i in sorted_by_term:
      dataset[sorted_by_term[i][0]]['is_valid'] = True
    
    return dataset



  def fasta_to_dictionary(self, file_path: str): 
    """
    Reads a fasta file and returns a dictionary with the uniprot_ID as key and dictionary containing the sequence, and as value.
    """
    with open(file_path, 'r') as file:
      seq_dict = {}
      for title, seq in SimpleFastaParser(file):
        seq_dict[title.split('|')[1]] = {"seq": seq, "title_line": title, "encoded_seq": self.encode_seq(seq), "is_valid": False }
    
    return seq_dict


  def get_dictionary(self, file_paths, fasta_file_path='./swissprot_under_150_noDup.fasta'):
    """Reads fasta file and returns dictionary with sequence, go terms & title line."""
    dump_path = "./sequence_dump_go.pkl"
    if os.path.isfile(dump_path):
      with open(dump_path, 'rb') as file:
        return pickle.load(file)
    else:
      seq_dict = self.fasta_to_dictionary(fasta_file_path)
      new_dict = self.read_xml_go_details(seq_dict, file_paths)
      # new_dict = self.encode_go_terms(new_dict)
      self.seq_dict = new_dict
      with open('./sequence_dump_go.pkl', 'wb') as file:
        pickle.dump(self.seq_dict, file)
      self.seq_dict = self.split_dataset(self.seq_dict)
      return self.seq_dict


  def read_xml_go_details(self, seq_dict: dict, file_paths=['./data/uniprot']) -> "dict":

    for file_path in file_paths:
      for i in os.listdir(file_path):
        if i not in seq_dict:
          continue
        root = ET.parse(file_path + '/{}/{}.xml'.format(i, i)).getroot()
        for j in root.findall('*')[0].findall('{http://uniprot.org/uniprot}dbReference'):
          if j.attrib['type'] == 'GO':
            if "go_terms" in seq_dict[i]:
              # seq_dict[i]['go_terms'].append(j.attrib['id'])
              seq_dict[i]['go_terms'] += '|' + j.attrib['id']
            else:
              # seq_dict[i]['go_terms'] = [j.attrib['id']]
              seq_dict[i]['go_terms'] = j.attrib['id']
      return {k: v for k, v in seq_dict.items() if "go_terms" in v} # Only return sequences that have a GO term

   

  def read_go_details(self, dictionary, file_path='data/uniprot'):
    """Gets GO term data for each item in a dictionary"""
    new_dict = {}
    non_exists = []
    for i in dictionary:
      if os.path.isdir(file_path + '/' + i + '/'):
        go_term_path = file_path + '/' + i + '/go_terms.txt'
        if os.path.isfile(go_term_path):
          with open(go_term_path, 'r') as file:
            new_dict[i] = dictionary[i]
            new_dict[i]['go_terms'] = file.readlines()
      else:
        non_exists.append(i)
        
    return new_dict, non_exists


  def encode_seq(self, sequence):

    alphabet = [
      'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', # standard 20 amino acids
      'B', # Aspartic acid OR Asparagine 
      'Z', # Glutamic acid OR Glutamine
      'X', # Any unknown amino acid
      'J', # Leucine OR Isoleucine
      'U', # Selenocysteine only present in some lineages
      'O', # Pyrrolysine used in archaea and bacteria but not present in humans
      ]
    
    if not self.char_to_idx_dict:
       self.char_to_idx_dict = dict((c, i) for i, c in enumerate(alphabet)) # Create a dict, key is the char, value is the index
   
    integer_encoded = [self.char_to_idx_dict[char] for char in sequence] # Which index for each letter
    onehot_encoded = []

    for value in integer_encoded:
        letter = [0 for _ in range(len(alphabet))] # Create a list of zeros
        letter[value] = 1 # At correct index, set to 1
        onehot_encoded.append(letter) # Append to list

    return np.array(onehot_encoded)


  def encode_go_terms(self, seq_dict) -> "list[list[int]]":

    if not self.all_go_terms or not self.go_to_idx_dict:
      self.get_all_go_terms(seq_dict)
    
    for id in seq_dict:
      terms = [self.go_to_idx_dict[i] for i in seq_dict[id]['go_terms']]
      zeroed = [0 for _ in range(len(self.all_go_terms))]
      for i in terms: 
        zeroed[i] = 1
      seq_dict[id]['encoded_go_terms'] =  zeroed
    return seq_dict
      

  def get_all_go_terms(self, seq_dict):
    if not self.all_go_terms:
      self.all_go_terms = set([x for i in seq_dict.values() for x in i['go_terms']]) # Get all unique GO terms
      self.go_to_idx_dict = dict((c, i) for i, c in enumerate(self.all_go_terms)) # Create a dict, key is the go term, value is the index



