import pickle
import itertools
from arguments import Arguments as arg
from module import DataProcessing
import os
import pandas as pd

from module import data
from extract_data import ExtractFeatureData

class ReverseData:
    
    def __init__(self, path='') :
        print('Data reversing ...')
        self.path = path
    
    def get_subdirectories(self,path=''):
        subdirectories = []
        for name in os.listdir(path):
            full_path = os.path.join(path, name)
            if os.path.isdir(full_path):
                subdirectories.append(full_path)
        return subdirectories
    
    def get_files(self, path=''):
        files = []
        for name in os.listdir(path):
            full_path = os.path.join(path, name)
            if os.path.isfile(full_path):
                files.append(full_path)
        return files
    
    def read_tsv(self, file_path=''):
        column1 = []
        column2 = []
        output = {'Sentence': [], 'NER': []}
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 1 :
                    first_part = ' '.join(column1)
                    second_part = ' '.join(column2)
                    output['Sentence'].append(first_part)
                    output['NER'].append(second_part)
                    column1 = []
                    column2 = []
                else:
                    column1.append(parts[0])
                    column2.append(parts[1])
        return output

    def replace_in_path_and_create(self, path, old_str, new_str):
        new_path = path.replace(old_str, new_str).replace('.tsv', '.csv')
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        return new_path

    def save_dict_to_csv_with_pandas(self, file_path, data):
        df = pd.DataFrame.from_dict(data)
        df.to_csv(file_path, index=False)
    
    def handle_repositories(self, path=''):
        subdirectories = self.get_subdirectories(path=path)
        for dir in subdirectories :
            files = self.get_files(path=dir)
            for file in files:
                output = self.read_tsv(file_path=file)
                new_path = self.replace_in_path_and_create(file,'datasets', 'processed')
                self.save_dict_to_csv_with_pandas(file_path=new_path, data=output)
                print(new_path)
        return
        
    def run(self):
        self.handle_repositories(path=self.path)
        print('Conversion ended 100% ...')
    
    
class DataProcessingMaster:
    
    def __init__(self, dataset=''):
        arg.choosen_dataset = dataset
        arg.raw_data_dir = '../data/processed/' + arg.choosen_dataset + '/'
        arg.raw_data_train = arg.raw_data_dir + 'train.csv'
        arg.raw_data_val = arg.raw_data_dir + 'devel.csv'
        arg.raw_data_test = arg.raw_data_dir + 'test.csv'
        arg.whole_world_corpora = arg.whole_world_corpora # corpora path
        # ReverseData(path=arg.gold_data_dir).run()
        
        
        # Read external corpora
        whole_world_corpora = [] # 
                    
        # Read original csv files
        sens_train, ents_train = DataProcessing.read_csv(arg.raw_data_train)
        sens_val, ents_val = DataProcessing.read_csv(arg.raw_data_val)
        sens_test, ents_test = DataProcessing.read_csv(arg.raw_data_test) #, is_test=True)

        # load tokenizer BioBert
        whole_world_corpora = ExtractFeatureData(input_path='').get_from_pmc()

        # whole_world_corpora = DataProcessing.read_text_file(arg.whole_world_corpora)
        # print("Nombre total de mots dans le corpus :", len(whole_world_corpora))
        # exit()
        # Build Word-to-Index table
        word_list = list(set(itertools.chain.from_iterable(sens_train + sens_val + sens_test + whole_world_corpora)))
        extra_sign_dict = {sign: idx for sign, idx in [arg.word_pad, arg.word_oov]}
        word2idx = DataProcessing.build_lookup(word_list, **extra_sign_dict)
        print('\n Tokens size : ', len(word_list), '\n')
        # Build Entity-to-Index table
        entity_list = list(set(itertools.chain.from_iterable(ents_train + ents_val + ents_test)))
        extra_sign_dict = {sign: idx for sign, idx in [arg.entity_pad, arg.entity_bos, arg.entity_eos]}
        entity2idx = DataProcessing.build_lookup(entity_list, **extra_sign_dict)
        
        # Convert words and name entities to integer index
        sens_train = [[word2idx[w] for w in sentence] for sentence in sens_train]
        sens_val = [[word2idx[w] for w in sentence] for sentence in sens_val]
        sens_test = [[word2idx[w] for w in sentence] for sentence in sens_test]
        
        ents_train = [[entity2idx[e] for e in ents] for ents in ents_train]
        ents_val = [[entity2idx[e] for e in ents] for ents in ents_val]
        ents_test = [[entity2idx[e] for e in ents] for ents in ents_test]
        
        # Pad sequences
        train_seq_len = max(len(sen) for sen in sens_train)
        sens_train_pad = DataProcessing.pad_sequence(sens_train, train_seq_len, word2idx[arg.word_pad[0]])
        ents_train_pad = DataProcessing.pad_sequence(ents_train, train_seq_len, entity2idx[arg.entity_pad[0]])

        val_seq_len = max(len(sen) for sen in sens_val)
        sens_val_pad = DataProcessing.pad_sequence(sens_val, val_seq_len, word2idx[arg.word_pad[0]])
        ents_val_pad = DataProcessing.pad_sequence(ents_val, val_seq_len, entity2idx[arg.entity_pad[0]])

        test_seq_len = max(len(sen) for sen in sens_test)
        sens_test_pad = DataProcessing.pad_sequence(sens_test, test_seq_len, word2idx[arg.word_pad[0]])
        ents_test_pad = DataProcessing.pad_sequence(ents_test, test_seq_len, entity2idx[arg.entity_pad[0]])
        
        # Store relevant data to file
        lookup = dict()
        lookup['word2idx'] = word2idx
        lookup['entity2idx'] = entity2idx
        
        with open(arg.lookup_path, 'wb') as fout:
            pickle.dump(lookup, fout)

        dataset = dict()
        dataset['sens_train'] = sens_train
        dataset['sens_val'] = sens_val
        dataset['sens_test'] = sens_test
        dataset['ents_train'] = ents_train
        dataset['ents_val'] = ents_val
        dataset['ents_test'] = ents_test

        with open(arg.dataset_path, 'wb') as fout:
            pickle.dump(dataset, fout)

        padded = dict()
        padded['sens_train'] = sens_train_pad
        padded['ents_train'] = ents_train_pad
        padded['sens_val'] = sens_val_pad
        padded['ents_val'] = ents_val_pad
        padded['sens_test'] = sens_test_pad
        padded['ents_test'] = ents_test_pad
        

        with open(arg.padded_dataset_path, 'wb') as fout:
            pickle.dump(padded, fout)

class DataProcessingTrans:
    
    def __init__(self, dataset=''):
        arg.choosen_dataset = dataset
        arg.raw_data_dir = '../data/processed/' + arg.choosen_dataset + '/'
        arg.raw_data_train = arg.raw_data_dir + 'train.csv'
        arg.raw_data_val = arg.raw_data_dir + 'devel.csv'
        arg.raw_data_test = arg.raw_data_dir + 'test.csv'
        # ReverseData(path=arg.gold_data_dir).run()
        # exit()
        # Read original csv files
        sens_train, ents_train = DataProcessing.read_csv(arg.raw_data_train)
        sens_val, ents_val = DataProcessing.read_csv(arg.raw_data_val)
        sens_test, ents_test = DataProcessing.read_csv(arg.raw_data_test, is_test=True)

        # Build Word-to-Index lookup table
        word_list = list(set(itertools.chain.from_iterable(sens_train + sens_val + sens_test)))
        extra_sign_dict = {sign: idx for sign, idx in [arg.word_pad, arg.word_oov]}
        word2idx = DataProcessing.build_lookup(word_list, **extra_sign_dict)
        
        # Build Entity-to-Index lookup table
        entity_list = list(set(itertools.chain.from_iterable(ents_train + ents_val)))
        extra_sign_dict = {sign: idx for sign, idx in [arg.entity_pad, arg.entity_bos, arg.entity_eos]}
        entity2idx = DataProcessing.build_lookup(entity_list, **extra_sign_dict)
        
        # Convert words and name entities to integer index
        sens_train = [[word2idx[w] for w in sentence] for sentence in sens_train]
        sens_val = [[word2idx[w] for w in sentence] for sentence in sens_val]
        sens_test = [[word2idx[w] for w in sentence] for sentence in sens_test]
        ents_train = [[entity2idx[e] for e in ents] for ents in ents_train]
        ents_val = [[entity2idx[e] for e in ents] for ents in ents_val]
        ents_test = [[entity2idx[e] for e in ents] for ents in ents_test]
        
        # Pad sequences
        train_seq_len = max(len(sen) for sen in sens_train)
        sens_train_pad = DataProcessing.pad_sequence(sens_train, train_seq_len, word2idx[arg.word_pad[0]])
        ents_train_pad = DataProcessing.pad_sequence(ents_train, train_seq_len, entity2idx[arg.entity_pad[0]])
        
        val_seq_len = max(len(sen) for sen in sens_val)
        sens_val_pad = DataProcessing.pad_sequence(sens_val, val_seq_len, word2idx[arg.word_pad[0]])
        ents_val_pad = DataProcessing.pad_sequence(ents_val, val_seq_len, entity2idx[arg.entity_pad[0]])

        test_seq_len = max(len(sen) for sen in sens_test)
        sens_test_pad = DataProcessing.pad_sequence(sens_test, test_seq_len, word2idx[arg.word_pad[0]])
        ents_test_pad = DataProcessing.pad_sequence(ents_test, test_seq_len, entity2idx[arg.entity_pad[0]])
        
        # Store relevant data to file
        lookup = dict()
        lookup['word2idx'] = word2idx
        lookup['entity2idx'] = entity2idx
        
        with open(arg.lookup_path, 'wb') as fout:
            pickle.dump(lookup, fout)

        dataset = dict()
        dataset['sens_train'] = sens_train
        dataset['sens_val'] = sens_val
        dataset['sens_test'] = sens_test
        dataset['ents_train'] = ents_train
        dataset['ents_val'] = ents_val
        dataset['ents_test'] = ents_test

        with open(arg.dataset_path, 'wb') as fout:
            pickle.dump(dataset, fout)

        padded = dict()
        padded['sens_train'] = sens_train_pad
        padded['ents_train'] = ents_train_pad
        padded['sens_val'] = sens_val_pad
        padded['ents_val'] = ents_val_pad
        padded['sens_test'] = sens_test_pad
        padded['ents_test'] = ents_test_pad

        with open(arg.padded_dataset_path, 'wb') as fout:
            pickle.dump(padded, fout)

# DataProcessingTrans(dataset='BC2GM')