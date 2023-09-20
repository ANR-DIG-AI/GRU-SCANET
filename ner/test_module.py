import os
import csv
import pickle
from module.data import DataProcessing

import torch
from train import add_line
from module import build_model, EarlyStopping, cal_f1score, DataLoader, load_embeddings, cal_scores
import json

def read_labels():
    filename = '../data/processed/labels.json'
    remove = '../data/processed/'
    output = {}
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
            for key in data:
                output[key.replace(remove,'')] = data[key]
            print(output)
            return output
    except FileNotFoundError:
        return {}
    
def adapter(order=0, value=''):
    expected = {'O': 'O', 'B': 'B'+str(order), 'I': 'I'+str(order)}
    if value in list(expected.values()):
        return value
    return expected[value]

def run_test(order, test_set, arg):
    sens_test, ents_test = DataProcessing.read_csv(test_set)
    
    print('Look up', arg.lookup_path)
    with open(arg.lookup_path, 'rb') as fin:
        lookup = pickle.load(fin)

    word2idx = lookup['word2idx']
    entity2idx = lookup['entity2idx']
    
    # idx2entity = {idx: ent for ent, idx in entity2idx.items()}
    # o_entity = entity2idx['O']
    arg.num_vocabs = len(word2idx)
    arg.num_entities = len(entity2idx)
    print('Count of entities : ', arg.num_entities)
    print(entity2idx)
    test_sens = [ [word2idx[_sen] for _sen in sen] for sen in sens_test] # dataset['sens_test']  
    test_ents = [[entity2idx[adapter(order=order,value=ent)] for ent in ents] for ents in ents_test]
    
    model = build_model(arg.model_name, arg).to(arg.device)

    # Load existed weights
    if os.path.exists(arg.test_ckpt):
        model.load_state_dict(torch.load(arg.test_ckpt))

    y_true, y_pred = [], []

    model.eval()
    for sentence, entity in zip(test_sens, test_ents):
        x = torch.tensor(sentence, dtype=torch.long).unsqueeze(0).to(arg.device)
        y = torch.tensor(entity, dtype=torch.long).unsqueeze(0).to(arg.device)
        _, preds = model(x, y)

        y_true.append(entity)
        y_pred.append(preds[0])
    y_true_flatten = sum(y_true, [])
    y_pred_flatten = sum(y_pred, [])
    precision, recall, fmeasure = cal_scores(y_true_flatten, y_pred_flatten)
    tmp_output = '[ITER] : precision, recall, F1-Score ' + str((precision, recall, fmeasure))
    print(tmp_output)
    add_line(file_name='../result/logs/logs.txt', lines=[tmp_output])
    
