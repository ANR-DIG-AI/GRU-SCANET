from data_processing import DataProcessingMaster
import train
from arguments import TransformerCRFArguments as arg
from train import add_line
from test_module import run_test, read_labels
from module import build_model, EarlyStopping, DataLoader, load_embeddings, cal_scores
import torch
import os
import pickle
import argparse
import time
# tests = ['NCBI-disease', 'BC5CDR-disease', 'BC5CDR-chem',
#          'BC4CHEMD',  'BC2GM', 'JNLPBA', 'linnaeus', 's800']
first_time = time.time()
if __name__ == '__main__':
    def arg_manager():
        parser = argparse.ArgumentParser()
        parser.add_argument("--choosen_dataset", type=str,
                            default="processed_1")
        return parser.parse_args()
    args = arg_manager()

    arg.num_epochs = 4
    dataset = args.choosen_dataset
    _in = 'Dataset ' + dataset + ' is running ... 0%'
    print(_in)
    add_line(file_name='../result/logs/logs.txt', lines=[_in])
    arg.choosen_dataset = dataset
    DataProcessingMaster(dataset=dataset)
    # exit()
    train.run(arg)
    out = 'Dataset ' + dataset + ' has finished 100% !'
    print(out)
    add_line(file_name='../result/logs/logs.txt', lines=[out])
    # read file order
    labels = read_labels()
    print('Look up', arg.lookup_path)
    lookup = None
    with open(arg.lookup_path, 'rb') as fin:
        lookup = pickle.load(fin)
    word2idx = lookup['word2idx']
    entity2idx = lookup['entity2idx']
    arg.num_vocabs = len(word2idx)
    arg.num_entities = len(entity2idx)

    out = ' Running time : ' + str(time.time() - first_time) + ' seconds'
    add_line(file_name='../result/logs/logs.txt', lines=[out])
    # exit()
    # model = build_model(arg.model_name, arg).to(arg.device)
    # # Load existed weights
    # if os.path.exists(arg.test_ckpt):
    #     model.load_state_dict(torch.load(arg.test_ckpt))
    # for i in range(len(tests)):
    #     test_set = tests[i]
    #     _label = labels[test_set]
    #     print(test_set)
    #     exit()
    #     test_set = arg.raw_data_test.replace('/LLM/', '/'+test_set+'/')
    #     out = 'Dataset ' + test_set + ' has started 0% !'
    #     print(out)
    #     run_test(_label, test_set, arg, model, lookup)
    #     out = 'Dataset ' + test_set + ' has finished 100% !'
    #     print(out)
    #     add_line(file_name='../result/logs/logs.txt', lines=[out])
