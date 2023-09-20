from data_processing import DataProcessingMaster
import train
from arguments import TransformerCRFArguments as arg
from train import add_line
from test_module import run_test, read_labels
from module import build_model, EarlyStopping, cal_f1score, DataLoader, load_embeddings, cal_scores


tests = ['NCBI-disease', 'BC5CDR-disease', 'BC5CDR-chem','BC4CHEMD',  'BC2GM', 'JNLPBA', 'linnaeus', 's800']
if __name__ == '__main__':
    arg.num_epochs = 2
    dataset = 'LLM'
    _in = 'Dataset ' + dataset + ' is running ... 0%'
    print(_in)
    add_line(file_name='../result/logs/logs.txt', lines=[_in])
    arg.choosen_dataset = dataset
    DataProcessingMaster(dataset=dataset)
    train.run(arg)
    out = 'Dataset ' + dataset + ' has finished 100% !'
    print(out)
    add_line(file_name='../result/logs/logs.txt', lines=[out])
    # read file order
    labels = read_labels()
    for i in range(len(tests)) :
        test_set = tests[i]
        _label = labels[test_set]
        test_set = arg.raw_data_test.replace('//','/'+test_set+'/')
        run_test(_label,test_set, arg)
        out = 'Dataset ' + test_set + ' has finished 100% !'
        print(out)
        add_line(file_name='../result/logs/logs.txt', lines=[out])
        
        
