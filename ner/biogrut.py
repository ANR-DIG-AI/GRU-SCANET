from data_processing import DataProcessingMaster
import train
from arguments import TransformerCRFArguments as arg
from train import add_line

datasets = [ 'NCBI-disease'] #, 'BC5CDR-disease', 'BC5CDR-chem','BC4CHEMD',  'BC2GM', 'JNLPBA', 'linnaeus', 's800']
if __name__ == '__main__':
    for dataset in datasets :
        _in = 'Dataset ' + dataset + ' is running ... 0%'
        add_line(file_name='../result/logs/logs.txt', lines=[_in])
        print(_in)
        arg.num_epochs = 2
        arg.choosen_dataset = dataset
        DataProcessingMaster(dataset=dataset)
        train.run(arg)
        out = 'Dataset ' + dataset + ' has finished 100% !'
        print(out)
        add_line(file_name='../result/logs/logs.txt', lines=[out])
        
        
