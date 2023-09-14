class Arguments:
    """
    General settings of arguments
    """

    # Device
    device = 'cpu' # cuda

    # Path
    
    gold_data_dir = '../data/datasets/'
    gold_data_train = 'train.tsv'
    gold_data_train_dev = 'train_dev.tsv'
    gold_data_val = 'devel.tsv'
    gold_data_test = 'test.tsv'
    
    choosen_dataset = 'BC2GM'
    raw_data_dir = '../data/processed/' + choosen_dataset + '/'
    raw_data_train = raw_data_dir + 'train.csv'
    raw_data_val = raw_data_dir + 'devel.csv'
    raw_data_test = raw_data_dir + 'test.csv'
    
    whole_world_corpora = '../data/corpus/to_process/'

    data_dir = '../data/pkl/'
    dataset_path = data_dir + 'dataset.pkl'
    lookup_path = data_dir + 'lookup.pkl'
    padded_dataset_path = data_dir + 'padded_dataset.pkl'

    result_dir = '../result'
    event_dir = result_dir + '/event'
    ckpt_dir = result_dir + '/ckpt'
    embed_dir = result_dir + '/embed'

    # Special tokens and corresponding _indexes
    word_pad = ('<pad>', 0)
    word_oov = ('<oov>', 1)
    entity_pad = ('<p>', 0)
    entity_bos = ('<bos>', 1)
    entity_eos = ('<eos>', 2)

    # Train
    finished_epoch = 0
    # num_epochs = 100
    batch_size = 64
    weight_decay = 0.001
    lr = 1e-3
    min_lr = 5e-5
    lr_decay_factor = 0.95

    # Model Common Part
    num_vocabs = None  # set automatically
    num_entities = None  # set automatically
    embed_dim = 128  # embedding size 
    model_dim = 256

    # Early Stop
    min_delta = 0.
    patience = 6

class TransformerCRFArguments(Arguments):
    model_name = 'biogrut'

    attention_type = 'scaled_dot'
    num_blocks = 1 
    num_heads = 4
    ff_hidden_dim = 512
    dropout_rate = 0.2

    gru_hidden_dim = Arguments.model_dim // 2

    