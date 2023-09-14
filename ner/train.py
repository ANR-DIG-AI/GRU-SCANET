import os
import pickle

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import random
import numpy as np
from module import build_model, EarlyStopping, cal_f1score, DataLoader, load_embeddings, cal_scores

def add_line(file_name, lines=[]):
    with open(file_name, 'a') as f:
        for line in lines:
            f.write(line + '\n')

def run(arg):

    # Create necessary directories
    if not os.path.exists(arg.event_dir):
        os.makedirs(arg.event_dir)

    ckpt_dir = os.path.join(arg.ckpt_dir, arg.model_name)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # Load padded dataset and lookup tables
    with open(arg.padded_dataset_path, 'rb') as fin:
        dataset = pickle.load(fin)

    with open(arg.lookup_path, 'rb') as fin:
        lookup = pickle.load(fin)

    train_sens = dataset['sens_train']
    train_ents = dataset['ents_train']
    val_sens = dataset['sens_val']
    val_ents = dataset['ents_val']
    
    test_sens = dataset['sens_test']
    test_ents = dataset['ents_test']

    word2idx = lookup['word2idx']
    entity2idx = lookup['entity2idx']

    train_data = DataLoader(train_sens, train_ents)
    val_data = DataLoader(val_sens, val_ents)
    test_data = DataLoader(test_sens, test_ents)

    # Set number of vocabularies and entities
    arg.num_vocabs = len(word2idx)
    arg.num_entities = len(entity2idx)

    model = build_model(arg.model_name, arg).to(arg.device)
    optimizer = optim.Adam(model.parameters(), lr=arg.lr, weight_decay=arg.weight_decay)
    lr_decay = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=arg.lr_decay_factor, verbose=True,
                                                    patience=0, min_lr=arg.min_lr)
    # Train
    add_line(file_name='../result/logs/logs.txt', lines=['Model: ' + arg.model_name])
    print('Model: {}\nStart training'.format(arg.model_name))
    finished_batch = 0
    num_parameters = sum(p.numel() for p in model.parameters())
    print("Model's parameters size : ", num_parameters)
    for epoch in range(1, arg.num_epochs + 1):
        model.train()
        for i, (sens, ents) in enumerate(train_data.gen_batch(arg.batch_size)):
            sens = torch.from_numpy(sens).long().to(arg.device)
            ents = torch.from_numpy(ents).long().to(arg.device)
            loss = model.loss(sens, ents)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            finished_batch += sens.size(0)

            if i % 10 == 0:
                pass
                
        # Save model
        torch.save(model.state_dict(), os.path.join(ckpt_dir, 'ckpt_epoch_{:02d}.pt'.format(epoch)))
        if os.path.exists(os.path.join(ckpt_dir, 'ckpt_epoch_{:02d}.pt'.format(epoch - arg.patience - 1))):
            os.remove(os.path.join(ckpt_dir, 'ckpt_epoch_{:02d}.pt'.format(epoch - arg.patience - 1)))

        # Validate per epoch
        model.eval()

        y_true, y_pred = [], []  # true entities, predicted entities
        val_acml_loss = 0 

        for sens, ents in val_data.gen_batch(arg.batch_size * 4, shuffle=False):
            val_size = sens.shape[0]
            sens = torch.from_numpy(sens).long().to(arg.device)
            ents = torch.from_numpy(ents).long().to(arg.device)
            loss = model.loss(sens, ents)
            val_acml_loss += loss.item() * val_size
            
        for sens, ents in test_data.gen_batch(arg.batch_size * 4, shuffle=False):
            val_size = sens.shape[0]
            sens = torch.from_numpy(sens).long().to(arg.device)
            ents = torch.from_numpy(ents).long().to(arg.device)

            _, preds = model(sens, ents)
            targets = ents.cpu().detach().numpy()

            y_true.extend([ent for sen in targets for ent in sen if ent != arg.entity_pad[1]])
            y_pred.extend([ent for sen in preds for ent in sen])
            
        val_loss = val_acml_loss / len(val_data)
        val_f1 = cal_f1score(y_true, y_pred)
        precision, recall, fmeasure = cal_scores(y_true, y_pred)
        _fmeasure = (2*precision*recall)/(precision + recall)
        tmp_output = '[ITER] : precision, recall, F1-Score ' + str((precision, recall, fmeasure))
        print(tmp_output)
        add_line(file_name='../result/logs/logs.txt', lines=[tmp_output])
        lr_decay.step(val_loss)
    print('Done')
