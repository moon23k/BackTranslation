import numpy as np
import os, yaml, random, argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from models.transformer import Transformer
from models.berts import NMT_BERT, NMT_KoBERT, NMT_MultiBERT

from modules.test import Tester
from modules.train import Trainer
from modules.inference import Translator
from modules.data import load_dataloader
from setup import load_tokenizers


class Config(object):
    def __init__(self, args):    
        with open('configs/model.yaml', 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
            params = params[args.model]
            for p in params.items():
                setattr(self, p[0], p[1])

        self.task = args.task
        self.scheduler = args.scheduler
        self.model_name = args.model
        
        self.unk_idx = 0
        self.pad_idx = 1
        self.bos_idx = 2
        self.eos_idx = 3

        self.clip = 1
        self.n_epochs = 10
        self.batch_size = 16
        self.learning_rate = 1e-4

        if self.task != 'train':
            self.ckpt = f'ckpt/{self.model_name}.pt'


        if self.task == 'inference':
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx, label_smoothing=0.1).to(self.device)

    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")


def set_seed(SEED=42):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    cudnn.benchmark = False
    cudnn.deterministic = True



def load_model(config):
    if config.model_name == 'transformer':
        model = Transformer(config)
    elif config.model_name == 'bert':
        model = NMT_BERT(config)
    elif config.model_name == 'ko_bert':
        model = NMT_KoBERT(config)
    elif config.model_name == 'multi_bert':
        model = NMT_MultiBERT(config)

    if config.task != 'train':
        model_state = torch.load(config.ckpt, map_location=config.device)['model_state_dict']
        model.load_state_dict(model_state)

    return model.to(config.device)


def main(config):
    model = load_model(config)

    if config.task == 'train':
        train_dataloader = load_dataloader(config, 'train')
        valid_dataloader = load_dataloader(config, 'valid')        
        trainer = Trainer(config, model, train_dataloader, valid_dataloader)
        trainer.train()
    
    elif config.task == 'test':
        test_dataloader = load_dataloader(config, 'test')
        src_tokenizer, trg_tokenizer = load_tokenizers()
        tester = Tester(config, model, test_dataloader, trg_tokenizer)
        tester.test()
    
    elif config.task == 'inference':
        src_tokenizer, trg_tokenizer = load_tokenizers()
        translator = Translator(config, model, src_tokenizer, trg_tokenizer)
        translator.translate()
    

if __name__ == '__main__':
    assert os.path.exists(f'data/train.json')
    assert os.path.exists(f'data/valid.json')
    assert os.path.exists(f'data/test.json')

    parser = argparse.ArgumentParser()
    parser.add_argument('-task', type=str, required=True)
    parser.add_argument('-model', type=str, required=True)
    parser.add_argument('-scheduler', type=str, default='constant', required=False)
    
    args = parser.parse_args()
    assert args.task in ['train', 'test', 'inference']
    assert args.model in ['transformer', 'bert', 'multi_bert', 'ko_bert']
 
    set_seed()
    config = Config(args)
    main(config)