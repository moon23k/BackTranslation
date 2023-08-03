import os, json, argparse
from collections import namedtuple

import torch
from torch.utils.data import DataLoader

from run import Config

from tokenizers.models import WordPiece
from tokenizers import Tokenizer, normalizers
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, Lowercase, StripAccents

from transformers import AutoTokenizer, MarianMTModel




class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        
        with open("data/train.json", 'r') as f:
            self.data = json.load(f)        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return src, trg



def load_dataloader(tokenizer):
    
    def tokenize(tokenizer, batch):
        return tokenizer(
            batch, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        ).input_ids


    def collate_fn(batch):
        src_batch, trg_batch = zip(*batch)

        src_encodings = tokenize(tokenizer, src_batch)
        trg_encodings = tokenize(tokenizer, src_batch)

        return {'src': src_encodings.input_ids,
                'trg': trg_encodings.input_ids}


    return DataLoader(
        Dataset(), 
        batch_size=32, 
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )



def generate_samples():
    samples = []

    args = namedtuple('args')
    config = Config(args)
    
    device = config.device
    mname = config.gen_mname  #'Helsinki-NLP/opus-mt-ko-en'

    model = MarianMTModel.from_pretrained(mname).to(device)
    tokenizer = AutoTokenizer.from_pretrained(mname)
    dataloader = load_dataloader(tokenizer)

    for elem in dataloader:
        input_batch = elem['src'].to(device) 
        label_batch = elem['trg'].tolist()

        sample_batch = model.generate(
            input_tensor, 
            use_cache=True, 
            num_beams=10, 
            num_return_sequences=10
        ).tolist()

        validated = validate_sample(sample, label)
        


    return


def validate_sample(sample, label):
    for beam in sample:
        for pred in sample:
            if pred != label:



    return




def save_data(data_obj, f_name):
    with open(f'data/{f_name}.json', 'w') as f:
        json.dump(data_obj, f)



def main(strategy):

    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-backtranslation', required=True)
    
    args = parser.parse_args()
    assert args.strategy in ['back']

    main(args.strategy)