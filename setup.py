import os, yaml, json, argparse, torch
from collections import namedtuple

from run import Config
from module import load_dataloader

from tqdm import tqdm
from transformers import MarianMTModel, AutoTokenizer

from tokenizers.models import BPE
from tokenizers import Tokenizer, normalizers
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, Lowercase, StripAccents




def read_data(f_name):
    with open(f"data/{f_name}", 'r') as f:
        data = json.load(f)
    return data


def create_corpus():
    corpus = []
    f_names = os.listdir('data')
    f_names = [f for f in f_names if 'json' in f]
    
    for f_name in f_names:
        data = read_data(f_name)

        for elem in data:
            corpus.append(elem['src'].lower())
            corpus.append(elem['trg'].lower())

    with open('data/corpus.txt', 'w') as f:
        f.write('\n'.join(corpus))

    print(f"{f_names} have used to create corpus file")



def train_tokenizer():
    corpus_path = f'data/corpus.txt'
    assert os.path.exists(corpus_path)
    
    assert os.path.exists('config.yaml')
    with open('config.yaml', 'r') as f:
        vocab_config = yaml.load(f, Loader=yaml.FullLoader)['vocab']

    tokenizer = Tokenizer(BPE(unk_token=vocab_config['unk_token']))
    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=vocab_config['vocab_size'], 
        special_tokens=[
            vocab_config['pad_token'], 
            vocab_config['unk_token'],
            vocab_config['bos_token'],
            vocab_config['eos_token']
            ]
        )

    tokenizer.train(files=[corpus_path], trainer=trainer)
    tokenizer.save(f"data/tokenizer.json")




def generate_samples(config):
    
    mname = config.mname
    device = config.device
    generate_kwargs = config.generate_kwargs

    tokenizer = AutoTokenizer.from_pretrained(mname)
    model = MarianMTModel.from_pretrained(mname).to(device)
    torch.compile(model)
    model.eval()

    sample_dataloader = load_dataloader(config, tokenizer, 'train')

    samples = []
    with torch.no_grad():
        for batch in tqdm(sample_dataloader):
            
            x = batch['y'].to(device)
            label = tokenizer.batch_decode(x, skip_special_tokens=True)
            label = [x.replace('▁', ' ').strip() for x in label]

            sample = model.generate(x, **generate_kwargs)
            sample = tokenizer.batch_decode(sample, skip_special_tokens=True)
            
            for src, trg in zip(sample, label):
                samples.append({'src': src, 'trg': trg})

    with open(f'data/{config.sampling}_sample.json', 'w') as f:
        json.dump(samples, f)




def main(sampling):

    if sampling != 'none':
        args = namedtuple('args', 'mode sampling search')
        args.mode = 'generate'
        args.sampling = sampling
        args.search = 'greedy'  #set to just default value

        config = Config(args)
        generate_samples(config)

    create_corpus()
    train_tokenizer()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-sampling', default='None', required=True)

    args = parser.parse_args()
    assert args.sampling.lower() in ['none', 'greedy', 'beam', 'topk']

    main(args.sampling)