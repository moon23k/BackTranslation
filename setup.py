import os, json, argparse, torch
from datasets import load_dataset
from transformers import set_seed
from transformers import T5TokenizerFast
from transformers import T5ForConditionalGeneration

from run import Config, set_training_args
from module.data import load_dataloader
from module.train import Trainer


def process_data(orig_data, tokenizer, volumn=12000):
    min_len = 10 
    max_len = 300
    max_diff = 50
    prefix = 'translate English to German: '

    volumn_cnt = 0
    processed = []

    for elem in orig_data:
        src, trg = elem['en'].lower(), elem['de'].lower()
        src_len, trg_len = len(src), len(trg)

        #define filtering conditions
        min_condition = (src_len >= min_len) & (trg_len >= min_len)
        max_condition = (src_len <= max_len) & (trg_len <= max_len)
        dif_condition = abs(src_len - trg_len) < max_diff

        if max_condition & min_condition & dif_condition:
            temp = dict()

            src_tokenized = tokenizer(prefix + src, max_length=512, truncation=True, padding=True)
            trg_tokenized = tokenizer(trg, max_length=512, truncation=True, padding=True)

            temp['src_ids'] = src_tokenized['input_ids']
            temp['src_attention_mask'] = src_tokenized['attention_mask']
            temp['trg'] = trg_tokenized['input_ids']
            temp['trg_attention_mask'] = src_tokenized['attention_mask']

            processed.append(temp)
            del temp
            
            #End condition
            volumn_cnt += 1
            if volumn_cnt == volumn:
                break

    return processed



def save_data(data_obj):
    #split data into train/valid/test sets
    train, valid, test = data_obj[:-2000], data_obj[-2000:-1000], data_obj[-1000:]
    data_dict = {k:v for k, v in zip(['train', 'valid', 'test'], [train, valid, test])}

    for key, val in data_dict.items():
        with open(f'data/{key}.json', 'w') as f:
            json.dump(val, f)        
        assert os.path.exists(f'data/{key}.json')



def setup_data():
    #Load Original Data & Pretrained Tokenizer
    orig = load_dataset('wmt14', 'de-en', split='train')['translation']
    tokenizer = T5TokenizerFast.from_pretrained('t5-small', model_max_length=512)

    #PreProcess Data
    processed = process_data(orig, tokenizer)
    
    #Save Data
    save_data(processed)
    print('--- Data Setup Completed!')


def generate_synthetics(config, model, tokenizer, train_dataloader):
    return



def setup_back():
    set_seed(42)
    config = Config(task='back_translation')
    model = T5ForConditionalGeneration.from_pretrained('t5-small')

    train_dataloader = load_dataloader(config, 'train')
    valid_dataloader = load_dataloader(config, 'valid')
    
    trainer = Trianer(config, model, train_dataloader, valid_dataloader)
    trainer.train()

    model.generate()    

    return






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', type=str, required=True)

    args = parser.parse_args()
    assert args.task in ['data', 'back_translation']

    if args.task == 'data':
        setup_data()
    else:
        setup_back()