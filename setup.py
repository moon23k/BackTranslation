import os, json, torch, argparse
from run import Config
from module.train import Trainer
from module.data import load_dataloader

from datasets import load_dataset
from transformers import (set_seed, 
                          T5TokenizerFast, 
                          T5ForConditionalGeneration)


def process_data(orig_data, tokenizer, volumn=12000):
    min_len = 10 
    max_len = 300
    max_diff = 50

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

            src_tokenized = tokenizer(src, max_length=300, truncation=True, padding=True)
            trg_tokenized = tokenizer(trg, max_length=300, truncation=True, padding=True)

            temp['src_ids'] = src_tokenized['input_ids']
            temp['src_mask'] = src_tokenized['attention_mask']
            temp['trg'] = trg_tokenized['input_ids']
            temp['trg_mask'] = src_tokenized['attention_mask']

            processed.append(temp)
            del temp
            
            #End condition
            volumn_cnt += 1
            if volumn_cnt == volumn:
                break

    return processed



def setup_data():
    #Load Original Data & Pretrained Tokenizer
    orig = load_dataset('wmt14', 'de-en', split='train')['translation']
    tokenizer = T5TokenizerFast.from_pretrained('t5-small', model_max_length=300)

    #PreProcess Data
    processed = process_data(orig, tokenizer)
    
    #Save Data
    train, valid, test = processed[:-2000], processed[-2000:-1000], processed[-1000:]
    data_dict = {k:v for k, v in zip(['train', 'valid', 'test'], [train, valid, test])}

    for key, val in data_dict.items():
        with open(f'data/{key}.json', 'w') as f:
            json.dump(val, f)        
        assert os.path.exists(f'data/{key}.json')
        
    print('--- Data Setup Completed!')




def setup_back():
    #Train Back Translation Model
    set_seed(42)
    config = Config(task='back_translation')
    model = T5ForConditionalGeneration.from_pretrained('t5-small')

    train_dataloader = load_dataloader(config, 'train')
    valid_dataloader = load_dataloader(config, 'valid')
    
    trainer = Trainer(config, model, train_dataloader, valid_dataloader)
    trainer.train()


    #Generate Synthetic data with trained BackTranslation Model
    assert os.path.exists(f'ckpt/{config.task}.pt')
    model_state = torch.load(config.ckpt, map_location=config.device)['model_state_dict']
    model.load_state_dict(model_state)

    synthetics = []
    for idx, batch in enumerate(train_dataloader):
        temp = dict()

        pred_ids = model.gerneate(batch['trg_ids'].to(config.device))
        pred_mask = torch.ones() #generate mask

        temp['src_ids'] = pred_ids.tolist()
        temp['src_mask'] = pred_mask.tolist()
        temp['trg_ids'] = batch['trg_ids'].tolist()
        temp['trg_mask'] = batch['trg_mask'].tolist()

        synthetics.append(temp)


    with open('data/synthetic.json', 'w') as f:
        json.dump(synthetics, f)
    print('--- Back Translation Setup Completed!')



if __name__ == '__main__':
    print('Setup Process Started!')

    parser = argparse.ArgumentParser()
    parser.add_argument('-task', required=True)
    args = parser.parse_args()

    if args.task == 'data':
        setup_data()
    else:
        setup_back()