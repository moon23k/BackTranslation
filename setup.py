import os, json, torch
import pandas as pd
from tqdm import tqdm
from transformers import (T5TokenizerFast,
                          PreTrainedTokenizerFast,
                          BartForConditionalGeneration)




def load_data(f_name):
    assert os.path.exists(f'data/{f_name}.json')
    with open(f'data/{f_name}.json', 'r') as f:
        data = json.load(f)
    return data



def save_data(en_list, ko_list):
    data = [{'en': en.lower(), 'ko': ko} for en, ko in zip(en_list, ko_list)]
    train, valid, test = data[:-2000], data[-2000:-1000], data[-1000:]

    data_dict = {k:v for k, v in zip(['train', 'valid', 'test'], [train, valid, test])}

    for key, val in data_dict.items():
        with open(f'data/{key}.json', 'w') as f:
            json.dump(val, f)        
        assert os.path.exists(f'data/{key}.json')    
    


#Generate Trg from Src
def generate_data(task, data_list):
    src_generated = []
    src, trg = task[:2], task[2:]
    punctuations = ['.', '?', "!"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mname = f'circulus/kobart-trans-{trg}-{src}-v2'
    model = BartForConditionalGeneration.from_pretrained(mname).to(device)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(mname, model_max_length=128)

    for idx in tqdm(range(0, 10000, 10)):
        trg_batch = data_list[idx: idx+10]
        trg_encodings = tokenizer(trg_batch, padding=True, truncation=True, return_tensors='pt')

        src_preds = model.generate(input_ids=trg_encodings.input_ids.to(device), 
                                   attention_mask=trg_encodings.attention_mask.to(device), 
                                   max_new_tokens=128, use_cache=True)
        src_batch = tokenizer.batch_decode(src_preds, skip_special_tokens=True)

        #Add up punctuation on predictions
        for _src, _trg in zip(src_batch, trg_batch):
            if _trg[-1] in punctuations:
                trg_punc = _trg[-1]
                if _src[-1] not in punctuations or _src[-1] != trg_punc:
                    _src += trg_punc
            
            if task == 'enko':
                src_generated.append({'en': _src.lower(), 'ko': _trg})
            elif task == 'koen':    
                src_generated.append({'en': _trg.lower(), 'ko': _src})

    with open(f'data/{task}_samples.json', 'w') as f:
        json.dump(src_generated, f)        



def train_tokenizer():
    orig_data = load_data('train') + \
                load_data('valid') + \
                load_data('test')
    en_samples = load_data('enko_samples')
    ko_samples = load_data('koen_samples')

    en_data = [elem['en'] for elem in (orig_data + en_samples + ko_samples)]
    ko_data = [elem['ko'] for elem in (orig_data + en_samples + ko_samples)]
    
    tot_data = en_data + ko_data
    
    old_tokenizer = T5TokenizerFast.from_pretrained('t5-small', model_max_length=128)
    new_tokenizer = old_tokenizer.train_new_from_iterator(tot_data, vocab_size=30000)
    new_tokenizer.save_pretrained('data/tokenizer')
    
    del old_tokenizer 
    del new_tokenizer


def main():
    assert os.path.exists('data/dialogue.xlsx')
    
    df = pd.read_excel('data/dialogue.xlsx')
    en_list = df.iloc[:, -1].values.tolist()[::8][:12000]
    ko_list = df.iloc[:, -2].values.tolist()[::8][:12000]

    save_data(en_list, ko_list)
    generate_data('enko', ko_list)
    generate_data('koen', en_list)
    train_tokenizer()



if __name__ == '__main__':
    main()