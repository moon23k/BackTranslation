import pandas as pd
import os, json, torch
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mname = f'circulus/kobart-trans-{src}-{trg}-v2'
    model = BartForConditionalGeneration.from_pretrained(mname).to(device)
    tokenizer = AutoTokenizer.from_pretrained(mname, model_max_length=128)

    for idx in range(1000):
        trg_batch = [data_list[idx: idx*10]]
        trg_encodings = tokenizer.batch_decode(trg_batch, padding=True, truncation=True)

        src_preds = model.generate(**trg_encodings, use_cache=True)
        src_batch = tokenizer.batch_decode(src_preds, skip_special_tokens=True)

        if task == 'koen':
            src_generated.extend([{'en': en.lower(), 'ko': ko} for en, ko in zip(trg_batch, src_batch)])
        elif task == 'enko':
            src_generated.extend([{'en': en.lower(), 'ko': ko} for en, ko in zip(src_batch, trg_batch)])

    with open(f'data/{src}_samples.json', 'w') as f:
        json.dump(src_generated, f)        



def train_tokenizer():
    orig_data = load_data('train')
    en_samples = load_data('en_samples')
    ko_samples = load_data('ko_samples')

    en_data = [elem['en'] for elem in (orig + en_samples + ko_samples)]
    ko_data = [elem['ko'] for elem in (orig + en_samples + ko_samples)]
    
    tot_data = en_data + ko_data
    
    old_tokenizer = T5TokenizerFast.from_pretrained('t5-small', model_max_length=128)
    new_tokenizer = old_tokenizer.train_new_from_iterator(tot_data, max_vocab_size=30000)
    new_tokenizer.save_pretrained(config.tok_path)
    
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