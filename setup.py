import os, json
import pandas as pd
from transformers import PreTrainedTokenizerFast



def main():
    processed, ko_list, en_list = [], [], []
    f_names = ['dialogue', 'spoken_1', 'spoken_2']
    tokenizer = PreTrainedTokenizerFast.from_pretrained('circulus/kobart-trans-en-ko-v2')


    for f_name in f_names:
        df = pd.read_excel(f'data/{f_name}.xlsx')
        os.remove(f'data/{f_name}.xlsx')
        
        ko_list.extend(df.iloc[:, -2].values.tolist()[::10])
        en_list.extend(df.iloc[:, -1].values.tolist()[::10])


    for ko_seq, en_seq in zip(ko_list, en_list):
        temp = dict()
        ko_encodings = tokenizer(ko_seq, max_length=300, truncation=True, padding=True)
        en_encodings = tokenizer(en_seq, max_length=300, truncation=True, padding=True)

        temp['ko_ids'] = ko_encodings['input_ids']
        temp['ko_mask'] = ko_encodings['attention_mask']
        temp['en_ids'] = en_encodings['input_ids']
        temp['en_mask'] = en_encodings['attention_mask']
        
        processed.append(temp)
        del temp


    train, valid, test = processed[:-6000], processed[-6000:-3000], processed[-3000:]
    data_dict = {k:v for k, v in zip(['train', 'valid', 'test'], [train, valid, test])}


    for key, val in data_dict.items():
        with open(f'data/{key}.json', 'w') as f:
            json.dump(val, f)        
        assert os.path.exists(f'data/{key}.json')



if __name__ == '__main__':
    main()