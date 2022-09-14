import json
import pandas as pd
import sentencepiece as spm



def load_data(data_name):
    df = pd.read_excel(f'd_name')
    src_list = df.iloc[:, -2].values.tolist()
    trg_list = df.iloc[:, -1].values.tolist()

    return src_list, trg_list



def build_vocab(lang):
    assert os.path.exists('configs/vocab.yaml')
    assert os.path.exists(f'data/{lang}_concat.txt')

    with open('configs/vocab.yaml', 'r') as f:
        vocab_dict = yaml.load(f, Loader=yaml.FullLoader)
    
    opt = f"--input=data/{lang}_concat.txt\
            --model_prefix=data/{lang}_tokenizer\
            --vocab_size={vocab_dict['vocab_size']}\
            --character_coverage={vocab_dict['coverage']}\
            --model_type={vocab_dict['type']}\
            --unk_id={vocab_dict['unk_id']} --unk_piece={vocab_dict['unk_piece']}\
            --pad_id={vocab_dict['pad_id']} --pad_piece={vocab_dict['pad_piece']}\
            --bos_id={vocab_dict['bos_id']} --bos_piece={vocab_dict['bos_piece']}\
            --eos_id={vocab_dict['eos_id']} --eos_piece={vocab_dict['eos_piece']}"

    spm.SentencePieceTrainer.Train(opt)
    os.remove(f'data/{lang}_concat.txt')



def load_tokenizers():
    tokenizers = []
    for lang in ['src', 'trg']:
        tokenizer = spm.SentencePieceProcessor()
        tokenizer.load(f'data/{lang}_tokenizer.model')
        tokenizer.SetEncodeExtraOptions('bos:eos')    
        tokenizers.append(tokenizer)
    return tokenizers


def tokenize_data(src_data, trg_data, src_tokenizer, trg_tokenizer):
    tokenized_data = []
    for src, trg in zip(src_data, trg_data):
        temp_dict = dict()
        
        temp_dict['src'] = src_tokenizer.EncodeAsIds(src)
        temp_dict['trg'] = trg_tokenizer.EncodeAsIds(trg)
        
        tokenized_data.append(temp_dict)

    return tokenized_data



def save_data(data_obj, data_name, format):
    if format == 'json'
        with open(f"data/{data_name}.{format}", 'w') as f:
            json.dump(data_obj)

    elif format == 'txt':
        with open(f"data/{data_name}.{format}", 'w') as f:
            f.write(data_obj)



def main():
    src, trg = [], []
    for d_name in ['1_구어체(1).xlsx', '1_구어체(2).xlsx']:
        assert os.path.exists(f'data/{d_name}')
        _src, _trg = load_data(d_name)
        src.extend(_src)
        trg.extend(_trg)

    save_data(src, 'src_concat', 'txt')
    save_data(trg, 'trg_concat', 'txt')
    build_vocab('src')
    build_vocab('trg')

    src_tokenizer, trg_tokenizer = load_tokenizers()
    data = tokenize_data(src, trg, src_tokenizer, trg_tokenizer)
    
    train, valid, test = data[:-6000], data[-6000:-3000], data[-3000:]

    save_data(train, 'train', 'json')
    save_data(valid, 'valid', 'json')
    save_data(test, 'test', 'json')



if __name__ == '__main__':
    main()