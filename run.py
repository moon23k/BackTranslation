import os, argparse, torch

from module.test import Tester
from module.train import Trainer
from module.data import load_dataloader

from tqdm import tqdm
from transformers import (set_seed,
                          T5Config,
                          T5TokenizerFast,
                          T5ForConditionalGeneration)



class Config(object):
    def __init__(self, args):    

        self.task = args.task
        self.mode = args.mode
        self.back_ratio = args.back_ratio
        
        self.src = self.task[:2]
        self.trg = self.task[2:]
        self.tok_path = 'data/tokenizer'
        self.ckpt = f"ckpt/{self.task}_{self.back_ratio}.pt"

        self.clip = 1
        self.lr = 5e-4
        self.max_len = 128
        self.n_epochs = 10
        self.batch_size = 32
        self.iters_to_accumulate = 4
        
        use_cuda = torch.cuda.is_available()
        self.device_type = 'cuda' if use_cuda else 'cpu'

        if self.mode == 'inference':
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if use_cuda else 'cpu')


    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")



def print_model_desc(model):

    def count_params(model):
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return params

    def check_size(model):
        param_size, buffer_size = 0, 0

        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb

    print(f"--- Model Params: {count_params(model):,}")
    print(f"--- Model  Size : {check_size(model):.3f} MB\n")



def init_weights(model):
    initrange = 0.1
    model.encoder.weight.data.uniform_(-initrange, initrange)
    model.decoder.bias.data.zero_()
    model.decoder.weight.data.uniform_(-initrange, initrange)    



def load_model(config):

    model_cfg = T5Config()
    model_cfg.vocab_size = config.vocab_size
    model_cfg.update({'decoder_start_token_id': config.pad_id})
    model = T5ForConditionalGeneration(model_cfg)
    model.apply(init_weights)

    print(f"Model for {config.task.upper()} Translator {config.mode.upper()} has loaded")

    if config.mode != 'train':
        assert os.path.exists(config.ckpt)
        model_state = torch.load(config.ckpt, map_location=config.device)['model_state_dict']        
        model.load_state_dict(model_state)
        print(f"Model States has loaded from {config.ckpt}")

    print_model_desc(model)
    return model.to(config.device)



def inference(config, model, tokenizer):
    model.eval()
    print(f'--- Inference Process Started! ---')
    print('[ Type "quit" on user input to stop the Process ]')
    
    while True:
        input_seq = input('\nUser Input Sequence >> ').lower()

        #End Condition
        if input_seq == 'quit':
            print('\n--- Inference Process has terminated! ---')
            break        

        #convert user input_seq into model input_ids
        input_ids = tokenizer(input_seq)['input_ids']
        output_ids = model.generate(input_ids, 
                                    beam_size=4,
                                    do_sample=True, 
                                    max_new_tokens=config.max_len, 
                                    use_cache=True)
        output_seq = tokenizer.decode(output_ids, skip_special_tokens=True)

        #Print Output Sequence
        print(f"Model Out Sequence >> {output_seq}")       



def train(config, model, tokenizer):
    train_dataloader = load_dataloader(config, tokenizer, 'train')
    valid_dataloader = load_dataloader(config, tokenizer, 'valid')
    trainer = Trainer(config, model, train_dataloader, valid_dataloader)
    trainer.train()



def test(config, model, tokenizer):
    test_dataloader = load_dataloader(config, tokenizer, 'test')
    tester = Tester(config, model, tokenizer, test_dataloader)
    tester.test()    
    return



def main(args):
    set_seed(42)
    config = Config(args)

    tokenizer = T5TokenizerFast.from_pretrained(config.tok_path, model_max_length=config.max_len)
    setattr(config, 'pad_id', tokenizer.pad_token_id)
    setattr(config, 'vocab_size', tokenizer.vocab_size)
    model = load_model(config)

    if config.mode == 'train':
        train(config, model, tokenizer)
        return

    elif config.mode == 'test':
        test(config, model, tokenizer)
        return
    
    elif config.mode == 'inference':
        inference(config, model, tokenizer)
        return



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', required=True)
    parser.add_argument('-mode', required=True)
    parser.add_argument('-back_ratio', required=True)
    
    args = parser.parse_args()
    assert args.task in ['enko', 'koen']
    assert args.mode in ['train', 'test', 'inference']
    assert args.back_ratio in ['zero', 'half', 'full']

    main(args)