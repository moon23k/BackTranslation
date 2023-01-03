import os, argparse, torch
import torch.nn as nn
import torch.optim as optim

from module.test import Tester
from module.train import Trainer
from module.data import load_dataloader

from transformers import (set_seed,
                          T5Config, 
                          T5TokenizerFast, 
                          T5ForConditionalGeneration)


class Config(object):
    def __init__(self, args):    
        self.task = args.task
        self.mode = args.mode
        self.ckpt = f"ckpt/{self.task}.pt"

        self.clip = 1
        self.n_epochs = 10
        self.batch_size = 16
        self.learning_rate = 5e-5
        self.iters_to_accumulate = 4

        use_cuda = torch.cuda.is_available()
        self.device_type = 'cuda' if use_cuda else 'cpu'

        if self.task == 'inference':
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if use_cuda else 'cpu')


    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")


def load_model(config):
    if config.mode == 'train':
        model = T5ForConditionalGeneration.from_pretrained('t5-small')
        print("Pretrained T5-Small Model for has loaded")
    
    if config.mode != 'train':
        assert os.path.exists(config.ckpt)
        model_config = T5Config.from_pretrained('t5-small')
        model = T5ForConditionalGeneration(model_config)
        print("Initialized T5-Small Model has loaded")

        model_state = torch.load(config.ckpt, map_location=config.device)['model_state_dict']
        model.load_state_dict(model_state)
        print(f"Trained Model states has loaded from {config.ckpt}")

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
    return model.to(config.device)



def inference(model, tokenizer):
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
        output_ids = model.generate(input_ids, beam_size=4)
        output_seq = tokenizer.decode(output_ids, skip_special_tokens=True)

        #Search Output Sequence
        print(f"Model Out Sequence >> {output_seq}")       



def main(args):
    set_seed()
    config = Config(args)
    model = load_model(config)

    setattr(config, 'pad_id', model.config.pad_token_id)
    setattr(config, 'max_length', model.config.max_length)
    setattr(config, 'num_beams', model.config.num_beams)

    if config.task != 'train':
        tokenizer = T5TokenizerFast.from_pretrained('t5-small', model_max_length=300)

    if config.mode == 'train':
        train_dataloader = load_dataloader(config, 'train')
        valid_dataloader = load_dataloader(config, 'valid')
        trainer = Trainer(config, model, train_dataloader, valid_dataloader)
        trainer.train()
    
    elif config.mode == 'test':
        test_dataloader = load_dataloader(config, 'test')
        tester = Tester(config, model, tokenizer, test_dataloader)
        tester.test()
    
    elif config.mode == 'inference':
        inference(model, tokenizer)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', required=True)
    parser.add_argument('-mode', required=True)
    
    args = parser.parse_args()
    assert args.mode in ['translation', 'back_translation']
    assert args.mode in ['train', 'test', 'inference']

    main(args)