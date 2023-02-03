import json, torch
from torch.utils.data import DataLoader



class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, split):
        super().__init__()
        
        assert split in ['train', 'valid', 'test']
        self.split = split
        self.src = config.src
        self.trg = config.trg
        self.task = config.task
        self.mode = config.mode
        self.back_ratio = config.back_ratio
        self.data = self.load_data()


    def load_data(self):
        with open(f"data/{self.split}.json", 'r') as f:
            data = json.load(f)

        if self.mode != 'train':
            return data

        else:
            if self.back_ratio == 'zero':
                return data

            with open(f"data/{self.task}_samples.json", 'r') as f:
                samples = json.load(f)
            if self.back_ratio == 'half':
                samples = samples[::2]

            return data + samples


    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, idx):
        src = self.data[idx][f'{self.src}']
        trg = self.data[idx][f'{self.trg}']
        return src, trg



def load_dataloader(config, tokenizer, split):
    
    def collate_fn(batch):
        src_batch, trg_batch = [], []
        for src, trg in batch:
            src_batch.append(src) 
            trg_batch.append(trg)

        src_encodings = tokenizer(src_batch, padding=True, truncation=True, return_tensors='pt')
        trg_encodings = tokenizer(trg_batch, padding=True, truncation=True, return_tensors='pt')

        return {'input_ids': src_encodings.input_ids,
                'attention_mask': src_encodings.attention_mask,
                'labels': trg_encodings.input_ids}


    return DataLoader(Dataset(config, split), 
                      batch_size=config.batch_size, 
                      shuffle=True,
                      collate_fn=collate_fn,
                      num_workers=2,
                      pin_memory=True)