import os, json, torch
from torch.utils.data import DataLoader




class Dataset(torch.utils.data.Dataset):

    def __init__(self, config, tokenizer, split):
        super().__init__()
        
        self.tokenizer = tokenizer
        self.data = self.load_data(split)


    def load_data(self):
        with open(f"data/{self.split}.json", 'r') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, idx):
        src = self.data[idx][f'{self.src}']
        trg = self.data[idx][f'{self.trg}']
        return src, trg



def load_dataloader(config, tokenizer, split):

    def tokenize(self, batch):
        return tokenizer(
            batch, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        )


    def collate_fn(batch):
        src_batch, trg_batch = zip(*batch)
        src_encodings = tokenize(tokenizer, src_batch)
        trg_encodings = tokenize(tokenizer, trg_batch)

        return {'input_ids': src_encodings.input_ids,
                'attention_mask': src_encodings.attention_mask,
                'labels': trg_encodings.input_ids}


    return DataLoader(
        Dataset(config, split), 
        batch_size=config.batch_size, 
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )


