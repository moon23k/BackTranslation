import os, json, torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence



class Dataset(torch.utils.data.Dataset):

    def __init__(self, split):
        super().__init__()
        self.data = self.load_data(split)

    @staticmethod
    def load_data(split):
        with open(f"data/{split}.json", 'r') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]['src'].lower()
        y = self.data[idx]['trg'].lower()
        return x, y



class Collator(object):

    def __init__(self, tokenizer, for_sampling, pad_id=None):

        self.tokenizer = tokenizer
        self.pad_id = pad_id
        self.for_sampling = for_sampling


    def tokenize(self, batch):
        if self.for_sampling:
            batch_ids = self.tokenizer(
                batch, padding=True, 
                truncation=True, return_tensors='pt'
            ).input_ids

        else:
            batch_ids = []
            for x in batch:
                ids = self.tokenizer.encode(x).ids
                batch_ids.append(torch.LongTensor(ids))
            
            batch_ids = pad_sequence(
                batch_ids, batch_first=True, 
                padding_value=self.pad_id
            )
        
        return batch_ids


    def __call__(self, batch):
        x_batch, y_batch = zip(*batch)
        return {'x': self.tokenize(x_batch), 
                'y': self.tokenize(y_batch)}


def load_dataloader(config, tokenizer, split):

    tok_type = str(type(tokenizer))
    for_sampling = True if 'marian' in tok_type else False
    pad_id = config.pad_id if not for_sampling else None
    shuffle = False if for_sampling else True


    return DataLoader(
        Dataset(split), 
        batch_size=config.batch_size, 
        shuffle=shuffle,
        collate_fn=Collator(tokenizer, for_sampling, pad_id),
        num_workers=2,
        pin_memory=True
    )
