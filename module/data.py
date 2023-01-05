import json, torch
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
        ko_ids = self.data[idx]['ko_ids']
        ko_mask = self.data[idx]['ko_mask']

        en_ids = self.data[idx]['en_ids']
        en_mask = self.data[idx]['en_mask']
        
        return ko_ids, ko_mask, en_ids, en_mask


def pad_batch(batch_list, pad_id):
    return pad_sequence(batch_list,
                        batch_first=True,
                        padding_value=pad_id)


def load_dataloader(config, split):
    global pad_id
    pad_id = config.pad_id    

    def collate_fn(batch):
        ko_ids_batch, ko_mask_batch = [], []
        en_ids_batch, en_mask_batch = [], []

        for ko_ids, ko_mask, en_ids, en_mask in batch:

            ko_ids_batch.append(torch.LongTensor(ko_ids))
            ko_mask_batch.append(torch.LongTensor(ko_mask))

            en_ids_batch.append(torch.LongTensor(en_ids))
            en_mask_batch.append(torch.LongTensor(en_mask))

        
        ko_ids_batch = pad_batch(ko_ids_batch, pad_id)
        ko_mask_batch = pad_batch(ko_mask_batch, pad_id)
        
        en_ids_batch = pad_batch(en_ids_batch, pad_id)
        en_mask_batch = pad_batch(en_mask_batch, pad_id)


        return {'ko_ids': ko_ids_batch, 
                'ko_mask': ko_mask_batch,
                'en_ids': en_ids_batch, 
                'en_mask': en_mask_batch}



    return DataLoader(Dataset(split), 
                      batch_size=config.batch_size, 
                      shuffle=True,
                      collate_fn=collate_fn,
                      num_workers=2,
                      pin_memory=True)