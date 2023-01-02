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
        src_ids = self.data[idx]['src_ids']
        src_mask = self.data[idx]['src_mask']

        trg_ids = self.data[idx]['trg_ids']
        trg_mask = self.data[idx]['trg_mask']
        
        return src_ids, src_mask, trg_ids, trg_mask


def pad_batch(batch_list, pad_id):
    return pad_sequence(batch_list,
                        batch_first=True,
                        padding_value=pad_id)


def load_dataloader(config, split):
    global pad_id
    task = config.task
    pad_id = config.pad_id    

    def collate_fn(batch):
        src_ids_batch, src_mask_batch = [], []
        trg_ids_batch, trg_mask_batch = [], []

        for src_ids, src_mask, trg_ids, trg_mask in batch:

            src_ids_batch.append(torch.LongTensor(src_ids))
            src_mask_batch.append(torch.LongTensor(src_mask))

            trg_ids_batch.append(torch.LongTensor(trg_ids))
            trg_mask_batch.append(torch.LongTensor(trg_mask))

        
        src_ids_batch = pad_batch(src_ids_batch, pad_id)
        src_mask_batch = pad_batch(src_mask_batch, pad_id)
        
        trg_ids_batch = pad_batch(trg_ids_batch, pad_id)
        trg_mask_batch = pad_batch(trg_mask_batch, pad_id)


        return {'src_ids': src_ids_batch, 
                'src_mask': src_mask_batch,
                'trg_ids': trg_ids_batch, 
                'trg_mask': trg_mask_batch}



    return DataLoader(Dataset(split), 
                      batch_size=config.batch_size, 
                      shuffle=True,
                      collate_fn=collate_fn,
                      num_workers=2,
                      pin_memory=True)