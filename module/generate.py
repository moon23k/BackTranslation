import json, torch
from module.train import Trainer


class Generator(Trainer):
	def __init__(self, config, model, tokenizer, dataloader):
		self.model = model
		self.tokenizer = tokenizer
		self.dataloader = dataloader




	def generate(self):
        for idx, batch in enumerate(self.train_dataloader):
            input_ids, attention_mask, labels = split_batch(batch)		
            preds = model.generate(input_ids)
            preds = self.tokenizer.batch_decode(preds)
            
		return



	@staticmethod
	def save_data():
		with open(f"data/back.json", 'w') as f:
			json.dump(data_obj, f)
		return