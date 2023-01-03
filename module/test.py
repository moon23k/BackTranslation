import torch, time
from datasets import load_metric



class Tester:
    def __init__(self, config, model, tokenizer, test_dataloader):
        super(Tester, self).__init__()
        
        self.model = model
        self.task = config.task
        self.tokenizer = tokenizer
        self.num_beams = config.num_beams
        self.dataloader = test_dataloader
        self.device_type = config.device_type


    @staticmethod
    def measure_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_min = int(elapsed_time / 60)
        elapsed_sec = int(elapsed_time - (elapsed_min * 60))
        return f"{elapsed_min}m {elapsed_sec}s"


    def test(self):
        self.model.eval()
        metric_module = load_metric('bleu')
        
        start_time = time.time()
        with torch.no_grad():
            for _, batch in enumerate(self.dataloader):   
                
                if self.task == 'translation':
                    input_ids = batch['src_ids']
                    labels = batch['trg_ids']
                
                elif self.task == 'back_translation':
                    input_ids = batch['src_ids']
                    labels = batch['trg_ids']
                
                with torch.autocast(device_type=self.device_type, dtype=torch.float16):
                    preds = self.model.genreate(input_ids, num_beams=self.num_beams, max_new_tokens=300)
                
                preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
                labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

                metric_module.add_batch(predictions=[p.split() for p in preds], 
                                        references=[[l.split()] for l in labels])            

        bleu_score = metric_module.compute()['bleu'] * 100

        print('Test Results')
        print(f"  >> BLEU Score: {bleu_score}")
        print(f"  >> Spent Time: {self.measure_time(start_time, time.time())}")
    

