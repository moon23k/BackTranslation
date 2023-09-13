import json, torch, evaluate



class Tester:
    def __init__(self, config, model, tokenizer, test_dataloader):
        super(Tester, self).__init__()
        
        self.model = model
        self.tokenizer = tokenizer
        self.dataloader = test_dataloader
        
        self.path = config.path
        self.pad_id = config.pad_id
        self.bos_id = config.bos_id
        self.device = config.device
        self.max_len = config.max_len
        
        self.metric_module = evaluate.load('bleu')



    def test(self):
        score = 0.0         
        self.model.eval()

        with torch.no_grad():
            for batch in self.dataloader:
                x = batch['src'].to(self.device)
                y = self.tokenize(batch['trg'])

                pred = self.predict(x)
                pred = self.tokenize(pred)
                
                score += self.evaluate(pred, y)

        txt = f"TEST Result"
        txt += f"\n-- Score: {round(score/len(self.dataloader), 2)}\n"
        print(txt)


    def tokenize(self, batch):
        return [self.tokenizer.decode(x) for x in batch.tolist()]


    def predict(self, x):

        batch_size = x.size(0)
        pred = torch.zeros((batch_size, self.max_len)).fill_(self.pad_id)
        pred = pred.type(torch.LongTensor).to(self.device)
        pred[:, 0] = self.bos_id

        e_mask = self.model.pad_mask(x)
        memory = self.model.encode(x, e_mask)

        for idx in range(1, self.max_len):
            y = pred[:, :idx]
            d_out = self.model.decode(y, memory, e_mask, None, None)

            logit = self.model.generator(d_out)
            pred[:, idx] = logit.argmax(dim=-1)[:, -1]

        return pred
                


    def evaluate(self, pred, label):

        if pred == ['' for _ in range(len(pred))]:
            return 0.0

        score = self.metric_module.compute(
            predictions=pred, 
            references =[[l] for l in label]
        )['bleu']

        return score * 100
