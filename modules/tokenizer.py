class Tokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, sents):
        return self.tokenizer(sents, padding=True)["input_ids"]
    
    def tokenize(self, instances):
        raise NotImplementedError()