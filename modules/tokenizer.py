class Tokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, sents):
        return self.tokenizer(sents, padding=True)["input_ids"]
    
    def tokenize(self, instances):
        for instance in instances:
            for edu in instance.edus:
                phrase = " ".join(edu.words)
                edu.tokens = self(phrase)
        return instances