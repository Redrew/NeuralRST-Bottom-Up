class Tokenizer:
    def __init__(self, tokenizer, remove_prefix=False, remove_postfix=False):
        self.tokenizer = tokenizer
        self.remove_prefix = remove_prefix
        self.remove_postfix = remove_postfix
    
    def __call__(self, sents):
        return self.tokenizer(sents, padding=True)["input_ids"]
    
    def tokenize(self, instances):
        for instance in instances:
            for edu in instance.edus:
                phrase = " ".join(edu.words)
                tokens = self(phrase)
                if self.remove_prefix:
                    tokens = tokens[1:]
                if self.remove_postfix:
                    tokens = tokens[:-1]
                edu.tokens = tokens
        return instances