class Tokenizer:
    def __init__(self, tokenizer, remove_prefix=False, remove_postfix=False, add_prefix_space=False):
        self.tokenizer = tokenizer
        self.remove_prefix = remove_prefix
        self.remove_postfix = remove_postfix
        self.add_prefix_space = add_prefix_space
    
    def __call__(self, sents):
        return self.tokenizer(sents, padding=True)["input_ids"]
    
    def tokenize(self, instances):
        for instance in instances:
            for edu_i, edu in enumerate(instance.edus):
                phrase = " ".join(edu.words)
                if self.add_prefix_space and edu_i != 0:
                    phrase = " " + phrase
                tokens = self(phrase)
                if self.remove_prefix:
                    tokens = tokens[1:]
                if self.remove_postfix:
                    tokens = tokens[:-1]
                edu.tokens = tokens
        return instances