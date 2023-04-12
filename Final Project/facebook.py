from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from typing import List

class m2m100:
    def __init__(self):
        self.model = M2M100ForConditionalGeneration.from_pretrained('facebook/m2m100_418M')
        self.tokenizer = M2M100Tokenizer.from_pretrained('facebook/m2m100_418M')
        self.encoded_str = ''
        self.lang = ''

    def encode(self, src_string: str, src_lang: str):
        self.lang = src_lang
        self.tokenizer.src_lang = src_lang
        self.encoded_str = self.tokenizer(src_string, return_tensors='pt')
        return self.encoded_str
    
    def decode(self, tgt_lang: str) -> str:
        tokens = self.model.generate(**self.encoded_str, forced_bos_token_id=self.tokenizer.get_lang_id(tgt_lang), max_new_tokens=256)
        output = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
        return output
    
    def get_tokens(self) -> List[int]:
        tokens = self.model.generate(**self.encoded_str, forced_bos_token_id=self.tokenizer.get_lang_id(self.lang))
        return tokens
    
    def get_model(self):
        return self.model
    
    