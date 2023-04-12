from transformers import T5Tokenizer, T5ForConditionalGeneration
from utils import id_2_language

class t5:
    def __init__(self):
        self.model = T5ForConditionalGeneration.from_pretrained('t5-small')
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small', use_fast=True)
        
    def translate(self, src: str, src_lang: str, tgt_lang: str):
        prefix = 'translate to ' + id_2_language(tgt_lang) + ': '
        #tokenized = self.tokenizer(prefix + src, return_tensors='pt')
        tokenized = self.tokenizer.encode(prefix + src, return_tensors='pt')
        outputs = self.model.generate(tokenized, max_new_tokens=200)
        res = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return res
    
    def get_model(self):
        return self.model