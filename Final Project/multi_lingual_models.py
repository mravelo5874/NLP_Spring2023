import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from utils import m2m_abbr, mbart_abbr

class _multi_lang_model_:
    def __init__(self):
        pass

class m2m100(_multi_lang_model_):
    def __init__(self, _gpu: bool=True):
        self.model = M2M100ForConditionalGeneration.from_pretrained('facebook/m2m100_1.2B')
        self.tokenizer = M2M100Tokenizer.from_pretrained('facebook/m2m100_1.2B')
        self.input_embed = self.model.get_input_embeddings()
        self.device = 'cpu'
        if _gpu: self.set_gpu()
        
    def name(self): return 'm2m'
        
    def set_gpu(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        #print ('device set to: ', self.device)
    
    def translate(self, src: str, _src_lang: str, _tgt_lang: str):
        # return src if no translation is needed
        if _src_lang == _tgt_lang: return src
        # translate src
        self.tokenizer.src_lang = _src_lang
        tokenized = self.tokenizer(src, return_tensors='pt').to(self.device)
        tokens = self.model.generate(**tokenized, forced_bos_token_id=self.tokenizer.get_lang_id(_tgt_lang), max_new_tokens=200)
        output = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
        return output
    
    def embed(self, src: str, lang: str):
        self.tokenizer.src_lang = lang
        encoded = self.tokenizer(src, return_tensors='pt')
        embeded = self.input_embed.forward(encoded.input_ids)
        return embeded
    
    def get_language_id(self, lang: str):
        return m2m_abbr.get(lang.lower())

class mbart(_multi_lang_model_):
    def __init__(self, _gpu: bool=True):
        self.model = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')
        self.tokenizer = MBart50TokenizerFast.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')
        self.input_embed = self.model.get_input_embeddings()
        self.device = 'cpu'
        if _gpu: self.set_gpu()
        
    def name(self): return 'mbart'
        
    def set_gpu(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        #print ('device set to: ', self.device)
        
    def translate(self, src: str, _src_lang: str, _tgt_lang: str):
        # return src if no translation is needed
        if _src_lang == _tgt_lang: return src
        # translate src
        self.tokenizer.src_lang = _src_lang
        encoded_hi = self.tokenizer(src, return_tensors='pt').to(self.device)
        generated_tokens = self.model.generate(**encoded_hi,forced_bos_token_id=self.tokenizer.lang_code_to_id[_tgt_lang], max_new_tokens=200)
        output = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return output
    
    def embed(self, src: str, lang: str):
        self.tokenizer.src_lang = lang
        encoded = self.tokenizer(src, return_tensors='pt')
        #print ('encoded: ', encoded.input_ids)
        embeded = self.input_embed.forward(encoded.input_ids)
        return embeded
    
    def get_language_id(self, lang: str):
        return mbart_abbr.get(lang.lower())

''' not in use atm '''
class mt0(_multi_lang_model_):
    def __init__(self):
        self.model = AutoModelForSeq2SeqLM.from_pretrained('bigscience/mt0-large')
        self.tokenizer = AutoTokenizer.from_pretrained('bigscience/mt0-large')
        #self.input_embed = self.model.get_input_embeddings()
        
    def translate(self, src: str, _src_lang: str, _tgt_lang: str):
        inputs = self.tokenizer.encode('translate from ' + _src_lang + ' to ' + _tgt_lang + ': ' + src, return_tensors='pt')
        generated = self.model.generate(inputs, max_new_tokens=200)
        output = self.tokenizer.decode(generated[0])
        return output
    
    def embed(self, src: str, lang: str):
        raise NotImplementedError

    def get_language_id(lang: str):
        raise NotImplementedError