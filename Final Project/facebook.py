from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from typing import List

"""
M2M LANGUAGES:
Afrikaans (af), Amharic (am), Arabic (ar), Asturian (ast), Azerbaijani (az), Bashkir (ba), Belarusian (be), 
Bulgarian (bg), Bengali (bn), Breton (br), Bosnian (bs), Catalan; Valencian (ca), Cebuano (ceb), Czech (cs), 
Welsh (cy), Danish (da), German (de), Greeek (el), English (en), Spanish (es), Estonian (et), Persian (fa), 
Fulah (ff), Finnish (fi), French (fr), Western Frisian (fy), Irish (ga), Gaelic; Scottish Gaelic (gd), Galician (gl), 
Gujarati (gu), Hausa (ha), Hebrew (he), Hindi (hi), Croatian (hr), Haitian; Haitian Creole (ht), Hungarian (hu), 
Armenian (hy), Indonesian (id), Igbo (ig), Iloko (ilo), Icelandic (is), Italian (it), Japanese (ja), Javanese (jv), 
Georgian (ka), Kazakh (kk), Central Khmer (km), Kannada (kn), Korean (ko), Luxembourgish; Letzeburgesch (lb), 
Ganda (lg), Lingala (ln), Lao (lo), Lithuanian (lt), Latvian (lv), Malagasy (mg), Macedonian (mk), Malayalam (ml), 
Mongolian (mn), Marathi (mr), Malay (ms), Burmese (my), Nepali (ne), Dutch; Flemish (nl), Norwegian (no), 
Northern Sotho (ns), Occitan (post 1500) (oc), Oriya (or), Panjabi; Punjabi (pa), Polish (pl), Pushto; Pashto (ps), 
Portuguese (pt), Romanian; Moldavian; Moldovan (ro), Russian (ru), Sindhi (sd), Sinhala; Sinhalese (si), Slovak (sk), 
Slovenian (sl), Somali (so), Albanian (sq), Serbian (sr), Swati (ss), Sundanese (su), Swedish (sv), Swahili (sw), 
Tamil (ta), Thai (th), Tagalog (tl), Tswana (tn), Turkish (tr), Ukrainian (uk), Urdu (ur), Uzbek (uz), Vietnamese (vi), 
Wolof (wo), Xhosa (xh), Yiddish (yi), Yoruba (yo), Chinese (zh), Zulu (zu)
"""

class m2m100:
    def __init__(self):
        self.model = M2M100ForConditionalGeneration.from_pretrained('facebook/m2m100_1.2B')
        self.tokenizer = M2M100Tokenizer.from_pretrained('facebook/m2m100_1.2B')
        self.input_embed = self.model.get_input_embeddings()
    
    def translate(self, src: str, src_lang: str, tgt_lang: str):
        self.tokenizer.src_lang = src_lang
        tokenized = self.tokenizer(src, return_tensors='pt')
        tokens = self.model.generate(**tokenized, forced_bos_token_id=self.tokenizer.get_lang_id(tgt_lang), max_new_tokens=200)
        output = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
        return output
    
    def embed(self, src: str, lang: str):
        self.tokenizer.src_lang = lang
        encoded = self.tokenizer(src, return_tensors='pt')
        embeded = self.input_embed.forward(encoded.input_ids)
        return embeded

"""
Arabic (ar_AR), Czech (cs_CZ), German (de_DE), English (en_XX), Spanish (es_XX), Estonian (et_EE), Finnish (fi_FI), French (fr_XX), 
Gujarati (gu_IN), Hindi (hi_IN), Italian (it_IT), Japanese (ja_XX), Kazakh (kk_KZ), Korean (ko_KR), Lithuanian (lt_LT), Latvian (lv_LV), 
Burmese (my_MM), Nepali (ne_NP), Dutch (nl_XX), Romanian (ro_RO), Russian (ru_RU), Sinhala (si_LK), Turkish (tr_TR), Vietnamese (vi_VN), 
Chinese (zh_CN), Afrikaans (af_ZA), Azerbaijani (az_AZ), Bengali (bn_IN), Persian (fa_IR), Hebrew (he_IL), Croatian (hr_HR), 
Indonesian (id_ID), Georgian (ka_GE), Khmer (km_KH), Macedonian (mk_MK), Malayalam (ml_IN), Mongolian (mn_MN), Marathi (mr_IN), 
Polish (pl_PL), Pashto (ps_AF), Portuguese (pt_XX), Swedish (sv_SE), Swahili (sw_KE), Tamil (ta_IN), Telugu (te_IN), Thai (th_TH), 
Tagalog (tl_XX), Ukrainian (uk_UA), Urdu (ur_PK), Xhosa (xh_ZA), Galician (gl_ES), Slovene (sl_SI)
"""

class mbart:
    def __init__(self):
        self.model = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')
        self.tokenizer = MBart50TokenizerFast.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')
        self.input_embed = self.model.get_input_embeddings()
        
    def translate(self, src: str, _src_lang: str, _tgt_lang: str):
        self.tokenizer.src_lang = _src_lang
        encoded_hi = self.tokenizer(src, return_tensors='pt')
        generated_tokens = self.model.generate(**encoded_hi,forced_bos_token_id=self.tokenizer.lang_code_to_id[_tgt_lang], max_new_tokens=200)
        output = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return output
    
    def embed(self, src: str, lang: str):
        self.tokenizer.src_lang = lang
        encoded = self.tokenizer(src, return_tensors='pt')
        embeded = self.input_embed.forward(encoded.input_ids)
        return embeded
