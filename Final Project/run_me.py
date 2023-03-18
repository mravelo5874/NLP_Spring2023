import transformers as trans
from m2m import M2M

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

def main():
    args = trans.HfArgumentParser(trans.TrainingArguments)
    # args.add_argument('--model', type=str,
    #                   default='google/electra-small-discriminator',
    #                   help="""This argument specifies the base model to fine-tune.
    #     This should either be a HuggingFace model ID (see https://huggingface.co/models)
    #     or a path to a saved model checkpoint (a folder containing config.json and pytorch_model.bin).""")
    
    src = 'The sun rose over the mountains, casting a golden glow across the valley.'
    print ('source: ', src)
    print ('translating src...\n')
    
    m2m_model = M2M()
    m2m_model.encode(src, 'en')
    out_es = m2m_model.decode('es')
    out_da = m2m_model.decode('da')
    out_ru = m2m_model.decode('ru')
    out_tr = m2m_model.decode('tr')
    out_mn = m2m_model.decode('mn')
    out_ja = m2m_model.decode('ja')
    out_zh = m2m_model.decode('zh')
    
    # translate back into english
    m2m_model.encode(out_es, 'es')
    out_es_en = m2m_model.decode('en')
    print ('spanish trans: ', out_es)
    print ('back 2 english: ', out_es_en)
    print ('')
    
    m2m_model.encode(out_da, 'da')
    out_da_en = m2m_model.decode('en')
    print ('danish trans: ', out_da)
    print ('back 2 english: ', out_da_en)
    print ('')
    
    m2m_model.encode(out_ru, 'ru')
    out_ru_en = m2m_model.decode('en')
    print ('russian trans: ', out_ru)
    print ('back 2 english: ', out_ru_en)
    print ('')
    
    m2m_model.encode(out_tr, 'tr')
    out_tr_en = m2m_model.decode('en')
    print ('turkish trans: ', out_tr)
    print ('back 2 english: ', out_tr_en)
    print ('')
    
    m2m_model.encode(out_mn, 'mn')
    out_mn_en = m2m_model.decode('en')
    print ('mongolian trans: ', out_mn)
    print ('back 2 english: ', out_mn_en)
    print ('')
    
    m2m_model.encode(out_ja, 'ja')
    out_ja_en = m2m_model.decode('en')
    print ('japanese trans: ', out_ja)
    print ('back 2 english: ', out_ja_en)
    print ('')
    
    m2m_model.encode(out_zh, 'zh')
    out_zh_en = m2m_model.decode('en')
    print ('chinese trans: ', out_zh)
    print ('back 2 english: ', out_zh_en)
    print ('')
    
    
    
if __name__ == "__main__":
    main()