from dataclasses import dataclass, field
import transformers as trans
from facebook import m2m100
from similarity import similar

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

@dataclass
class IN_ARGS():
    task: str

@dataclass
class OUT_ARGS():
    output_file: str = field(
        default=None,
        metadata={'help': 'output filename'})
    
def get_args():
    parser = trans.HfArgumentParser([IN_ARGS, OUT_ARGS])
    in_args, out_args, unknown_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    # show a warning on unknown arguments
    if len(unknown_args) > 0: print('[warning] unknown args found: ', unknown_args)
    return in_args, out_args
    
def main():
    in_args, out_args = get_args()
    
    if in_args.task == 'example': example_translate()
    elif in_args.task == 'sim': similarity() 
    
def similarity():
    print ('english:')
    sim_en = similar(['dog', 'cat', 'house', 'mouse'], 'en')
    sim_en.generate_semantic_relation_matrix()
    return
    
    print ('english:')
    sim_en = similar(['small', 'short', 'child', 'wife', 'mother'], 'en')
    sim_en.generate_semantic_relation_matrix()
    
    print ('spanish:')
    sim_es = similar(['pequeño', 'corto', 'niño', 'esposa', 'madre'], 'es')
    sim_es.generate_semantic_relation_matrix()
    
    print ('russian:')
    sim_ru = similar(['небольшой', 'короткий', 'ребенок', 'жена', 'мать'], 'ru')
    sim_ru.generate_semantic_relation_matrix()
    
    print ('chinese:')
    sim_zh = similar(['小', '短', '儿童', '妻子', '母亲'], 'zh')
    sim_zh.generate_semantic_relation_matrix()


def example_translate():
    ''' example sentences: '''
    # 'The sun rose over the mountains, casting a golden glow across the valley.'
    # 'The city\'s heartbeat echoed through the night, a symphony of sirens, footsteps, and distant laughter, as the streets pulsed with life and energy, relentless and unforgiving.'
    # 'The scent of roses wafted through the air, mingling with the salty tang of the sea, as the sun dipped below the horizon, casting the world into twilight.'
    # 'The theoretical framework adopted in this research drew upon established theories in the field of social sciences, providing a solid conceptual foundation for the study and guiding the formulation of research questions and hypotheses.'
    # 'The scent of roses wafted through the air, mingling with the salty tang of the sea, as the sun dipped below the horizon, casting the world into twilight.'
    
    src = 'cat dog mouse house'
    print ('source: ', src)
    print ('translating source sentence...\n')
    
    m2m_model = m2m100()
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