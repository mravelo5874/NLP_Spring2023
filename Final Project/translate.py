from dataclasses import dataclass, field
import transformers as trans
from facebook import m2m100, mbart
from google_research import t5
from similarity import similar

''' example sentences: '''
# 'The sun rose over the mountains, casting a golden glow across the valley.'
# 'The city\'s heartbeat echoed through the night, a symphony of sirens, footsteps, and distant laughter, as the streets pulsed with life and energy, relentless and unforgiving.'
# 'The scent of roses wafted through the air, mingling with the salty tang of the sea, as the sun dipped below the horizon, casting the world into twilight.'
# 'The theoretical framework adopted in this research drew upon established theories in the field of social sciences, providing a solid conceptual foundation for the study and guiding the formulation of research questions and hypotheses.'
# 'The scent of roses wafted through the air, mingling with the salty tang of the sea, as the sun dipped below the horizon, casting the world into twilight.'

@dataclass
class IN_ARGS():
    task: str
    model: str

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
    
    # example translations
    if in_args.task == 'example' and in_args.model == 'm2m': m2m_example_translate()
    elif in_args.task == 'example' and in_args.model == 'mbart': mbart_example_translate()
    elif in_args.task == 'example' and in_args.model == 'mt5': mt5_example_translate()
    
    # similarity matrix generation
    elif in_args.task == 'sim' and in_args.model == 'm2m': similarity('m2m')
    
    
    else: print ('[ERROR] Input arguments did not match any task or model.')
    
def similarity(model: str):

    print ('m2m:')
    sim_en_m2m = similar('m2m', ['red', 'green', 'blue', 'color', 'paint'], 'en')
    sim_en_m2m.generate_semantic_relation_matrix()
    
    print ('mbart:')
    sim_en_mbart = similar('mbart', ['red', 'green', 'blue', 'color', 'paint'], 'en_XX')
    sim_en_mbart.generate_semantic_relation_matrix()
    
    # print ('spanish:')
    # sim_es_m2m = similar('m2m', ['pequeño', 'corto', 'niño', 'esposa', 'madre'], 'es')
    # sim_es_m2m.generate_semantic_relation_matrix()
    
    # print ('spanish:')
    # sim_es_mbart = similar('mbart', ['pequeño', 'corto', 'niño', 'esposa', 'madre'], 'es_XX')
    # sim_es_mbart.generate_semantic_relation_matrix()
    
    # print ('russian:')
    # sim_ru = similar(model, ['небольшой', 'короткий', 'ребенок', 'жена', 'мать'], 'ru')
    # sim_ru.generate_semantic_relation_matrix()
    
    # print ('chinese:')
    # sim_zh = similar(model, ['小', '短', '儿童', '妻子', '母亲'], 'zh')
    # sim_zh.generate_semantic_relation_matrix()


def m2m_example_translate():
    
    src = 'The sun rose over the mountains, casting a golden glow across the valley.'
    print ('source: ', src)
    print ('translating source sentence...\n')
    
    m2m_model = m2m100()
    out_es = m2m_model.translate(src, 'en', 'es')
    out_da = m2m_model.translate(src, 'en', 'da')
    out_ru = m2m_model.translate(src, 'en', 'ru')
    out_tr = m2m_model.translate(src, 'en', 'tr')
    out_mn = m2m_model.translate(src, 'en', 'mn')
    out_ja = m2m_model.translate(src, 'en', 'ja')
    out_zh = m2m_model.translate(src, 'en', 'zh')

    # translate back into english
    out_es_en = m2m_model.translate(out_es, 'es', 'en')
    print ('spanish trans: ', out_es)
    print ('back 2 english: ', out_es_en)
    print ('')
    
    out_da_en = m2m_model.translate(out_da, 'da', 'en')
    print ('danish trans: ', out_da)
    print ('back 2 english: ', out_da_en)
    print ('')
    
    out_ru_en = m2m_model.translate(out_ru, 'ru', 'en')
    print ('russian trans: ', out_ru)
    print ('back 2 english: ', out_ru_en)
    print ('')
    
    out_tr_en = m2m_model.translate(out_tr, 'tr', 'en')
    print ('turkish trans: ', out_tr)
    print ('back 2 english: ', out_tr_en)
    print ('')
    
    out_mn_en = m2m_model.translate(out_mn, 'mn', 'en')
    print ('mongolian trans: ', out_mn)
    print ('back 2 english: ', out_mn_en)
    print ('')
    
    out_ja_en = m2m_model.translate(out_ja, 'ja', 'en')
    print ('japanese trans: ', out_ja)
    print ('back 2 english: ', out_ja_en)
    print ('')

    out_zh_en = m2m_model.translate(out_zh, 'zh', 'en')
    print ('chinese trans: ', out_zh)
    print ('back 2 english: ', out_zh_en)
    print ('')

def mbart_example_translate():
    src = 'The sun rose over the mountains, casting a golden glow across the valley.'
    print ('source: ', src)
    print ('translating source sentence...\n')
    
    bart_model = mbart()
    out_es = bart_model.translate(src, 'en_XX', 'es_XX')
    out_nl = bart_model.translate(src, 'en_XX', 'nl_XX')
    out_ru = bart_model.translate(src, 'en_XX', 'ru_RU')
    out_tr = bart_model.translate(src, 'en_XX', 'tr_TR')
    out_mn = bart_model.translate(src, 'en_XX', 'mn_MN')
    out_ja = bart_model.translate(src, 'en_XX', 'ja_XX')
    out_zh = bart_model.translate(src, 'en_XX', 'zh_CN')

    # translate back into english
    out_es_en = bart_model.translate(out_es, 'es_XX', 'en_XX')
    print ('spanish trans: ', out_es)
    print ('back 2 english: ', out_es_en)
    print ('')
    
    out_da_en = bart_model.translate(out_nl, 'nl_XX', 'en_XX')
    print ('dutch trans: ', out_nl)
    print ('back 2 english: ', out_da_en)
    print ('')
    
    out_ru_en = bart_model.translate(out_ru, 'ru_RU', 'en_XX')
    print ('russian trans: ', out_ru)
    print ('back 2 english: ', out_ru_en)
    print ('')
    
    out_tr_en = bart_model.translate(out_tr, 'tr_TR', 'en_XX')
    print ('turkish trans: ', out_tr)
    print ('back 2 english: ', out_tr_en)
    print ('')
    
    out_mn_en = bart_model.translate(out_mn, 'mn_MN', 'en_XX')
    print ('mongolian trans: ', out_mn)
    print ('back 2 english: ', out_mn_en)
    print ('')
    
    out_ja_en = bart_model.translate(out_ja, 'ja_XX', 'en_XX')
    print ('japanese trans: ', out_ja)
    print ('back 2 english: ', out_ja_en)
    print ('')

    out_zh_en = bart_model.translate(out_zh, 'zh_CN', 'en_XX')
    print ('chinese trans: ', out_zh)
    print ('back 2 english: ', out_zh_en)
    print ('')

def mt5_example_translate():
    src = 'The sun rose over the mountains, casting a golden glow across the valley.'
    print ('source: ', src)
    print ('translating source sentence...\n')
    
    t5_model = t5()
    res = t5_model.translate(src, 'English', 'French')
    print ('french trans: ', res)
    
    res = t5_model.translate(src, 'English', 'German')
    print ('german trans: ', res)
  
if __name__ == "__main__":
    main()