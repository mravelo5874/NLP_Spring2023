import transformers as trans
import utils
import time
from dataclasses import dataclass, field
from multi_lingual_models import m2m100, mbart, mt0
from similarity import mono_sim, duo_sim, multi_sim
from swadesh_words import english
from typing import List

''' word lists '''
english_words_small = ['small', 'short', 'child', 'wife', 'mother', 'construction', 'capitalism', 'capitalist', 'communism', 'father']
spanish_words_small = ['pequeño', 'corto', 'niño', 'esposa', 'madre', 'construcción', 'capitalismo', 'capitalista', 'comunismo', 'padre']
chinese_words_small = ['小', '短', '儿童', '妻子', '母亲', '建筑工程', '资本主义', '资本家', '共产主义', '父亲']
swadesh_yakhontov_english = [ 'I', 'you', 'this', 'who', 'what', 'one', 'two', 'fish', 'dog', 'louse', 'blood', 'bone', 'egg', 'horn', 'tail', 'ear', 'eye', 'nose', 
                             'tooth', 'tongue', 'hand', 'know', 'die', 'give', 'sun', 'moon', 'water', 'salt', 'stone', 'wind', 'fire', 'year', 'full', 'new', 'name']
# from https://www.nature.com/articles/s41467-018-03068-4
pereira_words_english = ['ability', 'accomplished', 'angry', 'apartment', 'applause', 'argument', 'argumentatively', 'art', 'attitude', 'bag', 'ball', 'bar', 
                         'bear', 'beat', 'bed', 'beer', 'big', 'bird', 'blood', 'body', 'brain', 'broken', 'building', 'burn', 'business', 'camera', 'carefully', 
                         'challenge', 'charity', 'charming', 'clothes', 'cockroach', 'code', 'collection', 'computer', 'construction', 'cook', 'counting', 
                         'crazy', 'damage', 'dance', 'dangerous', 'deceive', 'dedication', 'deliberately', 'delivery', 'dessert', 'device', 'dig', 'dinner', 
                         'disease', 'dissolve', 'disturb', 'do', 'doctor', 'dog', 'dressing', 'driver', 'economy', 'election', 'electron', 'elegance', 'emotion', 
                         'emotionally', 'engine', 'event', 'experiment', 'extremely', 'feeling', 'fight', 'fish', 'flow', 'food', 'garbage', 'gold', 'great', 
                         'gun', 'hair', 'help', 'hurting', 'ignorance', 'illness', 'impress', 'invention', 'investigation', 'invisible', 'job', 'jungle', 
                         'kindness', 'king', 'lady', 'land', 'laugh', 'law', 'left', 'level', 'liar', 'light', 'magic', 'marriage', 'material', 'mathematical', 
                         'mechanism', 'medication', 'money', 'mountain', 'movement', 'movie', 'music', 'nation', 'news', 'noise', 'obligation', 'pain', 
                         'personality', 'philosophy', 'picture', 'pig', 'plan', 'plant', 'play', 'pleasure', 'poor', 'prison', 'professional', 'protection', 
                         'quality', 'reaction', 'read', 'relationship', 'religious', 'residence', 'road', 'sad', 'science', 'seafood', 'sell', 'sew', 'sexy', 
                         'shape', 'ship', 'show', 'sign', 'silly', 'sin', 'skin', 'smart', 'smiling', 'solution', 'soul', 'sound', 'spoke', 'star', 'student', 
                         'stupid', 'successful', 'sugar', 'suspect', 'table', 'taste', 'team', 'texture', 'time', 'tool', 'toy', 'tree', 'trial', 'tried', 'typical', 
                         'unaware', 'usable', 'useless', 'vacation', 'war', 'wash', 'weak', 'wear', 'weather', 'willingly', 'word']


''' argument parsing stuff '''
@dataclass
class IN_ARGS():
    model: str = 'm2m'
    task: str = 'example'
    words: str = 'swadesh_110'
    l0: str = 'english'
    l1: str = 'spanish'
    sim_func: str = 'spearman'
    langs: str = 'english spanish french'

@dataclass
class OUT_ARGS():
    output_file: str = field(
        default=None,
        metadata={'help': 'output filename'})
    
def get_args():
    parser = trans.HfArgumentParser([IN_ARGS, OUT_ARGS])
    in_args, out_args, unknown_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    # show a warning on unknown arguments
    if len(unknown_args) > 0: print('[WARNING] Unknown args found: ', unknown_args)
    return in_args, out_args
    
    
def main():
    in_args, out_args = get_args()
    
    ''' example translations '''
    if in_args.task == 'example' and in_args.model == 'm2m': m2m_example_translate()
    elif in_args.task == 'example' and in_args.model == 'mbart': mbart_example_translate()
    elif in_args.task == 'example' and in_args.model == 'mt0': mt0_example_translate()
    
    
    ''' similarity matrix generation for single language '''
    if in_args.task == 'mono-sim' and in_args.model != '' and in_args.words != '' and in_args.l0 != '': 
        # get word list
        word_list = None
        use_words = None
        if in_args.words == 'small':
            word_list = english_words_small
        elif in_args.words == 'pereira':
            word_list = pereira_words_english
        elif in_args.words == 'swadesh-110':
            use_words = utils.get_swadesh_words(in_args.l0)
        elif in_args.words == 'swadesh-207' and in_args.l0 == 'english':
            use_words = english.eng_207
        # translate word list
        print ('Gathering word list for mono-sim calculation...')
        if in_args.model == 'm2m' and use_words == None:
            model = m2m100()
            use_words = utils.translate_english_words(word_list, model, in_args.l0, in_args.words)
        elif in_args.model == 'mbart' and use_words == None:
            model = mbart()
            use_words = utils.translate_english_words(word_list, model, in_args.l0, in_args.words)
        assert use_words != None  
        # print (in_args.l0, 'words:', use_words)
        mono_similarity(in_args.model, in_args.l0, use_words)

        
    ''' similarity between two languages '''
    if in_args.task == 'duo-sim' and in_args.model != '' and in_args.words != '' and in_args.l0 != '' and in_args.l1 != '':
        # get word list
        word_list = None
        words0 = None
        words1 = None
        if in_args.words == 'small':
            word_list = english_words_small
        elif in_args.words == 'pereira':
            word_list = pereira_words_english
        elif in_args.words == 'swadesh-110':
            words0 = utils.get_swadesh_words(in_args.l0)
            words1 = utils.get_swadesh_words(in_args.l1)
        # get list of translated words
        print ('Gathering word lists for duo-sim calculation...')
        if in_args.model == 'm2m' and in_args.words != 'swadesh_110':
            model = m2m100()
            words0 = utils.translate_english_words(word_list, model, in_args.l0, in_args.words)
            words1 = utils.translate_english_words(word_list, model, in_args.l1, in_args.words)
        elif in_args.model == 'mbart' and in_args.words != 'swadesh_110':
            model = mbart()
            words0 = utils.translate_english_words(word_list, model, in_args.l0, in_args.words)
            words1 = utils.translate_english_words(word_list, model, in_args.l1, in_args.words)
        assert words0 != None  
        assert words1 != None  
        # print (in_args.l0, 'words:', words0)
        # print (in_args.l1, 'words:', words1)
        duo_similarity(in_args.model, in_args.l0, in_args.l1, words0, words1, in_args.sim_func)


    ''' similarity between multiple languages '''
    if in_args.task == 'multi-sim' and in_args.model != '' and in_args.langs != []:
        # split langs
        langs_list = in_args.langs.split(',')
        assert len(langs_list) > 1
        # validate langs
        multi_similarity(in_args.model, langs_list, in_args.sim_func)
    
def multi_similarity(_model: str, _langs: List[str], _sim_func: str):
    print ('Computing similarity matrix between languages: %s with model \'%s\'. This may take some time...' % (_langs, _model))
    m_sim = multi_sim(_model, _langs)
    m_sim.compute_similarity_matrix(_sim_func)

def duo_similarity(_model: str, _lang_0: str, _lang_1: str, _words0: List[str], _words1: List[str], _sim_func: str):
    print ('Computing similarity between languages: \'%s\' and \'%s\'. This may take some time...' % (_lang_0, _lang_1))
    d_sim = duo_sim(_model, _lang_0, _lang_1, _words0, _words1)
    res = d_sim.compute_similarity(_sim_func, True)
    print ('The similarity between \'%s\' and \'%s\' using \'%s\' is: %f (between 0 and 1).' % (_lang_0, _lang_1, _model, res))

def mono_similarity(_model: str, _lang: str, _words: List[str]):
    print ('Computing mono-similarity using \'%s\' with \'%s\'. This may take some time...' % (_model, _lang))
    mono = mono_sim(_model, _words, _lang)
    mono.semantic_relation_matrix()

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

def mt0_example_translate():
    src = 'The sun rose over the mountains, casting a golden glow across the valley.'
    print ('source: ', src)
    print ('translating source sentence...\n')
    
    mt0_model = mt0()
    out_es = mt0_model.translate(src, 'english', 'spanish')
    out_fr = mt0_model.translate(src, 'english', 'french')
    out_da = mt0_model.translate(src, 'english', 'danish')
    out_ja = mt0_model.translate(src, 'english', 'japanese')
    out_zh = mt0_model.translate(src, 'english', 'chinese')

    # translate back into english
    out_es_en = mt0_model.translate(out_es, 'spanish', 'english')
    print ('spanish trans: ', out_es)
    print ('back 2 english: ', out_es_en)
    print ('')
    
    out_fr_en = mt0_model.translate(out_fr, 'french', 'english')
    print ('french trans: ', out_fr)
    print ('back 2 english: ', out_fr_en)
    print ('')
    
    out_da_en = mt0_model.translate(out_da, 'danish', 'english')
    print ('danish trans: ', out_da)
    print ('back 2 english: ', out_da_en)
    print ('')
    
    out_ja_en = mt0_model.translate(out_ja, 'japanese', 'english')
    print ('japanese trans: ', out_ja)
    print ('back 2 english: ', out_ja_en)
    print ('')

    out_zh_en = mt0_model.translate(out_zh, 'chinese', 'english')
    print ('chinese trans: ', out_zh)
    print ('back 2 english: ', out_zh_en)
    print ('')

# yoinked from: https://stackoverflow.com/questions/775049/how-do-i-convert-seconds-to-hours-minutes-and-seconds
def sec_to_hours(seconds):
    a=str(seconds//3600)
    b=str((seconds%3600)//60)
    c=str((seconds%3600)%60)
    d=["{} hours {} mins {} seconds".format(a, b, c)]
    return d

if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print("elapsed time: ", sec_to_hours(elapsed_time)[0])