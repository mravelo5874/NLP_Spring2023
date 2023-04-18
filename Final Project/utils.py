import numpy as np
import json
import os
from numpy.linalg import norm
from pandas import DataFrame as df
from scipy.spatial import distance
from typing import List

def inverse_interpolation(p0, p1, val):
    # clamp value to range if outside
    if (val > p1): return 1.0
    elif (val < p0): return 0.0
    # return t value
    return (val - p0) / (p1 - p0)
    
def cosine_sim(A, B):
    return np.dot(A, B) / (norm(A)*norm(B))
    
def cosine_dist(sim_0, sim_1):
    costine_dist =  distance.cosine(sim_0, sim_1) # 1 - cosine_sim(sim_0, sim_1)
    return costine_dist

def spearman(sim_0, sim_1):
    data_frame = df({'lang0': sim_0, 'lang1': sim_1 })
    spearman_correlation = data_frame['lang0'].corr(data_frame['lang1'], method='spearman')
    return spearman_correlation

# write list to memory
def write_list(a_list, file_name):
    with open(file_name, 'w') as fp:
        json.dump(a_list, fp)
        print ('created list file: ', file_name)

# read list from memory
def read_list(file_name):
    with open(file_name, 'rb') as fp:
        n_list = json.load(fp)
        print ('loaded from list file: ', file_name)
        return n_list

''' translates an english list into target language '''
def translate_english_words(_words: List[str], _model, _tgt_lang: str, _words_type: str):
    # make sure not translating english -> english
    if _tgt_lang == 'english': return _words
    
    # check if list has already been computed
    path = './word_lists/' + _tgt_lang + '_' + _words_type + '_' + _model.name() + '.json'
    if os.path.isfile(path):
        return read_list(path)
    
    trans_list = []
    for i in range(len(_words)):
        # translate word and lower-case
        trans_word: str = _model.translate(_words[i], _model.get_language_id('english'), _model.get_language_id(_tgt_lang))[0].lower()
        # if multiple words, get last word (removes 'la', 'le', 'el', 'los', 'les', etc...)
        if trans_word.find(' '):
            trans_splice = trans_word.split(' ')
            trans_word = trans_splice[-1]
        trans_list.append(trans_word)
    # write list to file
    write_list(trans_list, path)
    return trans_list

m2m_abbr = {    
    # 1 --- 10
    'afrikaans':        'af',   'afr': 'af',   'af': 'af',
    'amharic':          'am',   'amh': 'am',   'am': 'am',
    'arabic':           'ar',   'ara': 'ar',   'ar': 'ar',
    'asturian':         'ast',  'ast': 'ast',  'as': 'ast',
    'azerbaijani':      'az',   'aze': 'az',   'az': 'az',
    'bashkir':          'ba',   'bas': 'ba',   'ba': 'ba',
    'belarusian':       'be',   'bel': 'be',   'be': 'be',
    'bulgarian':        'bg',   'bul': 'bg',   'bg': 'bg',
    'bengali':          'bn',   'ben': 'bn',   'bn': 'bn',
    'breton':           'br',   'bre': 'br',   'br': 'br',
    # 11 --- 20
    'bosnian':          'bs',   'bos': 'bs',   'bs': 'bs',
    'catalan':          'ca',   'cat': 'ca',   'ca': 'ca',
    'cebuano':          'ceb',  'ceb': 'ceb',  'cb': 'ceb',
    'czech':            'cs',   'cze': 'cy',   'cy': 'cy',
    'welsh':            'cy',   'wel': 'br',   'br': 'br',
    'danish':           'da',   'dan': 'da',   'da': 'da',
    'german':           'de',   'ger': 'de',   'de': 'de',
    'greek':            'el',   'gre': 'el',   'el': 'el',
    'english':          'en',   'eng': 'en',   'en': 'en',
    'spanish':          'es',   'spa': 'es',   'es': 'es',
    # 21 --- 30
    'estonian':         'et',   'est':'et',    'et':'et',
    'persian':          'fa',   'per':'fa',    'fa':'fa',
    'fulah':            'ff',   'ful':'ff',    'ff':'ff',
    'finnish':          'fi',   'fin':'fi',    'fi':'fi',
    'french':           'fr',   'fre':'fr',    'fr':'fr',
    'frisian':          'fy',   'fri':'fy',    'fy':'fy',
    'irish':            'ga',   'iri':'ga',    'ga':'ga',
    'gaelic':           'gd',   'gae':'gd',    'gd':'gd',
    'galician':         'gl',   'gal':'gl',    'gl':'gl',
    'gujarati':         'gu',   'guj':'gu',    'gu':'gu',
    # 31 --- 40
    'hausa':            'ha',   'hau':'ha',    'ha':'ha',
    'hebrew':           'he',   'heb':'he',    'he':'he',
    'hindi':            'hi',   'hin':'hi',    'hi':'hi',
    'croatian':         'hr',   'cro':'hr',    'hr':'hr',
    'haitian':          'ht',   'hai':'ht',    'ht':'ht',
    'hungarian':        'hu',   'hun':'hu',    'hu':'hu',
    'armenian':         'hy',   'arm':'hy',    'hy':'hy',
    'indonesian':       'id',   'ind':'id',    'id':'id',
    'igbo':             'ig',   'igb':'ig',    'ig':'ig',
    'iloko':            'ilo',  'ilo':'ilo',   'il':'ilo',
    # 41 --- 50
    'icelandic':        'is',   'ice':'is',    'is':'is',
    'italian':          'it',   'ita':'it',    'it':'it',
    'japanese':         'ja',   'jap':'ja',    'ja':'ja',
    'javanese':         'jv',   'jav':'jv',    'jv':'jv',
    'georgian':         'ka',   'geo':'ka',    'ka':'ka',
    'kazakh':           'kk',   'kaz':'kk',    'kk':'kk',
    'khmer':            'km',   'khm':'km',    'km':'km',
    'kannada':          'kn',   'kan':'kn',    'kn':'kn',
    'korean':           'ko',   'kor':'ko',    'ko':'ko',
    'luxembourgish':    'lb',   'lux':'lb',    'lb':'lb',
    # 51 --- 60
    'ganda':            'lg',   'gan':'lg',    'lg':'lg',
    'lingala':          'ln',   'lin':'ln',    'ln':'ln',
    'lao':              'lo',   'lao':'lo',    'lo':'lo',
    'lithuanian':       'lt',   'lit':'lt',    'lt':'lt',
    'latvian':          'lv',   'lat':'lv',    'lv':'lv',
    'malagasy':         'mg',   'mal':'mg',    'mg':'mg',
    'macedonian':       'mk',   'mac':'mk',    'mk':'mk', 
    'malayalam':        'ml',   'mal':'ml',    'ml':'ml',
    'mongolian':        'mn',   'mon':'mn',    'mn':'mn', 
    'marathi':          'mr',   'mar':'mr',    'mr':'mr', 
    # 61 --- 70
    'malay':            'ms',   'may':'ms',    'ms':'ms',
    'burmese':          'my',   'bur':'my',    'my':'my', 
    'nepali':           'ne',   'nap':'ne',    'ne':'ne',
    'dutch':            'nl',   'dut':'nl',    'nl':'nl',
    'norwegian':        'no',   'nor':'no',    'no':'no',
    'sotho':            'ns',   'sot':'ns',    'ns':'ns',
    'occitan':          'oc',   'occ':'oc',    'oc':'oc',
    'oriya':            'or',   'ori':'or',    'or':'or',
    'punjabi':          'pa',   'pun':'pa',    'pa':'pa',
    'polish':           'pl',   'pol':'pl',    'pl':'pl',
    # 71 --- 80
    'pushto':           'ps',
    'portuguese':       'pt', 
    'romanian':         'ro',
    'russian':          'ru', 
    'sindhi':           'sd', 
    'sinhala':          'si',
    'slovak':           'sk', 
    'slovene':          'sl', 
    'somali':           'so', 
    'albanian':         'sq', 
    'serbian':          'sr', 
    'swati':            'ss', 
    'sundanese':        'su', 
    'swedish':          'sv', 
    'swahili':          'sw', 
    'tamil':            'ta', 
    'thai':             'th', 
    'tagalog':          'tl', 
    'tswana':           'tn', 
    'turkish':          'tr', 
    'ukrainian':        'uk', 
    'urdu':             'ur', 
    'uzbek':            'uz', 
    'vietnamese':       'vi', 
    'wolof':            'wo', 
    'xhosa':            'xh', 
    'yiddish':          'yi', 
    'yoruba':           'yo', 
    'chinese':          'zh', 
    'zulu':             'zu'
}

mbart_abbr = {
'arabic':       'ar_AR',
'czech':        'cs_CZ',
'german':       'de_DE', 
'english':      'en_XX', 
'spanish':      'es_XX', 
'estonian':     'et_EE', 
'finnish':      'fi_FI', 
'french':       'fr_XX', 
'gujarati':     'gu_IN',
'hindi':        'hi_IN', 
'italian':      'it_IT', 
'japanese':     'ja_XX', 
'kazakh':       'kk_KZ', 
'korean':       'ko_KR', 
'lithuanian':   'lt_LT', 
'latvian':      'lv_LV', 
'burmese':      'my_MM', 
'nepali':       'ne_NP', 
'dutch':        'nl_XX', 
'romanian':     'ro_RO', 
'russian':      'ru_RU', 
'sinhala':      'si_LK', 
'turkish':      'tr_TR', 
'vietnamese':   'vi_VN', 
'chinese':      'zh_CN', 
'afrikaans':    'af_ZA', 
'azerbaijani':  'az_AZ', 
'bengali':      'bn_IN', 
'persian':      'fa_IR', 
'hebrew':       'he_IL', 
'croatian':     'hr_HR', 
'indonesian':   'id_ID',
'georgian':     'ka_GE', 
'khmer':        'km_KH', 
'macedonian':   'mk_MK', 
'malayalam':    'ml_IN', 
'mongolian':    'mn_MN', 
'marathi':      'mr_IN', 
'polish':       'pl_PL', 
'pashto':       'ps_AF', 
'portuguese':   'pt_XX', 
'swedish':      'sv_SE', 
'swahili':      'sw_KE', 
'tamil':        'ta_IN', 
'telugu':       'te_IN', 
'thai':         'th_TH', 
'tagalog':      'tl_XX', 
'ukrainian':    'uk_UA', 
'urdu':         'ur_PK', 
'xhosa':        'xh_ZA', 
'galician':     'gl_ES', 
'slovene':      'sl_SI',
}