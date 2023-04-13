import numpy as np
import json
import os
from numpy.linalg import norm
from pandas import DataFrame as df
from scipy.spatial import distance
from typing import List
from swadesh_words import english, finnish, french, german, japanese, portuguese, spanish, ukrainian

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
    
def get_swadesh_words(lang: str):
    print ('getting swadesh words for: ', lang)
    if      lang == 'english':      return english.eng_110
    elif    lang == 'finnish':      return finnish.fin_110
    elif    lang == 'french':       return french.fra_110
    elif    lang == 'german':       return german.deu_110
    elif    lang == 'japanese':     return japanese.jpn_110
    elif    lang == 'portuguese':   return portuguese.por_110
    elif    lang == 'spanish':      return spanish.spa_110
    elif    lang == 'ukrainian':    return ukrainian.ukr_100
    else:
        print ('Could not find swadesh words for \'%s\'. Translating the english word list instead.' % lang)
        return translate_english_words(english.eng_110, 'm2m', lang, 'swadesh-temp')

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
    'afrikaans':        'af', 
    'amharic':          'am', 
    'arabic':           'ar', 
    'asturian':         'as', 
    'azerbaijani':      'az', 
    'bashkir':          'ba', 
    'belarusian':       'be', 
    'bulgarian':        'bg', 
    'bengali':          'bn', 
    'breton':           'br', 
    'bosnian':          'bs', 
    'catalan':          'ca',   # Valencian
    'cebuano':          'ce', 
    'czech':            'cs', 
    'welsh':            'cy', 
    'danish':           'da', 
    'german':           'de',
    'greek':            'el', 
    'english':          'en', 
    'spanish':          'es', 
    'estonian':         'et', 
    'persian':          'fa', 
    'fulah':            'ff', 
    'finnish':          'fi', 
    'french':           'fr', 
    'west_frisian':     'fy', 
    'irish':            'ga', 
    'gaelic':           'gd',  # Scottish Gaelic
    'galician':         'gl', 
    'gujarati':         'gu', 
    'hausa':            'ha', 
    'hebrew':           'he', 
    'hindi':            'hi', 
    'croatian':         'hr', 
    'haitian':          'ht', # Haitian Creole
    'hungarian':        'hu', 
    'armenian':         'hy',
    'indonesian':       'id', 
    'igbo':             'ig', 
    'iloko':            'il', 
    'icelandic':        'is', 
    'italian':          'it', 
    'japanese':         'ja', 
    'javanese':         'jv', 
    'georgian':         'ka', 
    'kazakh':           'kk', 
    'central_khmer':    'km', 
    'kannada':          'kn', 
    'korean':           'ko', 
    'luxembourgish':    'lb', # Letzeburgesch 
    'ganda':            'lg', 
    'lingala':          'ln', 
    'lao':              'lo', 
    'lithuanian':       'lt', 
    'latvian':          'lv', 
    'malagasy':         'mg', 
    'macedonian':       'mk', 
    'malayalam':        'ml', 
    'mongolian':        'mn', 
    'marathi':          'mr', 
    'malay ':           'ms', 
    'burmese':          'my', 
    'nepali':           'ne', 
    'dutch':            'nl', # Flemish
    'norwegian':        'no', 
    'northern_sotho':   'ns', 
    'occitan':          'oc', # (post 1500)
    'oriya':            'or', 
    'panjabi':          'pa', # Punjabi
    'polish':           'pl', 
    'pushto':           'ps', # Pashto
    'portuguese':       'pt', 
    'romanian':         'ro', # Moldavian; Moldovan
    'russian':          'ru', 
    'sindhi':           'sd', 
    'sinhala':          'si', # Sinhalese
    'slovak':           'sk', 
    'slovenian':        'sl', 
    'somali':           'so', 
    'albanian':         'sq', 
    'serbian':          'sr', 
    'swati':            'ss', 
    'sundanese':        'su', 
    'swedish ':         'sv', 
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
    'yoruba':            'yo', 
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



"""
[VALID MTO LANGUAGES]
Afrikaans
Amharic
Arabic
Azerbaijani
Belarusian
Bulgarian
Bengali
Catalan
Cebuano
Corsican
Czech
Welsh
Danish
German
Greek
English
Esperanto
Spanish
Estonian
Basque
Persian
Finnish
fil
French
Western Frisian
Irish
Scottish Gaelic
Galician
Gujarati
Hausa
haw
Hindi
hmn
Haitian
Hungarian
Armenian
Igbo
Icelandic
Italian
iw
Japanese
Javanese
Georgian
Kazakh
Khmer
Kannada
Korean
Kurdish
Kyrgyz
Latin
Luxembourgish
Lao
Lithuanian
Latvian
Malagasy
Māori
Macedonian
Malayalam
Mongolian
Marathi
Malay
Maltese
Burmese
Nepali
Dutch
Norwegian
Chichewa
Panjabi
Polish
Pashto
Portuguese
Romanian
Russian
Sindhi
Sinhala
Slovak
Slovenian
Samoan
Shona
Somali
Albanian
Serbian
Southern Sotho
Sundanese
Swedish
Swahili
Tamil
Telugu
Tajik
Thai
Turkish
Ukrainian
und
Urdu
Uzbek
Vietnamese
Xhosa
Yiddish
Yoruba
Chinese
Zulu
"""