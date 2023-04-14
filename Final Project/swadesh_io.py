from swadesh_words import afrikaans, amharic, arabic

from swadesh_words import english, finnish, french, german, japanese, portuguese, spanish, ukrainian

''' language with swadesh word list '''
valid_languages = [
'afrikaans',    'afr',      'af',
'amharic',      'amh',      'am',
'arabic',       'ara',      'ar',

]

def get_swadesh_words(lang: str, version: str):
    assert lang in valid_languages
    if version == 'swadesh-110':
        if      lang=='english'         or lang=='eng' or lang=='en': return english.eng_110
        elif    lang=='finnish'         or lang=='eng' or lang=='en': return finnish.fin_110
        elif    lang=='french'          or lang=='eng' or lang=='en': return french.fra_110
        elif    lang=='german'          or lang=='eng' or lang=='en': return german.deu_110
        elif    lang=='japanese'        or lang=='eng' or lang=='en': return japanese.jpn_110
        elif    lang=='portuguese'      or lang=='eng' or lang=='en': return portuguese.por_110
        elif    lang=='spanish'         or lang=='eng' or lang=='en': return spanish.spa_110
        elif    lang=='ukrainian'       or lang=='eng' or lang=='en': return ukrainian.ukr_100
        else:
            print ('[ERROR]: Could not find swadesh-110 list for \'%s\'.' % lang)
    elif version == 'swadesh-207':
        if   lang=='afrikaans'          or lang=='afr' or lang=='af': return afrikaans.afr_207
        elif lang=='amharic'            or lang=='amh' or lang=='am': return amharic.amh_207
        elif lang=='arabic'             or lang=='ara' or lang=='ar': return arabic.ara_207
        
        
        
       
        else:
            print ('[ERROR]: Could not find swadesh-207 list for \'%s\'.' % lang)