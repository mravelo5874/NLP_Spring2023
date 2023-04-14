# 1 --- 10
from swadesh_words import afrikaans, amharic, arabic, asturian, azerbaijani, bashkir, belarusian, bulgarian, bengali, breton
# 11 --- 20
from swadesh_words import bosnian, catalan, cebuano, czech, welsh, danish, german, greek, english, spanish
# 21 --- 30

''' language with swadesh word list '''
valid_languages = [
# 1 --- 10
'afrikaans',    'afr',      'af',
'amharic',      'amh',      'am',
'arabic',       'ara',      'ar',
'asturian',     'ast',      'as',
'azerbaijani',  'aze',      'az',
'bashkir',      'bas',      'ba',
'belarusian',   'bel',      'be',
'bulgarian',    'bul',      'bu',
'bengali',      'ben',      'bn',
'breton',       'bre',      'br',
# 11 --- 20
'bosnian',      'bos',      'bs',
'catalan',      'cat',      'ca',
'cebuano',      'ceb',      'cb',
'czech',        'cze',      'cs',
'welsh',        'wel',      'cy',
'danish',       'dan',      'da',
'german',       'ger',      'de',
'greek',        'gre',      'el',
'english',      'eng',      'en',
'spanish',      'spa',      'es',
# 21 --- 30

]

from swadesh_words import finnish, french, german, japanese, portuguese, spanish, ukrainian
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
        else:   print ('[ERROR]: Could not find swadesh-110 list for \'%s\'.' % lang)
        
    elif version == 'swadesh-207':
        # 1 --- 10
        if   lang=='afrikaans'          or lang=='afr' or lang=='af': return afrikaans.afr_207
        elif lang=='amharic'            or lang=='amh' or lang=='am': return amharic.amh_207
        elif lang=='arabic'             or lang=='ara' or lang=='ar': return arabic.ara_207
        elif lang=='asturian'           or lang=='ast' or lang=='as': return asturian.ast_207
        elif lang=='azerbaijani'        or lang=='aze' or lang=='az': return azerbaijani.aze_207
        elif lang=='bashkir'            or lang=='bas' or lang=='ba': return bashkir.bas_207
        elif lang=='belarusian'         or lang=='bel' or lang=='be': return belarusian.bel_207
        elif lang=='bulgarian'          or lang=='bul' or lang=='bu': return bulgarian.bul_207
        elif lang=='bengali'            or lang=='ben' or lang=='bn': return bengali.ben_207
        elif lang=='breton'             or lang=='bre' or lang=='br': return breton.bre_207
        # 11 --- 20
        # TODO generate swadesh list -> elif lang=='bosnian'            or lang=='bos' or lang=='bs': return bosnian.bos_207
        elif lang=='catalan'            or lang=='cat' or lang=='ca': return catalan.cat_207
        elif lang=='cebuano'            or lang=='ceb' or lang=='cb': return cebuano.ceb_207
        elif lang=='czech'              or lang=='cze' or lang=='cs': return czech.cze_207
        elif lang=='welsh'              or lang=='wel' or lang=='cy': return welsh.wel_207
        elif lang=='danish'             or lang=='dan' or lang=='da': return danish.dan_207
        elif lang=='german'             or lang=='gar' or lang=='de': return german.deu_207
        elif lang=='greek'              or lang=='gre' or lang=='el': return greek.gre_207
        elif lang=='english'            or lang=='eng' or lang=='en': return english.eng_207
        elif lang=='spanish'            or lang=='spa' or lang=='es': return spanish.spa_207
        # 21 --- 30
       
        else:
            print ('[ERROR]: Could not find swadesh-207 list for \'%s\'.' % lang)