# 1 --- 10
from swadesh_words import afrikaans, amharic, arabic, asturian, azerbaijani, bashkir, belarusian, bulgarian, bengali, breton
# 11 --- 20
from swadesh_words import bosnian, catalan, cebuano, czech, welsh, danish, german, greek, english, spanish
# 21 --- 30
from swadesh_words import estonian, persian, fulah, finnish, french, frisian, irish, gaelic, galician, gujarati
# 31 --- 40
from swadesh_words import hausa, hebrew, hindi, croatian, haitian, hungarian, armenian, indonesian, igbo, iloko
# 41 --- 50
from swadesh_words import icelandic, italian, japanese, javanese, georgian, kazakh, khmer, kannada, korean, luxembourgish
# 51 --- 60
from swadesh_words import ganda, lingala, lao, lithuanian, latvian, malagasy, macedonian, malayalam, mongolian, marathi
# 61 --- 70
from swadesh_words import malay, burmese, nepali, dutch, norwegian, sotho, occitan, oriya, punjabi, polish
# 71 --- 80


''' all inclusive list '''
all_langs = ['afrikaans','amharic','arabic','asturian','azerbaijani','bashkir','belarusian','bulgarian','bengali','breton',
            'bosnian','catalan','cebuano','czech','welsh','danish','german','greek','english','spanish','estonian','persian',
            'fulah','finnish','french','frisian','irish','gaelic','galician','gujarati','hausa','hebrew','hindi','croatian',
            'haitian','hungarian','armenian','indonesian','igbo','iloko','icelandic','italian','japanese','javanese','georgian',
            'kazakh','khmer','kannada','korean','luxembourgish','ganda','lingala','lao','lithuanian','latvian','malagasy',
            'macedonian','malayalam','mongolian','marathi','malay','burmese','nepali','dutch','norwegian','sotho','occitan',
            'oriya','punjabi','polish']
not_implemented_langs = ['bosnian','fulah','igbo','kazakh','luxembourgish','ganda','lao','sotho','oriya']



def get_all_langs(): 
    all = []
    for i in range(len(all_langs)):
        if all_langs[i] not in not_implemented_langs:
            all.append(all_langs[i])
    return all

''' languages with swadesh word lists '''
valid_languages = [
# 1 --- 10
'afrikaans',    'afr', 'af',
'amharic',      'amh', 'am',
'arabic',       'ara', 'ar',
'asturian',     'ast', 'as',
'azerbaijani',  'aze', 'az',
'bashkir',      'bas', 'ba',
'belarusian',   'bel', 'be',
'bulgarian',    'bul', 'bu',
'bengali',      'ben', 'bn',
'breton',       'bre', 'br',
# 11 --- 20
'bosnian',      'bos', 'bs',
'catalan',      'cat', 'ca',
'cebuano',      'ceb', 'cb',
'czech',        'cze', 'cs',
'welsh',        'wel', 'cy',
'danish',       'dan', 'da',
'german',       'ger', 'de',
'greek',        'gre', 'el',
'english',      'eng', 'en',
'spanish',      'spa', 'es',
# 21 --- 30
'estonian',     'est', 'et',
'persian',      'per', 'fa',
'fulah',        'ful', 'ff',
'finnish',      'fin', 'fi',
'french',       'fre', 'fr',
'frisian',      'fri', 'fy',
'irish',        'iri', 'ga',
'gaelic',       'gae', 'gd',
'galician',     'gal', 'gl',
'gujarati',     'guj', 'gu',
# 31 --- 40
'hausa',        'hau', 'ha',
'hebrew',       'heb', 'he',
'hindi',        'hin', 'hi',
'croatian',     'cro', 'hr',
'haitian',      'hai', 'ht',
'hungarian',    'hun', 'hu',
'armenian',     'arm', 'hy',
'indonesian',   'ind', 'id',
'igbo',         'igb', 'ig',
'iloko',        'ilo', 'il',
# 41 --- 50
'icelandic',    'ice', 'is',
'italian',      'ita', 'it',
'japanese',     'jap', 'ja',
'javanese',     'jav', 'jv',
'georgian',     'geo', 'ka',
'kazakh',       'kaz', 'kk',
'khmer',        'khm', 'km',
'kannada',      'kan', 'kn',
'korean',       'kor', 'ko',
'luxembourgish','lux', 'lb',
# 51 --- 60
'ganda',        'gan', 'lg',
'lingala',      'lin', 'ln',
'lao',          'lao', 'lo',
'lithuanian',   'lit', 'lt',
'latvian',      'lat', 'lv',
'malagasy',     'mal', 'mg',
'macedonian',   'mac', 'mk',
'malayalam',    'mal', 'ml',
'mongolian',    'mon', 'mn',
'marathi',      'mar', 'mr',
# 61 --- 70
'malay',        'may', 'ms',
'burmese',      'bur', 'my',
'nepali',       'nap', 'ne',
'dutch',        'dut', 'nl',
'norwegian',    'nor', 'no',
'sotho',        'sot', 'ns',
'occitan',      'occ', 'oc',
'oriya',        'ori', 'or',
'punjabi',      'pun', 'pa',
'polish',       'pol', 'pl',
# 71 --- 80


]

from swadesh_words import german, portuguese, spanish, ukrainian
def get_swadesh_words(lang: str, version: str):
    assert lang in valid_languages
    
    # only 8 languages available for swadesh-110
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
        elif lang=='estonian'           or lang=='est' or lang=='et': return estonian.est_207
        elif lang=='persian'            or lang=='per' or lang=='fa': return persian.per_207
        # TODO generate swadesh list -> elif lang=='fulah'            or lang=='ful' or lang=='ff': return fulah.ful_207
        elif lang=='finnish'            or lang=='fin' or lang=='fi': return finnish.fin_207
        elif lang=='french'             or lang=='fre' or lang=='fr': return french.fre_207
        elif lang=='frisian'            or lang=='fri' or lang=='fy': return frisian.fri_207
        elif lang=='irish'              or lang=='iri' or lang=='ga': return irish.iri_207
        elif lang=='gaelic'             or lang=='gae' or lang=='gd': return gaelic.gae_207
        elif lang=='galician'           or lang=='gal' or lang=='gl': return galician.gal_207
        elif lang=='gujarati'           or lang=='guj' or lang=='gu': return gujarati.guj_207
        # 31 --- 40
        elif lang=='hausa'              or lang=='hau' or lang=='ha': return hausa.hau_207
        elif lang=='hebrew'             or lang=='heb' or lang=='he': return hebrew.heb_207
        elif lang=='hindi'              or lang=='hin' or lang=='hi': return hindi.hin_207
        elif lang=='croatian'           or lang=='cro' or lang=='hr': return croatian.cro_207
        elif lang=='haitian'            or lang=='hai' or lang=='ht': return haitian.hai_207
        elif lang=='hungarian'          or lang=='hun' or lang=='hu': return hungarian.hun_207
        elif lang=='armenian'           or lang=='arm' or lang=='hy': return armenian.arm_207
        elif lang=='indonesian'         or lang=='ind' or lang=='id': return indonesian.ind_207
        # TODO generate swadesh list -> elif lang=='igbo'            or lang=='igb' or lang=='ig': return igbo.igb_207
        elif lang=='iloko'              or lang=='ilo' or lang=='il': return iloko.ilo_207
        # 41 --- 50
        elif lang=='icelandic'          or lang=='ice' or lang=='is': return icelandic.ice_207
        elif lang=='italian'            or lang=='ita' or lang=='it': return italian.ita_207
        elif lang=='japanese'           or lang=='jap' or lang=='ja': return japanese.jpn_hira_207
        elif lang=='javanese'           or lang=='jav' or lang=='jv': return javanese.jav_207
        elif lang=='georgian'           or lang=='geo' or lang=='ka': return georgian.geo_207
        # TODO generate swadesh list -> elif lang=='kazakh'            or lang=='kaz' or lang=='aa': return kazakh.kaz_207
        elif lang=='khmer'              or lang=='khm' or lang=='kk': return khmer.khm_207
        elif lang=='kannada'            or lang=='kan' or lang=='kn': return kannada.kan_207
        elif lang=='korean'             or lang=='kor' or lang=='ko': return korean.kor_207
        # TODO generate swadesh list -> elif lang=='luxembourgish'            or lang=='lux' or lang=='lb': return luxembourgish.lux_207
        # 51 --- 60
        # TODO generate swadesh list -> elif lang=='ganda'            or lang=='gan' or lang=='lg': return ganda.gan_207
        elif lang=='lingala'            or lang=='lin' or lang=='ln': return lingala.lin_207
        # TODO generate swadesh list -> elif lang=='lao'            or lang=='lao' or lang=='lo': return lao.lao_207
        elif lang=='lithuanian'         or lang=='lit' or lang=='lt': return lithuanian.lit_207
        elif lang=='latvian'            or lang=='lat' or lang=='lv': return latvian.lat_207
        elif lang=='malagasy'           or lang=='mal' or lang=='mg': return malagasy.mal_207
        elif lang=='macedonian'         or lang=='mac' or lang=='mk': return macedonian.mac_207
        elif lang=='malayalam'          or lang=='mal' or lang=='ml': return malayalam.mal_207
        elif lang=='mongolian'          or lang=='mon' or lang=='mn': return mongolian.mon_207
        elif lang=='marathi'            or lang=='mar' or lang=='mr': return marathi.mar_207
        # 61 --- 70
        elif lang=='malay'              or lang=='may' or lang=='ms': return malay.may_207
        elif lang=='burmese'            or lang=='bur' or lang=='my': return burmese.bur_207
        elif lang=='nepali'             or lang=='nap' or lang=='ne': return nepali.nep_207
        elif lang=='dutch'              or lang=='dut' or lang=='nl': return dutch.dut_207
        elif lang=='norwegian'          or lang=='nor' or lang=='no': return norwegian.nor_207
        # TODO generate swadesh list -> elif lang=='sotho'            or lang=='sot' or lang=='so': return sotho.sot_207
        elif lang=='occitan'            or lang=='occ' or lang=='oc': return occitan.occ_207
        # TODO generate swadesh list -> elif lang=='oriya'            or lang=='ori' or lang=='or': return oriya.ori_207
        elif lang=='punjabi'            or lang=='pun' or lang=='pa': return punjabi.pun_207
        elif lang=='polish'             or lang=='pol' or lang=='pl': return polish.pol_207
        # 71 --- 80
        
        else: print ('[ERROR]: Could not find swadesh-207 list for \'%s\'.' % lang)