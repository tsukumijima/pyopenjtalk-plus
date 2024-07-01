from sudachipy import tokenizer
from sudachipy import dictionary
from .yomi_model.nani_predict import predict

def merge_njd_marine_features(njd_features, marine_results):
    features = []

    marine_accs = marine_results["accent_status"]
    marine_chain_flags = marine_results["accent_phrase_boundary"]

    assert (
        len(njd_features) == len(marine_accs) == len(marine_chain_flags)
    ), "Invalid sequence sizes in njd_results, marine_results"

    for node_index, njd_feature in enumerate(njd_features):
        _feature = {}
        for feature_key in njd_feature.keys():
            if feature_key == "acc":
                _feature["acc"] = int(marine_accs[node_index])
            elif feature_key == "chain_flag":
                _feature[feature_key] = int(marine_chain_flags[node_index])
            else:
                _feature[feature_key] = njd_feature[feature_key]
        features.append(_feature)
    return features


def modify_kanji_yomi(text, pyopen_njd, multi_read_kanji_list):
    sudachi_yomi = sudachi_analyze(text, multi_read_kanji_list)
    return_njd = []
    pre_dict = None

    for dict in reversed(pyopen_njd):
        if dict['orig'] in multi_read_kanji_list:
            try:
                correct_yomi = sudachi_yomi.pop()
            except:
                return pyopen_njd
            if correct_yomi[0] != dict['orig']:
                return pyopen_njd
            elif dict['orig'] == '何':
                is_read_nan = predict([pre_dict])
                if is_read_nan == 1:
                    dict['pron'] = 'ナン'
                    dict['read'] = 'ナン'
                else:
                    dict['pron'] = 'ナニ'
                    dict['read'] = 'ナニ'
                return_njd.append(dict)

            else:
                if correct_yomi[0] == '方' and correct_yomi[1] == 'ホウ':
                    correct_yomi[1] = 'ホオ'
                dict['pron'] = correct_yomi[1]
                dict['read'] = correct_yomi[1]
                return_njd.append(dict)
        else:
            return_njd.append(dict)
        pre_dict = dict

    return_njd.reverse()
    return return_njd

def sudachi_analyze(text, multi_read_kanji_list):
    """
    複数の読み方をする漢字の読みをsudachiで形態素解析した結果からリストで返却
    例: 風がこんな風に吹く→[('風', 'カゼ'), ('風', 'フウ')]
    ----------------------------------------------------
    parameters
    text : str
        読み対象となるテキスト
    multi_read_kanji_list : list[chr]
        複数の読み方をする漢字のリスト(ex : 何、風、方)
    ----------------------------------------------------
    return
    yomi_list : (漢字,カタカナ読み)のタプルリスト
    """
    text = text.replace('ー','')
    tokenizer_obj = dictionary.Dictionary().create()
    mode = tokenizer.Tokenizer.SplitMode.C
    m_list = tokenizer_obj.tokenize(text, mode)
    yomi_list = [[m.surface(),m.reading_form()] for m in m_list if m.surface() in multi_read_kanji_list]
    return yomi_list

def retreat_acc_nuc(njd_features):
    """
    長母音、重母音、撥音がアクセント核に来た場合にひとつ前のモーラにアクセント核がズレるルールの実装

    Args:
        njd_features (list): run_frontendの結果
    """

    inappropriate_for_nuclear_chars = ["ー", "ッ" ,"ン"]
    delete_youon = str.maketrans('', '', 'ャュョァィゥェォ')
    for i, njd in enumerate(njd_features):
        # アクセント境界直後のnode(chain_flag 0 or -1)にアクセント核の位置の情報が入っている
        if njd["chain_flag"] in [0, -1]:
            head = njd
            acc = njd["acc"]
            phase_len = 0
        
        phase_len +=  njd["mora_size"]
        pron = njd["pron"].translate(delete_youon)
        if len(pron) == 0:
            pron = njd["pron"]
            
        if acc > 0:
            if acc <= njd["mora_size"]:
                nuc_pron = pron[acc-1]
                if nuc_pron in inappropriate_for_nuclear_chars:
                    head["acc"] += -1 
                acc = -1
            else:
                acc = acc - njd["mora_size"]


    return njd_features
