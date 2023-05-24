import pickle
import pandas as pd

X_COLS = ['pos','pos_group1','pos_group2','pron','ctype','cform']
with open('/pyopenjtalk/pyopenjtalk/yomi_model/nani_enc.pickle', 'rb') as f:
    enc = pickle.load(f)

with open('/pyopenjtalk/pyopenjtalk/yomi_model/nani_model.pickle', 'rb') as f:
    model = pickle.load(f)

def predict(input_njd):
    if input_njd == [None]:
        return 0
    else:
        input_df = pd.DataFrame(input_njd)[X_COLS]
        input = enc.transform(input_df)
        return int(model.predict(input)[0])
