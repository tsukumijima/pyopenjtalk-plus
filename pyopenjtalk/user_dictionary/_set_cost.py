#linuxでpypyで実行することを推奨します(pypiではなく高速化されたpythonを使用してください)
#じゃないと遅い
#pypyはpyenvからインストールを推奨

#def set sort_cost1():

from tqdm import tqdm
"""
openjtalk本体辞書のコストを計算するコードのつもりだったが個別の追加辞書に適用した方がよいかもしれない
ルールは以下
まず名前順にソートする
その後上から順に次のルールを適用する
現在の行の表層系が2文字以上の場合で
上の文字が2文字以上でかつ現在の文字に含まれる場合
現在の文字のコストを上の文字のコストに１引いたものにする(定数はお好みで、というより試しながらやるしかない)
"""
with open("./pyopenjtalk/user_dictionary/english.csv", "r", encoding="utf-8") as f:
    data = f.read()
data_list = data.split("\n")
data_list.sort(reverse=True)

bak_line = "<dummy>"
bak_cost = "0"
#最初の一文字目はスキップ
skip = True
out = []
for line in tqdm(data_list):
    
    split_line = line.split(",",4)
    if len(split_line[0]) >= 2 and len(bak_line) >= 2:
        #先頭一致
        if  bak_line.find(split_line[0]) != -1:
            #完全一致は除外
            if bak_line != split_line[0]:
                #分割するのは使うとこまでじゃないと多分遅くなる
                
                split_line = split_line[:3] +[str(int(bak_cost) + 500)] + [split_line[4]]
                line = ",".join(split_line)
            else:
                continue
        if len(split_line[0]) == 2:
            split_line = split_line[:3] +["10000"] + [split_line[4]]
            line = ",".join(split_line)


    bak_line = split_line[0]
    bak_cost = split_line[3]


    line = ",".join(split_line)
    out.append(line)

with open("./pyopenjtalk/user_dictionary/english.csv", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
