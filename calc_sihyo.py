def calc_precicsion(tp=0, fp=0):
    pre = tp / (tp + fp)
    return pre

# 再現率の算出
def calc_recall(tp=0, fn=0):
    rec = tp / (tp + fn)
    return rec

# F値の算出
def calc_f(pre=0, rec=0):
    f = (2 * pre * rec) / (pre + rec)
    return f


tp_s=90
tp_f=142-90
fp_f=106
fp_s=112-106
pre=calc_precicsion(tp_s,fp_s)
rec = calc_recall(tp_s,tp_f)
f=calc_f(pre,rec)
print("正解文",'適合率',pre,'再現率',rec,'f値',f)

pre=calc_precicsion(fp_f,tp_f)
rec = calc_recall(fp_f,fp_s)
f=calc_f(pre,rec)
print("不正解文",'適合率',pre,'再現率',rec,'f値',f)

print(tp_s+fp_s)
print(tp_f+fp_f)