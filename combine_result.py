import sys
import os
import pandas
import ipdb

classes = ['exam', 'med', 'other', 'plan', 'explained',
           'timespent', 'signsymptom', 'sdoh', 'consult',
           'discharge', 'proc', 'thankyou', 'allergy']
idx2cls_nmae = {i: name for i, name in enumerate(classes)}

model_folder = sys.argv[1]
out_file = os.path.join(model_folder, 'result.xlsx')
num_fold = 5

pred = []
for fold_id in range(num_fold):
    with open(os.path.join(model_folder, str(fold_id), 'pred_results.txt'), 'r') as f:
        for line in f.readlines():
            idx, prob = line.split(':')
            idx = int(idx)
            prob = eval(prob)

            pred += [(idx, prob)]
pred.sort(key=lambda tup: tup[0])

train_file = 'data/train/clinicalSTS2019.train.V2.master.single_sent.xlsx'
data = pandas.read_excel(train_file)
assert len(pred) == len(data)

for i, class_name in enumerate(classes):
    data[class_name] = [p[i] for _, p in pred]
data.to_excel(out_file, index=None)
