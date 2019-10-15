import numpy as np
from scipy.stats import wilcoxon
import os

folder = '../selected_exps'

normal_conv_accs = np.array([64.3, 66.26, 66.91, 42.09, 39.84, 42.56, 47.41, 47.45, 49.6, 50.61, 40.78, 48.93, 50.48, 49.0, 54.96])

multi_conv_accs = np.array([66.5, 70.97, 70.68, 47.85, 44.95, 51.32, 55.85, 51.76, 48.75, 53.05, 51.71, 49.0, 50.84, 49.86, 55.01])



iemo =  [48.93, 50.48, 49.0, 54.96]
iemo2 = [49.0, 50.84, 49.86, 55.01]

w, p = wilcoxon(iemo, iemo2)

#print (w, p)

contents = os.listdir(folder)
contents = [os.path.join(folder, x) for x in contents]
contents = list(filter(lambda x: 'conv' in x, contents))
arch = {'a1':{'conv':[],'multi':[]},
        'a2':{'conv':[],'multi':[]},
        'a3':{'conv':[],'multi':[]},
        'a4':{'conv':[],'multi':[]}}
datasets = {'emodb':{'conv':[],'multi':[]},
            'ravdess':{'conv':[],'multi':[]},
            'tess':{'conv':[],'multi':[]},
            'iemocap':{'conv':[],'multi':[]}}
all = {'conv':[],'multi':[]}


for i in contents:
    c = i
    m = i.replace('conv' , 'multi')
    curr_dataset = i.split('/')[-1].split('_')[0]
    curr_arch = i.split('/')[-1].split('_')[1]


    res = np.load(c, allow_pickle=True).item()
    keys = list(res.keys())
    keys.remove('summary')
    for k in keys:
        arch[curr_arch]['conv'].append(res[k]['test_acc'])
        datasets[curr_dataset]['conv'].append(res[k]['test_acc'])
        all['conv'].append(res[k]['test_acc'])

    res = np.load(m, allow_pickle=True).item()
    keys = list(res.keys())
    keys.remove('summary')
    for k in keys:
        arch[curr_arch]['multi'].append(res[k]['test_acc'])
        datasets[curr_dataset]['multi'].append(res[k]['test_acc'])
        all['multi'].append(res[k]['test_acc'])

'''
for i in contents:
    c = i
    m = i.replace('conv' , 'multi')
    curr_dataset = i.split('/')[-1].split('_')[0]
    curr_arch = i.split('/')[-1].split('_')[1]

    res = np.load(c, allow_pickle=True).item()
    arch[curr_arch]['conv'].append(res['summary']['test']['mean_acc'])
    datasets[curr_dataset]['conv'].append(res['summary']['test']['mean_acc'])
    all['conv'].append(res['summary']['test']['mean_acc'])

    res = np.load(m, allow_pickle=True).item()
    arch[curr_arch]['multi'].append(res['summary']['test']['mean_acc'])
    datasets[curr_dataset]['multi'].append(res['summary']['test']['mean_acc'])
    all['multi'].append(res['summary']['test']['mean_acc'])

'''


overall_w, overall_p = wilcoxon(all['conv'], all['multi'])
print ('overall:', overall_w, overall_p)

for i in arch:
    curr_w, curr_p = wilcoxon(arch[i]['conv'], arch[i]['multi'])
    print (i, curr_w, curr_p)

for i in datasets:
    curr_w, curr_p = wilcoxon(datasets[i]['conv'], datasets[i]['multi'])
    print (i, curr_w, curr_p)


results = {}
