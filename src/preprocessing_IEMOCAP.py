import utility_functions as uf

label_to_int = {'neu':0,
                'ang':1,
                'fru':2,
                'hap':3,
                'sad':4,
                'exc':5,
                'fea':6,
                'sur':7,
                'dis':8,
                'xxx':9}
num_classes_IEMOCAP = 10
wavname = 'Ses01F_impro01_F001.wav'

def get_label_IEMOCAP(wavname):
    session = int(wavname.split('_')[0][3:5])
    trans_file = '_'.join(wavname.split('_')[:2]) + '.txt'
    ID = wavname.split('.')[0]
    print (ID)

    trans_file = '/home/eric/Desktop/Ses01F_impro01.txt'
    with open(trans_file) as f:
        contents = f.readlines()

    str_label = list(filter(lambda x: ID in x, contents))[0].split('\t')[2]
    int_label = label_to_int[str_label]
    one_hot_label = uf.onehot(int_label, num_classes_IEMOCAP)
    print (one_hot_label)



get_label_IEMOCAP(wavname)
