from __future__ import print_function
import numpy as np
import sys, os
import xlsxwriter

try:
    in_folder = sys.argv[1]
    out_name = sys.argv[2]
except:
    pass

#read results folder
contents = os.listdir(in_folder)
num_exps = len(contents)
out_name = os.path.join(in_folder, out_name)

#init workbook
workbook = xlsxwriter.Workbook(out_name)
worksheet = workbook.add_worksheet()

#define styles
values_format = workbook.add_format({'align': 'center','border': 1})
blank_format = workbook.add_format({'align': 'center','border': 1})
bestvalue_format = workbook.add_format({'align': 'center', 'bold':True, 'border':1, 'bg_color':'green'})
bestvalueSTD_format = workbook.add_format({'align': 'center', 'bold':True, 'border':1, 'bg_color':'blue'})
header_format = workbook.add_format({'align': 'center', 'bold':True,'border': 1, 'bg_color':'green'})
parameters_format = workbook.add_format({'align': 'center', 'bold':True,'border': 1, 'bg_color':'yellow'})
accuracy_format = workbook.add_format({'align': 'center', 'bold':True,'border': 1, 'bg_color':'orange'})
loss_format = workbook.add_format({'align': 'center', 'bold':True,'border': 1, 'bg_color':'red'})
percs_format = workbook.add_format({'align': 'center', 'bold':True,'border': 1, 'bg_color':'#800080'})
separation_border = workbook.add_format({'border': 1,'bottom': 6, 'bottom_color':'#ff0000'})

#define column names
exp_id_c = 0
comment1_c = 1
comment2_c = 2
train_acc_c = 3
val_acc_c = 4
test_acc_c = 5
train_acc_std_c = 6
val_acc_std_c = 7
test_acc_std_c = 8
train_loss_c = 9
val_loss_c = 10
test_loss_c = 11
train_loss_std_c = 12
val_loss_std_c = 13
test_loss_std_c = 14

v_offset = 2
v_end = v_offset + num_exps + 1


#write header
#title
worksheet.merge_range(v_offset-2, exp_id_c, v_offset-2, test_loss_std_c, "RESULTS", header_format)
#parameters
worksheet.merge_range(v_offset-1,exp_id_c, v_offset-1, comment2_c, "PARAMETERS", parameters_format)
#mean and std acc and loss
worksheet.merge_range(v_offset-1,train_acc_c, v_offset-1, test_acc_c, " MEAN ACCURACY", accuracy_format)
worksheet.merge_range(v_offset-1,train_loss_c, v_offset-1, test_loss_c, "MEAN LOSS", loss_format)
worksheet.merge_range(v_offset-1,train_acc_std_c, v_offset-1, test_acc_std_c, "ACCURACY STD", accuracy_format)
worksheet.merge_range(v_offset-1,train_loss_std_c, v_offset-1, test_loss_std_c, "LOSS STD", loss_format)

#write column names
worksheet.write(v_offset, exp_id_c, 'ID',parameters_format)
worksheet.set_column(exp_id_c,exp_id_c,6)

worksheet.write(v_offset, comment1_c, 'comment 1',parameters_format)
worksheet.set_column(comment1_c,comment1_c,35)

worksheet.write(v_offset, comment2_c, 'comment 2',parameters_format)
worksheet.set_column(comment2_c,comment2_c,35)

worksheet.write(v_offset, train_acc_c, 'train',accuracy_format)
worksheet.write(v_offset, val_acc_c, 'val',accuracy_format)
worksheet.write(v_offset, test_acc_c, 'test',accuracy_format)
worksheet.write(v_offset, train_loss_c, 'train',loss_format)
worksheet.write(v_offset, val_loss_c, 'val',loss_format)
worksheet.write(v_offset, test_loss_c, 'test',loss_format)

worksheet.write(v_offset, train_acc_std_c, 'train',accuracy_format)
worksheet.write(v_offset, val_acc_std_c, 'val',accuracy_format)
worksheet.write(v_offset, test_acc_std_c, 'test',accuracy_format)
worksheet.write(v_offset, train_loss_std_c, 'train',loss_format)
worksheet.write(v_offset, val_loss_std_c, 'val',loss_format)
worksheet.write(v_offset, test_loss_std_c, 'test',loss_format)

worksheet.set_column(train_acc_c,test_loss_std_c,10)

#fill values
#iterate every experiment
for i in contents:
    if '.npy' in i:
        temp_path = os.path.join(in_folder, i)
        dict = np.load(temp_path)
        dict = dict.item()
        keys = dict[0].keys()

        #print grobal parameters
        #write ID
        exp_ID = i.split('_')[-1].split('.')[0][3:]
        curr_row = int(exp_ID)+v_offset
        worksheet.write(curr_row, exp_id_c, exp_ID,values_format)
        #write comment
        parameters = dict['summary']['parameters'].split('/')
        comment_1 = '/'
        comment_2 = '/'
        for i in parameters:
            if 'comment' in i:
                comment_1 = i.split('=')[1].replace('"', '')
            if 'comment_2' in i:
                comment_2 = i.split('=')[1].replace('"', '')
        worksheet.write(curr_row, comment1_c, comment_1,values_format)
        worksheet.write(curr_row, comment2_c, comment_2,values_format)


        #extract results
        #training results
        tr = dict['summary']['training']
        tr_acc = tr['mean_acc']
        tr_loss = tr['mean_loss']
        tr_acc_std = tr['acc_std']
        tr_loss_std = tr['loss_std']
        #validation results
        val = dict['summary']['validation']
        val_acc = val['mean_acc']
        val_loss = val['mean_loss']
        val_acc_std = val['acc_std']
        val_loss_std = val['loss_std']
        #test results
        test = dict['summary']['test']
        test_acc = test['mean_acc']
        test_loss = test['mean_loss']
        test_acc_std = test['acc_std']
        test_loss_std = test['loss_std']

        #print results
        #acc
        worksheet.write(curr_row, train_acc_c, tr_acc,values_format)
        worksheet.write(curr_row, val_acc_c, val_acc,values_format)
        worksheet.write(curr_row, test_acc_c, test_acc,values_format)
        #acc std
        worksheet.write(curr_row, train_acc_std_c, tr_acc_std,values_format)
        worksheet.write(curr_row, val_acc_std_c, val_acc_std,values_format)
        worksheet.write(curr_row, test_acc_std_c, test_acc_std,values_format)
        #loss
        worksheet.write(curr_row, train_loss_c, tr_loss,values_format)
        worksheet.write(curr_row, val_loss_c, val_loss,values_format)
        worksheet.write(curr_row, test_loss_c, test_loss,values_format)
        #loss std
        worksheet.write(curr_row, train_loss_std_c, tr_loss_std,values_format)
        worksheet.write(curr_row, val_loss_std_c, val_loss_std,values_format)
        worksheet.write(curr_row, test_loss_std_c, test_loss_std,values_format)



explist = list(range(1, num_exps+1))
endlist = []
#apply blank formatting to blank and non-ending lines
#this is necessary due to a bug
for end in explist:
    #if end not in endlist:
    if end not in []:
        worksheet.conditional_format( v_offset+end,0,v_offset+end,test_loss_std_c, {'type': 'blanks','format': blank_format})

#highlight best values
#loss
worksheet.conditional_format(v_offset, train_loss_c, v_offset+num_exps, train_loss_c,
                                {'type': 'bottom','value': '1','format': bestvalue_format})

worksheet.conditional_format(v_offset, val_loss_c, v_offset+num_exps, val_loss_c,
                                {'type': 'bottom','value': '1','format': bestvalue_format})

worksheet.conditional_format(v_offset, test_loss_c, v_offset+num_exps, test_loss_c,
                                {'type': 'bottom','value': '1','format': bestvalue_format})
#loss std
worksheet.conditional_format(v_offset, train_loss_std_c, v_offset+num_exps, train_loss_std_c,
                                {'type': 'bottom','value': '1','format': bestvalueSTD_format})

worksheet.conditional_format(v_offset, val_loss_std_c, v_offset+num_exps, val_loss_std_c,
                                {'type': 'bottom','value': '1','format': bestvalueSTD_format})

worksheet.conditional_format(v_offset, test_loss_std_c, v_offset+num_exps, test_loss_std_c,
                                {'type': 'bottom','value': '1','format': bestvalueSTD_format})


#acc
worksheet.conditional_format(v_offset, train_acc_c, v_offset+num_exps, train_acc_c,
                                {'type': 'top','value': '1','format': bestvalue_format})

worksheet.conditional_format(v_offset, val_acc_c, v_offset+num_exps, val_acc_c,
                                {'type': 'top','value': '1','format': bestvalue_format})

worksheet.conditional_format(v_offset, test_acc_c, v_offset+num_exps, test_acc_c,
                                {'type': 'top','value': '1','format': bestvalue_format})
#acc std
worksheet.conditional_format(v_offset, train_acc_std_c, v_offset+num_exps, train_acc_std_c,
                                {'type': 'bottom','value': '1','format': bestvalueSTD_format})

worksheet.conditional_format(v_offset, val_acc_std_c, v_offset+num_exps, val_acc_std_c,
                                {'type': 'bottom','value': '1','format': bestvalueSTD_format})

worksheet.conditional_format(v_offset, test_acc_std_c, v_offset+num_exps, test_acc_std_c,
                                {'type': 'bottom','value': '1','format': bestvalueSTD_format})


workbook.close()
