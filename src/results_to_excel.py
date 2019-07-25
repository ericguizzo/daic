import numpy as np
import sys, os
import xlsxwriter

highlight_bounds = [(1,12),(13,24),(25,36),(37,48)]
num_exps = 48

try:
    in_folder = sys.argv[1]
    out_name = sys.argv[2]
except:
    pass


contents = os.listdir(in_folder)
out_name = os.path.join(in_folder, out_name)

workbook = xlsxwriter.Workbook(out_name)
worksheet = workbook.add_worksheet()

values_format = workbook.add_format({'align': 'center','border': 1})
blank_format = workbook.add_format({'align': 'center','border': 1})
bestvalue_format = workbook.add_format({'align': 'center', 'bold':True, 'border':1, 'bg_color':'green'})
header_format = workbook.add_format({'align': 'center', 'bold':True,'border': 1, 'bg_color':'green'})
parameters_format = workbook.add_format({'align': 'center', 'bold':True,'border': 1, 'bg_color':'yellow'})
accuracy_format = workbook.add_format({'align': 'center', 'bold':True,'border': 1, 'bg_color':'orange'})
loss_format = workbook.add_format({'align': 'center', 'bold':True,'border': 1, 'bg_color':'red'})
percs_format = workbook.add_format({'align': 'center', 'bold':True,'border': 1, 'bg_color':'#800080'})
conv_format = workbook.add_format({'align': 'center', 'bold':True,'border': 1, 'bg_color':'blue'})
multi_format = workbook.add_format({'align': 'center','border': 1, 'bg_color':'cyan'})
separation_border = workbook.add_format({'border': 1,'bottom': 6, 'bottom_color':'#ff0000'})

exp_id_c = 0
conv_type_c = 1
stretch_factors_c = 2
channels_c= 3
architecture_c = 4
regularization_c = 5

train_acc_c = 6
val_acc_c = 7
test_acc_c = 8
train_loss_c = 9
val_loss_c = 10
test_loss_c = 11
train_perc_stretches_c = 12
val_perc_stretches_c = 13
test_perc_stretches_c = 14

v_offset_BVL = 2
v_end_BVL = v_offset_BVL + num_exps + 1



worksheet.merge_range(v_offset_BVL-2, exp_id_c, v_offset_BVL-2, test_perc_stretches_c, "BEST VALIDATION LOSS MODEL", header_format)
worksheet.merge_range(v_offset_BVL-1,exp_id_c, v_offset_BVL-1, regularization_c, "PARAMETERS", parameters_format)
worksheet.merge_range(v_offset_BVL-1,train_acc_c, v_offset_BVL-1, test_acc_c, "ACCURACY", accuracy_format)
worksheet.merge_range(v_offset_BVL-1,train_loss_c, v_offset_BVL-1, test_loss_c, "LOSS", loss_format)
worksheet.merge_range(v_offset_BVL-1,train_perc_stretches_c, v_offset_BVL-1, test_perc_stretches_c, "PERCENTAGE OF USED STRETCHES (same order as stretch factors)", percs_format)

#write column names
worksheet.write(v_offset_BVL, exp_id_c, 'ID',parameters_format)
worksheet.set_column(exp_id_c,exp_id_c,6)
worksheet.write(v_offset_BVL, conv_type_c, 'conv type',parameters_format)
worksheet.set_column(conv_type_c,conv_type_c,10)
worksheet.write(v_offset_BVL, stretch_factors_c, 'stretch factors',parameters_format)
worksheet.set_column(stretch_factors_c,stretch_factors_c,35)
worksheet.write(v_offset_BVL, channels_c, 'channels',parameters_format)
worksheet.set_column(channels_c,channels_c,8)
worksheet.write(v_offset_BVL, architecture_c, 'architecture',parameters_format)
worksheet.set_column(architecture_c,architecture_c, 12)
worksheet.write(v_offset_BVL, regularization_c, 'regularization',parameters_format)
worksheet.set_column(regularization_c,regularization_c,12)


worksheet.write(v_offset_BVL, train_loss_c, 'train loss',loss_format)
worksheet.write(v_offset_BVL, val_loss_c, 'val loss',loss_format)
worksheet.write(v_offset_BVL, test_loss_c, 'test loss',loss_format)
worksheet.write(v_offset_BVL, train_perc_stretches_c, 'train perc stretches',percs_format)
worksheet.write(v_offset_BVL, val_perc_stretches_c, 'val perc stretches',percs_format)
worksheet.write(v_offset_BVL, test_perc_stretches_c, 'test perc stretches',percs_format)

worksheet.set_column(train_acc_c,test_loss_c,10)
worksheet.set_column(train_perc_stretches_c,test_perc_stretches_c,40)



#iterate every experiment
for i in contents:
    if '.npy' in i:
        temp_path = os.path.join(in_folder, i)
        dict = np.load(temp_path)
        dict = dict.item()
        keys = dict[0].keys()

        #print grobal parameters
        exp_ID = i.split('_')[-1].split('.')[0][3:]
        curr_row_BVL = int(exp_ID)+v_offset_BVL
        worksheet.write(curr_row_BVL, exp_id_c, exp_ID,values_format)
        parameters = dict['summary']['parameters'].split('/')
        regularization_lambda = 0.001  #default parameter
        for param in parameters:
            exec(param)
        if layer_type == 'conv':
            layer_format = conv_format
        elif layer_type == 'multi':
            layer_format = multi_format
        worksheet.write(curr_row_BVL, conv_type_c, layer_type,layer_format)
        worksheet.write(curr_row_BVL, regularization_c, regularization_lambda, values_format)
        stretch_factors_cut = [1.]
        for i in stretch_factors:
            stretch_factors_cut.append(i[0])
        if layer_type == 'conv':
            stretch_factors_cut = 'nan'
        worksheet.write(curr_row_BVL, stretch_factors_c, str(stretch_factors_cut),values_format)
        if network_type == '3layers':
            channels = '10, 20, 30'
        worksheet.write(curr_row_BVL, channels_c, channels,values_format)
        worksheet.write(curr_row_BVL, architecture_c, network_type,values_format)

        #init vectors
        for key in keys:
            init_string = key + '=[]'
            exec (init_string)

        #iterate folds, append values
        for fold in dict.keys():
            if fold != 'summary':  #bypass summary
                #iterate every key in a fold
                for key in keys:
                    key_content = np.array(dict[fold][key])
                    append_string = key + '.append(key_content)'
                    exec (append_string)

        #iterate folds, compute mean values
        for fold in dict.keys():
            if fold != 'summary':  #bypass summary
                #iterate every key in a fold
                for key in keys:
                    mean_string = key + '_mean = np.round(np.mean(' + key + ', axis=0), decimals=3)'
                    try:
                        exec (mean_string)
                    except ValueError:
                        pass

        #print results to file
        worksheet.write(curr_row_BVL, train_acc_c, train_acc_BVL_mean,values_format)
        worksheet.write(curr_row_BVL, val_acc_c, val_acc_BVL_mean,values_format)
        worksheet.write(curr_row_BVL, test_acc_c, test_acc_BVL_mean,values_format)

        worksheet.write(curr_row_BVL, train_loss_c, train_loss_BVL_mean,values_format)
        worksheet.write(curr_row_BVL, val_loss_c, val_loss_BVL_mean,values_format)
        worksheet.write(curr_row_BVL, test_loss_c, test_loss_BVL_mean,values_format)

        worksheet.write(curr_row_BVL, train_perc_stretches_c, str(train_stretch_percs_BVL_mean),values_format)
        worksheet.write(curr_row_BVL, val_perc_stretches_c, str(train_stretch_percs_BVL_mean),values_format)
        worksheet.write(curr_row_BVL, test_perc_stretches_c, str(test_stretch_percs_BVL_mean),values_format)

explist = list(range(1, num_exps+1))
endlist = []
#print red h lines of highlighting bounds
for begin,end in highlight_bounds:
    endlist.append(end)
    worksheet.conditional_format( v_offset_BVL+end,0,v_offset_BVL+end,regularization_c, { 'type' : 'no_errors' , 'format' : separation_border})
#apply blank formatting to blank and non-ending lines
#this is necessary due to a bug

for end in explist:
    #if end not in endlist:
    if end not in []:
        worksheet.conditional_format( v_offset_BVL+end,0,v_offset_BVL+end,test_perc_stretches_c, {'type': 'blanks','format': blank_format})


#highlight best accs
for begin,end in highlight_bounds:
    print(v_offset_BVL+begin,v_offset_BVL+end)

    worksheet.conditional_format(v_offset_BVL+begin, train_acc_c, v_offset_BVL+end, train_acc_c,
                                {'type': 'top','value': '1','format': bestvalue_format})
    worksheet.conditional_format(v_offset_BVA+begin, train_acc_c, v_offset_BVA+end, train_acc_c,
                                {'type': 'top','value': '1','format': bestvalue_format})

    worksheet.conditional_format(v_offset_BVL+begin, val_acc_c, v_offset_BVL+end, val_acc_c,
                                {'type': 'top','value': '1','format': bestvalue_format})

    worksheet.conditional_format(v_offset_BVA+begin, val_acc_c, v_offset_BVA+end, val_acc_c,
                                {'type': 'top','value': '1','format': bestvalue_format})

    worksheet.conditional_format(v_offset_BVL+begin, test_acc_c, v_offset_BVL+end, test_acc_c,
                                {'type': 'top','value': '1','format': bestvalue_format})
    worksheet.conditional_format(v_offset_BVA+begin, test_acc_c, v_offset_BVA+end, test_acc_c,
                                {'type': 'top','value': '1','format': bestvalue_format})

    #highlight best losses
    worksheet.conditional_format(v_offset_BVL+begin, train_loss_c, v_offset_BVL+end, train_loss_c,
                                {'type': 'bottom','value': '1','format': bestvalue_format})
    worksheet.conditional_format(v_offset_BVA+begin, train_loss_c, v_offset_BVA+end, train_loss_c,
                                {'type': 'bottom','value': '1','format': bestvalue_format})

    worksheet.conditional_format(v_offset_BVL+begin, val_loss_c, v_offset_BVL+end, val_loss_c,
                                {'type': 'bottom','value': '1','format': bestvalue_format})
    worksheet.conditional_format(v_offset_BVA+begin, val_loss_c, v_offset_BVA+end, val_loss_c,
                                {'type': 'bottom','value': '1','format': bestvalue_format})

    worksheet.conditional_format(v_offset_BVL+begin, test_loss_c, v_offset_BVL+end, test_loss_c,
                                {'type': 'bottom','value': '1','format': bestvalue_format})
    worksheet.conditional_format(v_offset_BVA+begin, test_loss_c, v_offset_BVA+end, test_loss_c,
                                {'type': 'bottom','value': '1','format': bestvalue_format})


workbook.close()
