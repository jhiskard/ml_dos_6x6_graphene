from sklearn import datasets, svm, metrics
import random
import os, sys
import numpy as np
import NanoCore as nc
from link import link


def mycmp(a, b): return cmp(a[0], b[0])


def predict(atom_category, input_list, broad=0.05, erange=10):
    # Initialize Data set path
    #path = "../../Graphene_6x6_doping/" + atom_category
    path = "../Graphene_6x6_doping/" + atom_category

    # Initialize default value
    data_set = []
    data_Ef = []
    target_list = []
    data_set_struct = []
    ref_xyz = 'ml/ref_coord2.xyz'
    #ref_xyz = 'ref_coord2.xyz'
    ref_at = nc.io.read_xyz(ref_xyz)
    tol = 0.01

    # Parse Data
    for i in os.listdir(path):
        for j in os.listdir(path + '/' + i):
            if j == 'DOS_%3.2f_%2.2i' % (broad, erange):

                # Parse Ef, DOS data
                f = open(path+'/'+i+'/'+j, 'r')
                f_lines = f.readlines()
                data_Ef.append( float(f_lines[9].split()[3]) )
                result = []
                for line in f_lines[13:]:
                    data = line.split('\n')[0].split(' ')
                    temp = []
                    for k in data:
                        if k != '':
                            temp.append(k)
                    result.append(np.float64(temp[1]))
                f.close()
                data_set.append(result)

                # Parse Struct data
                at = nc.siesta.read_fdf(path+'/'+i+'/STRUCT.fdf')
                i_str = 0
                struct_result = []

                for atm in at:
                    symb = atm.get_symbol()
                    x, y, z = atm.get_position()

                    # find the real serial number
                    serial = 1
                    for atm1 in ref_at:
                        x1, y1, z1 = atm1.get_position()
                        if abs(x1-x) < tol and abs(y1-y) < tol and abs(z1-z) < tol:
                            break
                        else:
                            serial += 1

                    # C or B/N?
                    if symb == 'C':
                        struct_result.append([serial, 2])
                    else:
                        struct_result.append([serial, 1])
                    i_str += 1

                #print "OUT: unsorted", np.array(struct_result)[:,0]
                struct_result.sort(mycmp)
                #print "OUT: SORTed", np.array(struct_result)[:,0]
                struct_result = list( np.array( np.array(struct_result)[:,1] ) )
                #print i, j

                data_set_struct.append(struct_result)

                # Add target_list
                target_list.append(i)

    #print data_set[:2]         # DOS value
    #print target_list[:2]      # folder name
    #print data_set_struct[:2]  # structure

    # Create a classifier: a support vector classifier
    classifier = svm.SVC(gamma=0.001)

    # Input
    c_list = []
    for i in range(72):
        c_list.append('2')
    temp = input_list.split(',')
    for item in temp:
        c_list[int(item)] = '1'

    link.sort(mycmp)
    link_arr = np.array(link)[:,1]

    c_list2 = np.array(np.zeros(72), dtype=int)
    i_c = 0
    for l in link_arr:
        c_list2[l-1] = c_list[i_c]
        i_c += 1
    c_list = list( np.array(c_list2, dtype=int) )
    print "input list:", c_list

    # identical configuration?
    i = 0
    ex_i = -1
    for struct in data_set_struct:
        print struct[:10]
        if struct == c_list:
            ex_i = i
            break
        i += 1
    print "ex_i =", ex_i

    expected_data = np.zeros(1001)
    if ex_i == -1:
        print "OUT: No identical configuration"
        classifier.fit( data_set_struct, target_list )

    else:
        print "OUT: Exclude the identical configuration"
        classifier.fit( list(data_set_struct[:ex_i]) + list(data_set_struct[ex_i+1:]),
                        list(target_list[:ex_i]) + list(target_list[ex_i+1:])
                      )
        expected_data = data_set[ex_i]

    predicted_name = classifier.predict( np.array(c_list) )[0].split()[0]
    print "predicted_name,", predicted_name

    predicted_index = target_list.index(predicted_name)
    #print predicted_index

    predicted_data = list(data_set[predicted_index])
    #print predicted_data

    index = -1
    for i in range(len(data_set_struct)):
        if data_set_struct[i] == c_list:
            index = i

    # Find input list in shuffled list
    return [predicted_data, expected_data]

