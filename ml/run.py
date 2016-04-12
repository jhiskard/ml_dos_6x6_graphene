from predict import *
import matplotlib.pyplot as plt
import numpy as np

#input_list = '11,12,13,14,15,16,17,51,52,53,54,65,66,68'
#input_list = '0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17'
#input_list = '0,1,2,3,4,5,6,7,8'
input_list = '0,1'
atom_category = 'B'
broad = 0.05
erange = 10

predicted_data, expected_list = predict(atom_category, input_list, broad, erange)
x_list = np.linspace(-erange,erange,1001)

#print predicted_data
#print len(predicted_data)

fig = plt.figure()
fig1 = fig.add_subplot(111)
fig1.plot(x_list, predicted_data, 'k-', label='prediction')
fig1.plot(x_list, expected_list, 'r-', label='expectation')
fig1.legend()
fig.savefig('result.png')

