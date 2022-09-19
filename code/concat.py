import numpy as np
import pandas as pd
import csv
import random


csv_file = 'diseaseFeature.csv'
txt_file = 'diseaseFeature.txt'
with open(txt_file, "w") as my_output_file:
    with open(csv_file, "r") as my_input_file:
        [my_output_file.write(" ".join(row) + '\n') for row in csv.reader(my_input_file)]
    my_output_file.close()

csv_file = 'lncRNAFeature.csv'
txt_file = 'lncRNAFeature.txt'
with open(txt_file, "w") as my_output_file:
    with open(csv_file, "r") as my_input_file:
        [my_output_file.write(" ".join(row) + '\n') for row in csv.reader(my_input_file)]
    my_output_file.close()

csv_file = '../datasets/LD_adjmat.csv'
txt_file = 'LD_adjmat.txt'
with open(txt_file, "w") as my_output_file:
    with open(csv_file, "r") as my_input_file:
        [my_output_file.write(" ".join(row) + '\n') for row in csv.reader(my_input_file)]
    my_output_file.close()

DisLnc = np.loadtxt('LD_adjmat.txt', dtype=np.float64)
x1 = []
x2 = []
for i in range(DisLnc.shape[0]):
    for j in range(DisLnc.shape[1]):
        if DisLnc[i][j] == 1:
            x1.append(j)
            x2.append(i)

#obtain positive lncRNA-disease pairs
with open("diseaseFeature.txt") as xh:
    with open('lncRNAFeature.txt') as yh:
        with open("PositiveSampleFeature.txt", "w") as zh:
            xlines = xh.readlines()
            ylines = yh.readlines()
            for k in range(len(x1)):
                for i in range(len(xlines)):
                    for j in range(len(ylines)):
                        if i == x1[k] and j==x2[k]:
                            line = xlines[i].strip() + ' ' + ylines[j]
                            zh.write(line)

DisLnc = np.loadtxt('LD_adjmat.txt', dtype=np.float64)
x1 = []
x2 = []
for i in range(DisLnc.shape[0]):
    for j in range(DisLnc.shape[1]):
        if DisLnc[i][j] == 0:
            x1.append(j)
            x2.append(i)

#obtain negative lncRNA-disease pairs
with open("diseaseFeature.txt") as xh:
    with open('lncRNAFeature.txt') as yh:
        with open("NegativeSampleFeature.txt", "w") as zh:
            xlines = xh.readlines()
            ylines = yh.readlines()
            selected = random.sample(range(len(x1)), 1569)
            for k in range(len(selected)):
                for i in range(len(xlines)):
                    for j in range(len(ylines)):
                        if i == x1[selected[k]] and j==x2[selected[k]]:
                            line = xlines[i].strip() + ' ' + ylines[j]
                            zh.write(line)

PositiveSampleFeature = np.loadtxt('PositiveSampleFeature.txt', dtype=np.float64)
NegativeSampleFeature = np.loadtxt('NegativeSampleFeature.txt', dtype=np.float64)
SampleFeature = []
SampleFeature.extend(PositiveSampleFeature)
SampleFeature.extend(NegativeSampleFeature)
SampleFeature = np.array(SampleFeature)
saveSampleFeature=pd.DataFrame(SampleFeature)
saveSampleFeature.to_csv('SampleFeature.csv')



