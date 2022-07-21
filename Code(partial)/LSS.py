import numpy as np
import Levenshtein



load_profile1 = open('sequence1.txt', "r")
read_it1 = load_profile1.read()
load_profile2 = open('sequence2.txt', "r")
read_it2 = load_profile2.read()

sequence1set=[]
sequence2set=[]

for line1 in read_it1.splitlines():
    sequence1set.append(line1)
for line2 in read_it2.splitlines():
    sequence2set.append(line2)

temp=np.zeros((489,489))

for i in range(len(sequence1set)):
    for j in range(len(sequence2set)):
        temp[i][j]=Levenshtein.ratio(sequence1set[i],sequence2set[j])
        
np.savetxt('lncRNA_sequence_similarity.txt', temp, fmt="%f", delimiter=" ")

