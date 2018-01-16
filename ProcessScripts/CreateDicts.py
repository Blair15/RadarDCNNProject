import os
import gc
import numpy as np
import cPickle as pickle

arr_txt = [x for x in os.listdir(os.path.dirname(os.path.realpath(__file__))) if x.endswith(".npy")]

##arr_txt = ["Walk_D_Set2.dat1sFrame.npy"]

## These are 6 Dictionaries to be pickled and reloaded into a NN
train_batch_1 = {}
train_batch_2 = {}
train_batch_3 = {}
train_batch_4 = {}
train_batch_5 = {}
test_batch = {}

files = []

i = 0
## Here we're distributing each frame from each numpy file to one of 6 batches
##  each "batch" is a python dict where key = action_label, value = list of frames
for filen in arr_txt:
    action_label = filen.split("_")[0]
    frames = np.load(filen)
    for frame in frames:
        if i%6 == 0:
            if action_label not in train_batch_1:
                train_batch_1[action_label] = [frame]
            else:
                train_batch_1[action_label].append(frame)
        if i%6 == 1:
            if action_label not in train_batch_2:
                train_batch_2[action_label] = [frame]
            else:
                train_batch_2[action_label].append(frame)     
        if i%6 == 2:
            if action_label not in train_batch_3:
                train_batch_3[action_label] = [frame]
            else:
                train_batch_3[action_label].append(frame)
        if i%6 == 3:
            if action_label not in train_batch_4:
                train_batch_4[action_label] = [frame]
            else:
                train_batch_4[action_label].append(frame) 
        if i%6 == 4:
            if action_label not in train_batch_5:
                train_batch_5[action_label] = [frame]
            else:
                train_batch_5[action_label].append(frame)
        if i%6 == 5:
            if action_label not in test_batch:
                test_batch[action_label] = [frame]
            else:
                test_batch[action_label].append(frame)
        i += 1
    print "*** Processed " + str(filen) + " ***"
    files.append(filen)

pickle.dump(train_batch_1, open("train_batch_1.p", "wb"))
pickle.dump(train_batch_2, open("train_batch_2.p", "wb"))
pickle.dump(train_batch_3, open("train_batch_3.p", "wb"))
pickle.dump(train_batch_4, open("train_batch_4.p", "wb"))
pickle.dump(train_batch_5, open("train_batch_5.p", "wb"))
pickle.dump(test_batch, open("test_batch.p", "wb"))


print files
