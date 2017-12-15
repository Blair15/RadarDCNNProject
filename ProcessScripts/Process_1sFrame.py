import csv
import gc 
import os

import numpy as np

##arr_txt = [x for x in os.listdir(os.path.dirname(os.path.realpath(__file__))) if x.endswith(".dat")]

arr_txt = ["Walk_D_Set2.dat"]

def processFile(file):
    data = np.genfromtxt(file, dtype=None, delimiter='', skip_header=4)

    re = []
    im = []

    ##Split the real and imaginary parts and store in seperate lists 
    for value in data:
        value = value[:-1]
        value = value.split("+")
        re.append(int(value[0]))
        im.append(int(value[1]))

    print "Real and imaginary lists created of length " + str(len(re)) + " " + str(len(im))
    print "*** Should be 7,680,000 ***"

    ##Now we want (128) element vectors which represent the values collected
    ## at each chirp of the radar by processing the re minus im list which
    ##  should be of length 7,680,000
    
    ##sweeps should be of length 60,000 once processed
    ##sweeps = []

    ##There are 60k sweeps for every 60second recording,
    ## given that there are 7,680,000 complex numbers collected and
    ##  60k sweeps per second this means that 128 complex numbers per sweep
    ##   with both a real and imaginary component 
    ##for i in range(0, 60000):
    ##    start_index = i*128
    ##    end_index = ((i+1)*128)
    ##    re_sweep = np.array(re[start_index:end_index])
    ##    im_sweep = np.array(im[start_index:end_index])
    ##    sweep = np.array([re_sweep, im_sweep])
    ##    sweeps.append(sweep)

    ##print "List of sweeps created of length " + str(len(sweeps))
    ##print "*** Should be 60k ***"

    ##Now want to take the 60k (128,2) vectors and split them into 20 (3000,128,2)
    ## where each of these vectors corresponds to a 3s timeframe of the original
    ##  recording

    frames = []

    ##split this 7,680,000 length list into 60 128,000 sections
    ## i.e 60 1s frames
    for i in range(0,60):
        start_index = i*128000
        end_index = (i+1)*128000
        re_frame = re[start_index:end_index]
        im_frame = im[start_index:end_index]
        frame = [re_frame, im_frame]
        frames.append(frame)

    frames = np.reshape(frames, (60,128000,2))

    np.save(file+"1sFrame", np.array(frames))

files = []

for action in arr_txt:
    processFile(action)
    files.append(action)

print "Files processed " + str(files)
