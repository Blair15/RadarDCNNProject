import csv
import gc 
import os

arr_txt = [x for x in os.listdir(os.path.dirname(os.path.realpath(__file__))) if x.endswith(".dat")]

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

    re_minus_im = []

    ## Rather than having 2, 128 element arrays, lets just use the
    ##  absolute value of the difference between the corresponding real
    ##   and imaginary parts
    for i in range(0, len(re)):
        re_minus_im.append(abs(re[i]-im[i]))

    ##Now we want (128) element vectors which represent the values collected
    ## at each chirp of the radar by processing the re minus im list which
    ##  should be of length 7,680,000
    
    ##chirps should be of length 60,000 once processed
    chirps = []

    ##There are 60k chirps for every 60second recording,
    ## given that there are 7,680,000 complex numbers collected and
    ##  60k chirps per second this means that 128 complex numbers per chirp
    ##   with both a real and imaginary component 
    for i in range(0, 60000):
        start_index = i*128
        end_index = ((i+1)*128)
        re_chirp = np.array(re_minus_im[start_index:end_index])
        chirp = np.array([re_chirp])
        chirps.append(chirp)

    print "List of chirps created of length " + str(len(chirps))
    print "*** Should be 60k ***"

    np.save(file+"ValuesComplexNp", np.array(chirps))

files = []

for action in arr_txt:
    processFile(action)
    files.append(action)

print "Files processed " + str(files)
