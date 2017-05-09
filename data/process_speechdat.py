import glob
import sys
import os
import re

# Get globals

# define paths
place = os.getcwd()
meta_path = place + "/meta/"
pattern_path = place + "/patterns/"
data_path = place + "/raw/"

# Clear directories
os.system("rm patterns/*")
os.system("rm meta/*")

# Make pattern, matches any of the data files and extracts recording name and nationality
pattern = "([^\s\.]*)\.(\w\w)[aA]"
prog = re.compile(pattern)

#String for sox command
cmd = "sox -t raw -c 1 -e a-law -r 8000 {inFile} -e signed-integer -b 16 {out}.wav"

k = 0
n = 10
for path, dirs, files in os.walk(data_path):
    if len(files) > 0 and k < n:
        for i in files:
                res = prog.match(i)
                if res:
                    file_path = path+"/"+i
                    out_path = pattern_path + str(k)
                    os.system(cmd.format(inFile = file_path, out = out_path))
                    k += 1
    elif k == n:
        break
                

