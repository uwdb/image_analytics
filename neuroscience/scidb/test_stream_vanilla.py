#!/usr/bin/python

#
#DFZ 11/15/2016: it's hard to control the chunk size read from the 
# stream() interface, see run_mri_stream.output for a concrete idea.
#

import sys
from __builtin__ import float
end_of_interaction = 0

while (end_of_interaction != 1):
  header = sys.stdin.readline().rstrip()
  if(header != "0"):
    #We receive a message from the SciDB instance:
    num_lines = int(header)  #how many lines did we get?
    
    #Collect all lines into a list:
    input_lines = []
#    list_data = []
    for i in range(0, num_lines):
      line = sys.stdin.readline().rstrip()
      input_lines.append(line)

      #construct the values into a numpy array for MRI
#      list_data.append(float(line))
#      data = np.reshape(np.asarray(input_data, dtype=np.float32), (145, 174, 145, 9))
    
    print(2)
    print("Total lines: " + str(num_lines))
    print("----> First line: " + input_lines[0])
    #Print a response: 
#     print(num_lines+1)
#     for i in range(0, num_lines):
#        print("I got\t" + input_lines[i])
#     print("THX!")
    sys.stdout.flush()

    #This will appear in the scidb-sterr.log file:
    sys.stderr.write("I got a chunk with "+ str(num_lines) + " lines of text!\n")
  else:
    #If we receive "0", it means the SciDB instance has no more
    #Data to give us. Here we have the option of also responding with "0"
    #Or sending some other message (i.e. a global sum):
    end_of_interaction = 1
    print("1")
    print("KTHXBYE")
    sys.stdout.flush()
    sys.stderr.write("I got the end-of-data message. Exiting.\n")
ok = 0
# So I cannot 'return' or 'print' even after 'return'; the following statements would cause errors
exit(0)
# print "Start at " + str(time.ctime())
