#Retrieve all 25 subjects 

import time
print "Start at", time.ctime()

cmd_aws = 'aws s3 sync s3://imagedb-data/'
#local_path = '/home/ubuntu/download/subjects/'
local_path = '/run/shm/subjects/'

cnt = 0
from os import system
with open('subjects.file') as f:
    for line in f:
        cnt += 1
        subject_id = line.strip()
        print 'Downloading subject %d (id = %s)' % (cnt, subject_id)
        system(cmd_aws + subject_id + '  ' + local_path + subject_id)

print "Done at", time.ctime()


