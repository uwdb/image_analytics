#scp -r /home/ubuntu/scidb_mri/test_stream.py 172.31.11.134:~/.

iquery -aq "stream(build(<val:double> [i=0:9,4,0], i), '/home/ubuntu/scidb_mri/test_stream_vanilla.py')"
