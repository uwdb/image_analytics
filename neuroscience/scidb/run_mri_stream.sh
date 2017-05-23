#scp -r /home/ubuntu/scidb_mri/test_stream.py 172.31.11.134:~/.

#replace the array name with the name from ingestion
iquery -aq "stream(original_half_1st, '/home/dongfang/scidb_mri/test_stream_denoise.py')"
iquery -aq "stream(original_half_2nd, '/home/dongfang/scidb_mri/test_stream_denoise.py')"
