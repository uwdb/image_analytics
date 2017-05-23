#scp -r /home/ubuntu/scidb_mri/test_stream.py 172.31.11.134:~/.

#replace the array name with the name from ingestion
iquery -aq "load_library('stream')"
iquery -aq "stream(mean_b0_24, '/home/ubuntu/scidb_mri/stream_mask.py')"
