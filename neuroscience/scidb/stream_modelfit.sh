#scp -r /home/ubuntu/scidb_mri/test_stream.py 172.31.11.134:~/.

#replace the array name with the name from ingestion
iquery -anq "load_library('stream')"
iquery -aq "stream(denoised_parsed, '/home/ubuntu/scidb_mri/stream_modelfit.py')"
