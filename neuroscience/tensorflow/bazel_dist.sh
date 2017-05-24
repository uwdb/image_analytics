#!/bin/bash

# Starts bazel servers on all nodes

# user on aws; usually 'ubuntu'
username="ubuntu"
path_to_key="/path/to/key.pem"

# declare array with node ips (aws)
# put your ips including ports here!
i_line=0
filename="$1"
while read line
do
    name="$line"
    node_ips[$i_line]=$name
    echo "Name read from file - $name"
    i_line=$(($i_line+1))
done < "$filename"

# get length of an array
n_ips=${#node_ips[@]}

host=2500
# create cluster_spec argument
cluster_spec='worker|'
for (( i=0; i<${n_ips}; i++ ));
do
    cluster_spec=$cluster_spec${node_ips[$i]}":"$host
    host=$((host+1))
    if (( i < ${n_ips}-1 ));
    then
        cluster_spec=$cluster_spec"; "
    fi
done

echo $cluster_spec

# start sessions on all nodes
for (( i=1; i<${n_ips}; i++ ));
do
echo ${node_ips[$i]}
ssh -i ${path_to_key} -o StrictHostKeyChecking=no ${username}@${node_ips[$i]} screen -d -m pkill -u ubuntu
ssh -i ${path_to_key} -o StrictHostKeyChecking=no ${username}@${node_ips[$i]} screen -d -m "/home/ubuntu/tensorflow/bazel-bin/tensorflow/core/distributed_runtime/rpc/grpc_tensorflow_server --cluster_spec='${cluster_spec}' --job_name=worker --task_id=$i &"
done
