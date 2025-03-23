source openr1/bin/activate
cd /workspace/rlpvr-open-r1/
export C3S_KEYTAB=/workspace/rlpvr-open-r1/c3s.search-gpt.keytab 
export C3S_ACCOUNT=search-gpt 
kinit -kt ${C3S_KEYTAB} ${C3S_ACCOUNT}@C3.NAVER.COM 
klist -kte ${C3S_KEYTAB} 
export HDFS_CONNECTOR_PATH=/root/c3s-hdfs-connector-0.7/bin/hdfs-connector
# download 
mkdir datasets
$HDFS_CONNECTOR_PATH -get -f hdfs://jmt/user/search-gpt/models/multimodal/datasets/ ./
mkdir models
$HDFS_CONNECTOR_PATH -get -f hdfs://jmt/user/search-gpt/models/multimodal/Qwen2.5-Math-7B ./models/