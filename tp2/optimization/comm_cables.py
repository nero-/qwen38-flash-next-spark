import argparse, json, os, subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--rank',type=int,required=True);p.add_argument('--cables', type=int, choices=(1,2), required=True);p.add_argument('--label', required=True);a=p.parse_args()
r=a.rank;root=Path.home()/'builds/qwen-tp2';image='eugr/spark-vllm-b12x@sha256:5249a162cd39aa090e803243aef376e1f87f0fd4826f9249ea1ecd0c814b2c7e'
(root/'optimization/comm-cache').mkdir(exist_ok=True)
env={'HOME':'/tmp','RANK':str(r),'LOCAL_RANK':'0','WORLD_SIZE':'2','MASTER_ADDR':'10.100.168.2','MASTER_PORT':'29659','NCCL_IB_HCA':['rocep1s0f1,roceP2p1s0f1','rocep1s0f0,roceP2p1s0f0'][r],'NCCL_SOCKET_IFNAME':['enp1s0f1np1','enp1s0f0np0'][r],'NCCL_IB_GID_INDEX':'3','NCCL_IB_TC':'106','NCCL_IB_MERGE_NICS':'1','B12X_ROCE_TRAFFIC_CLASS':'106','NCCL_NET_PLUGIN':'none','NCCL_DEBUG':'WARN','B12X_BENCH_GIT_REV':'8a99d639410e39d5f39cb4037675331beceea1d4','B12X_BENCH_GIT_STATUS':''}
if a.cables == 2:
    env['NCCL_IB_HCA']=['rocep1s0f1,roceP2p1s0f0','rocep1s0f0,roceP2p1s0f1'][r]
env['B12X_ROCE_HCA']=env['NCCL_IB_HCA']
cmd=['docker','run','--detach','--name','qwen-comm-check','--gpus','all','--network','host','--ipc','host','--device','/dev/infiniband','--ulimit','memlock=-1','--user','1000:1000','--mount',f'type=bind,src={root}/optimization,dst=/work','--mount',f'type=bind,src={root}/optimization/comm-cache,dst=/tmp','--entrypoint','/usr/bin/python3']
for k,v in env.items():cmd+=['--env',f'{k}={v}']
cmd += [image,'/work/benchmark_roce_oneshot.py','--sizes','8192,20480,30720,40960,65536,262144,1048576,2097152','--gather-rows','4,5,6,16,64','--gather-cols','124160','--warmups','10','--samples','40','--blocks','2','--graph-ops','10','--output',f'/work/{a.label}.json']
print(subprocess.check_output(cmd,text=True))
