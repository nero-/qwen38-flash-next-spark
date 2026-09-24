import datetime, os
import torch
import torch.distributed as dist
rank = int(os.environ['RANK'])
dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=90))
for n in [1, 1024, 1048576]:
    x = torch.full((n,), rank + 1.0, device='cuda')
    dist.all_reduce(x)
    torch.cuda.synchronize()
    assert torch.all(x == 3).item(), (rank, n)
    print(f'PASS rank={rank} all_reduce elements={n}', flush=True)
dist.destroy_process_group()
