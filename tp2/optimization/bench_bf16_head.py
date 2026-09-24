"""Isolated full-vocabulary TP2 BF16 head probe; never modifies model weights."""
import json, statistics, time
import torch
import triton
import triton.language as tl

@triton.jit
def head_gemv(X,W,Y,N:tl.constexpr,K:tl.constexpr,M:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    n=tl.program_id(0)*BN+tl.arange(0,BN)
    k=tl.arange(0,BK)
    w=tl.load(W+n[:,None]*K+k[None,:],(n[:,None]<N)&(k[None,:]<K),0).to(tl.float32)
    for m in tl.static_range(M):
        x=tl.load(X+m*K+k,k<K,0).to(tl.float32)
        y=tl.sum(w*x[None,:],1)
        tl.store(Y+m*N+n,y,n<N)

def timing(fn):
    for _ in range(3):fn()
    torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(8):fn()
    samples=[]
    for _ in range(5):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record();g.replay();b.record();b.synchronize();samples.append(a.elapsed_time(b)/8)
    return statistics.median(samples)

def main():
    torch.manual_seed(42)
    N,K=124160,2560
    w=torch.randn((N,K),device='cuda',dtype=torch.bfloat16)*0.02
    records=[]
    for m in (1,4,32):
        x=torch.randn((m,K),device='cuda',dtype=torch.bfloat16)
        y=torch.empty((m,N),device='cuda',dtype=torch.bfloat16)
        baseline=lambda:torch.mm(x,w.T,out=y)
        ms=timing(baseline);baseline();reference=y.clone()
        rows=[dict(kernel='torch.mm',ms=ms)]
        if m<=4:
            for bn in (4,8):
                fn=lambda:head_gemv[(triton.cdiv(N,bn),)](x,w,y,N,K,m,bn,triton.next_power_of_2(K),num_warps=4,enable_fp_fusion=False)
                elapsed=timing(fn);fn();torch.cuda.synchronize()
                # Compare both outputs to a float64 oracle on 128 fixed vocabulary rows.
                ids=torch.linspace(0,N-1,128,device='cuda').long()
                truth=x.double()@w[ids].double().T
                err=(y.float()-reference.float()).abs()
                rows.append(dict(kernel=f'triton-gemv-n{bn}',ms=elapsed,speedup=ms/elapsed,
                    max_abs_difference=float(err.max()),rms_difference=float(err.square().mean().sqrt()),
                    candidate_oracle_rms=float((y[:,ids].double()-truth).square().mean().sqrt()),
                    baseline_oracle_rms=float((reference[:,ids].double()-truth).square().mean().sqrt()),
                    argmax_agreement=float((y.argmax(-1)==reference.argmax(-1)).float().mean())))
        records.append(dict(m=m,n=N,k=K,weight_mib=w.numel()*w.element_size()/2**20,results=rows))
    print(json.dumps(dict(timestamp=time.time(),gpu=torch.cuda.get_device_name(),torch=torch.__version__,records=records),indent=2))
if __name__=='__main__':main()
