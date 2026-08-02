"""A worked non-physics instance: graph sparsification.

Instantiates the transfer-guarantee framework on a setting with no physics and no
surrogate model. `X` = weighted graphs on a fixed vertex set; `‖·‖_A` = relative
Frobenius error of the Laplacian (the aggregate quantity a cut-oriented sparsifier
controls); `Q` = algebraic connectivity (the Fiedler value), read for clustering,
mixing time, and robustness conclusions.

The perturbation family weakens the bridge edges between two clusters by a
fraction `w`, redistributing the removed weight inside one cluster so the total
edge weight is unchanged. This is what an aggressive sparsifier does when it
undersamples low-effective-resistance structure.

Result: the pair sits in row 2 of the trichotomy, and the framework distinguishes
two certificates that both look "spectral" -- a sampled Rayleigh-quotient check,
which leaves nearly a decade of silent risk, and a Fiedler-directed check, which
closes it.

Standalone: depends only on numpy.

Usage::  python scripts/run_graph_instance.py
"""

import numpy as np

np.random.seed(0)
n1=n2=12; n=n1+n2
W=np.zeros((n,n))
for i in range(n1):
    for j in range(i+1,n1):
        if np.random.rand()<0.5: W[i,j]=W[j,i]=1.0
for i in range(n1,n):
    for j in range(i+1,n):
        if np.random.rand()<0.5: W[i,j]=W[j,i]=1.0
bridges=[(0,n1),(1,n1+1)]
for (i,j) in bridges: W[i,j]=W[j,i]=1.0
def lap(W): return np.diag(W.sum(1))-W
L0=lap(W); ev0,V0=np.linalg.eigh(L0); Q0=ev0[1]; F0=np.linalg.norm(L0,'fro')
intra=[(i,j) for i in range(n1) for j in range(i+1,n1) if W[i,j]>0]
R=np.random.randn(n,32); R-=R.mean(0)          # 32 random test vectors, mean-zero
R/=np.linalg.norm(R,axis=0)

lam2 = Q0 if 'Q0' in dir() else np.linalg.eigvalsh(lap(W))[1]


def make(direction, w):
    """Two perturbation directions: bridge weakening, intra-cluster reshuffling."""
    W2 = W.copy()
    if direction == 'bridge':
        rem = 0.0
        for (i, j) in bridges:
            d = w * W[i, j]; W2[i, j] -= d; W2[j, i] -= d; rem += d
        per = rem / len(intra)
        for (i, j) in intra: W2[i, j] += per; W2[j, i] += per
    else:
        k = len(intra) // 2
        for (i, j) in intra[:k]: W2[i, j] += w; W2[j, i] += w
        for (i, j) in intra[k:2 * k]: W2[i, j] -= w; W2[j, i] -= w
    return W2


def perturb(w):
    W2=W.copy(); rem=0.0
    for (i,j) in bridges:
        d=w*W[i,j]; W2[i,j]-=d; W2[j,i]-=d; rem+=d
    per=rem/len(intra)
    for (i,j) in intra: W2[i,j]+=per; W2[j,i]+=per
    return W2

def metrics(w):
    L2=lap(perturb(w))
    agg=np.linalg.norm(L2-L0,'fro')/F0
    q=abs(np.linalg.eigvalsh(L2)[1]-Q0)/Q0
    num=np.einsum('ij,jk,ki->i',R.T,L2-L0,R); den=np.einsum('ij,jk,ki->i',R.T,L0,R)
    d_rand=np.max(np.abs(num/den))                      # sampled Rayleigh check
    f=V0[:,1]; d_fied=abs(f@ (L2-L0) @ f)/(f@L0@f)      # Fiedler-directed check
    return agg,q,d_rand,d_fied

print(f"Q = lambda_2 = {Q0:.5f}\n")
print(f"{'w':>6} {'aggregate':>10} {'|dQ|/Q':>8} {'d_random':>9} {'d_fiedler':>10}")
for w in (0.05,0.1,0.2,0.4,0.7,1.0):
    a,q,dr,df=metrics(w); print(f"{w:>6.2f} {a:>10.4f} {q:>8.4f} {dr:>9.4f} {df:>10.4f}")

def bis(f,t,lo=1e-6,hi=1.0):
    for _ in range(60):
        m=np.sqrt(lo*hi)
        if f(m)<t: lo=m
        else: hi=m
    return np.sqrt(lo*hi)

TAU=0.05
print(f"\ngate tolerance 0.05 on the detection statistic; tau = {TAU}")
print(f"{'eps':>7} {'omega':>7} {'omega_D rand':>13} {'omega_D fiedler':>16}")
for eps in (0.20,0.10,0.05,0.02,0.01):
    wa=bis(lambda x:metrics(x)[0],eps)
    wr=bis(lambda x:metrics(x)[2],0.05)
    wf=bis(lambda x:metrics(x)[3],0.05)
    f=lambda w: metrics(min(w,1.0)*0.999)[1]
    print(f"{eps:>7.3f} {f(wa):>7.4f} {f(min(wa,wr)):>13.4f} {f(min(wa,wf)):>16.4f}")
wr=bis(lambda x:metrics(x)[2],0.05); wf=bis(lambda x:metrics(x)[3],0.05)
wc=bis(lambda x:metrics(x)[1],TAU)
print(f"\nw_corrupt(tau=0.05) = {wc:.4f}   w_det(random) = {wr:.4f}   w_det(Fiedler) = {wf:.4f}")
print(f"|B| random  = {max(0,np.log10(wr/wc)):.3f} decades")
print(f"|B| Fiedler = {max(0,np.log10(wf/wc)):.3f} decades")


# ---------------------------------------------------------------------------
# Composition slack: a numerical witness that Proposition 3 is not tight.
#
# Stage 1  Q1 : graph -> Laplacian spectrum, measured in ||.||_B = worst
#               ABSOLUTE eigenvalue move.
# Stage 2  Q2 : spectrum -> relative lambda_2 error, Lipschitz with constant
#               1/lambda_2 in ||.||_B.
#
# The two stages have different worst directions -- intra-cluster reshuffling
# maximises the absolute spectral move, bridge weakening maximises the lambda_2
# error -- so the composite bound omega2 o omega1 overstates the truth.
# ---------------------------------------------------------------------------

def _slack_stats(direction, w):
    L2 = lap(make(direction, w)); ev2 = np.linalg.eigvalsh(L2)
    agg = np.linalg.norm(L2 - L0, 'fro') / F0
    spec = float(np.max(np.abs(ev2 - ev0)))
    q = abs(ev2[1] - lam2) / lam2
    return agg, spec, q


def composition_slack():
    print("\nComposition slack (Proposition 3 is a bound, not an identity)")
    print(f"  {'eps':>7} {'w1(eps)':>9} {'bound':>9} {'true':>9} {'slack':>8}")
    for eps in (0.05, 0.02, 0.01, 0.005):
        w1 = comp = 0.0
        for d in ('bridge', 'intra'):
            w = bis(lambda x: _slack_stats(d, x)[0], eps)
            _, sp, q = _slack_stats(d, w * 0.999)
            w1 = max(w1, sp); comp = max(comp, q)
        bound = w1 / lam2
        print(f"  {eps:>7.3f} {w1:>9.4f} {bound:>9.4f} {comp:>9.4f} {bound/comp:>7.2f}x")
    print("\n  worst directions differ, which is exactly the non-tightness condition:")
    for d in ('bridge', 'intra'):
        w = bis(lambda x: _slack_stats(d, x)[0], 0.02)
        _, sp, q = _slack_stats(d, w * 0.999)
        print(f"    {d:>7}: ||.||_B move = {sp:.4f}   relative lambda_2 error = {q:.4f}")


composition_slack()
