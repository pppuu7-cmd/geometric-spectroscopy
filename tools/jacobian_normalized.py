import numpy as np
import csv
import os

def ensure_dir(path):
    os.makedirs(path,exist_ok=True)

def write_csv(path,rows):
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

def compute_stats(J):

    s=np.linalg.svd(J,compute_uv=False)
    smax=float(s[0])
    smin=float(s[-1])
    cond=float(smax/smin)
    return smax,smin,cond

def main():

    from geometric_spectroscopy.jacobian import build_jacobian,realify_complex_residuals

    model="hayward"
    alpha=0.0

    Jc=build_jacobian(model=model,alpha=alpha,L_max=6,order=6)
    Jr=realify_complex_residuals(Jc)

    smax,smin,cond=compute_stats(Jr)

    # normalization
    scale=np.std(Jr,axis=0)
    Jr_norm=Jr/scale

    smax_n,smin_n,cond_n=compute_stats(Jr_norm)

    rows=[{
        "raw_sigma_min":smin,
        "raw_cond":cond,
        "norm_sigma_min":smin_n,
        "norm_cond":cond_n
    }]

    ensure_dir("artifacts/order_stability")
    write_csv("artifacts/order_stability/jacobian_normalized.csv",rows)

if __name__=="__main__":
    main()