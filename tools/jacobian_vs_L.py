import csv
import os
import numpy as np

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def write_csv(path, rows):
    if not rows:
        print("No rows.")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] wrote {path}")

def import_mod(path):
    __import__(path)
    import sys
    return sys.modules[path]

def compute_svd(J):
    s = np.linalg.svd(J, compute_uv=False)
    smax = float(s[0])
    smin = float(s[-1])
    cond = float(smax/smin) if smin>0 else float("inf")
    rank = int(np.sum(s > 1e-12*smax))
    return rank, smax, smin, cond

def main():

    pot = import_mod("geometric_spectroscopy.potentials")
    qnm = import_mod("geometric_spectroscopy.qnm_wkb")

    model = "hayward"
    alpha = 0.0

    L_values = [6,12,24]
    ells = [2,3,4,5]
    ns = [0,1,2]

    rows = []

    for L_max in L_values:

        theta0 = np.zeros(L_max+1)

        def data_vector(theta):
            out=[]
            for ell in ells:
                rstar,V,_=pot.build_potential_rstar(
                    model=model,
                    ell=ell,
                    alpha=alpha,
                    theta=theta,
                    L_max=L_max,
                    N_r=6000,
                    r_min=2.2,
                    r_max=60.0,
                )
                for n in ns:
                    res=qnm.qnm_wkb6(rstar,V,ell=ell,n=n,pade=True)
                    out.append(complex(res.omega))
            out=np.array(out)
            return np.concatenate([out.real,out.imag])

        delta=1e-3
        y0=data_vector(theta0)
        J=np.zeros((len(y0),len(theta0)))

        for j in range(len(theta0)):
            th=theta0.copy()
            th[j]+=delta
            y1=data_vector(th)
            J[:,j]=(y1-y0)/delta

        rank,smax,smin,cond=compute_svd(J)

        rows.append({
            "L_max":L_max,
            "rank":rank,
            "sigma_max":smax,
            "sigma_min":smin,
            "cond":cond
        })

    ensure_dir("artifacts/order_stability")
    write_csv("artifacts/order_stability/jacobian_vs_L.csv",rows)

if __name__=="__main__":
    main()