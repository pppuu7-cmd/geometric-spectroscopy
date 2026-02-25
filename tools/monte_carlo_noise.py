import csv
import os
import numpy as np

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def write_csv(path, rows):
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

def import_mod(path):
    __import__(path)
    import sys
    return sys.modules[path]

def main():

    pot = import_mod("geometric_spectroscopy.potentials")
    qnm = import_mod("geometric_spectroscopy.qnm_wkb")

    model="hayward"
    alpha=0.0
    L_max=6
    theta0=np.zeros(L_max+1)

    ells=[2,3,4]
    ns=[0,1,2]

    def data_vector(theta):
        out=[]
        for ell in ells:
            rstar,V,_=pot.build_potential_rstar(
                model=model,ell=ell,alpha=alpha,
                theta=theta,L_max=L_max,
                N_r=6000,r_min=2.2,r_max=60.0)
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

    JTJ=np.linalg.inv(J.T@J)@J.T

    noise_levels=[1e-6,1e-5,1e-4,1e-3]
    trials=100

    rows=[]

    for sigma in noise_levels:
        errors=[]
        for _ in range(trials):
            noise=np.random.normal(scale=sigma,size=len(y0))
            theta_rec=JTJ@(y0+noise)
            errors.append(np.linalg.norm(theta_rec))
        rows.append({
            "noise":sigma,
            "mean_error":float(np.mean(errors)),
            "std_error":float(np.std(errors))
        })

    ensure_dir("artifacts/order_stability")
    write_csv("artifacts/order_stability/monte_carlo_noise.csv",rows)

if __name__=="__main__":
    main()