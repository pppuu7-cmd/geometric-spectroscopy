import numpy as np
import csv
import os

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

def compute_svd(J):
    s = np.linalg.svd(J, compute_uv=False)
    return s

def tikhonov_inverse(J, y, lam):
    JTJ = J.T @ J
    d = JTJ.shape[0]
    return np.linalg.solve(JTJ + lam*np.eye(d), J.T @ y)

def main():

    from geometric_spectroscopy.potentials import build_potential_rstar
    from geometric_spectroscopy.qnm_wkb import qnm_wkb6

    model="hayward"
    alpha=0.0
    L_max=12
    theta0=np.zeros(L_max+1)

    ells=[2,3,4]
    ns=[0,1,2]

    def data_vector(theta):
        out=[]
        for ell in ells:
            rstar,V,_=build_potential_rstar(
                model=model,ell=ell,alpha=alpha,
                theta=theta,L_max=L_max,
                N_r=6000,r_min=2.2,r_max=60.0)
            for n in ns:
                res=qnm_wkb6(rstar,V,ell=ell,n=n,pade=True)
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

    s = compute_svd(J)
    sigma_max = s[0]

    lambdas=np.logspace(-12,0,25)
    noise_sigma=1e-4
    trials=50

    rows=[]

    for lam in lambdas:

        errors=[]
        gains=[]

        for _ in range(trials):
            noise=np.random.normal(scale=noise_sigma,size=len(y0))
            theta_rec=tikhonov_inverse(J,y0+noise,lam)
            errors.append(np.linalg.norm(theta_rec))

        # effective noise gain
        filter_factors = s/(s**2 + lam)
        gain = np.max(filter_factors)

        rows.append({
            "lambda":lam,
            "mean_error":float(np.mean(errors)),
            "std_error":float(np.std(errors)),
            "noise_gain":float(gain)
        })

    ensure_dir("artifacts/order_stability")
    write_csv("artifacts/order_stability/tikhonov_scan.csv",rows)

if __name__=="__main__":
    main()