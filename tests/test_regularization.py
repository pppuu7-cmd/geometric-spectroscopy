def test_tikhonov_stabilizes():
    import numpy as np
    J = np.diag([1.0, 1e-6])
    y = np.array([1.0,1.0])
    lam = 1e-3
    theta_reg = np.linalg.solve(J.T@J + lam*np.eye(2), J.T@y)
    theta_unreg = np.linalg.pinv(J)@y
    assert np.linalg.norm(theta_reg) < np.linalg.norm(theta_unreg)