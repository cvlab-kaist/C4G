import numpy as np

def compute_umeyama(source_points, target_points):
    """
    Umeyama 알고리즘을 사용하여 Source 점들을 Target 점들에 맞추는 
    Sim3 변환 (s, R, t)을 계산합니다.
    
    Args:
        source_points: (N, 3) numpy array
        target_points: (N, 3) numpy array
    
    Returns:
        s: scale factor (float)
        R: rotation matrix (3, 3)
        t: translation vector (3,)
    """
    assert source_points.shape == target_points.shape, "Point sets must have same shape"
    
    n = source_points.shape[0]

    # 1. 중심점(Centroid) 계산 및 제거
    mu_s = np.mean(source_points, axis=0)
    mu_t = np.mean(target_points, axis=0)
    
    # 중심을 원점으로 이동 (Centered points)
    p_s = source_points - mu_s
    p_t = target_points - mu_t

    # 2. 공분산 행렬 (Covariance Matrix) 계산
    sigma_s = np.mean(np.sum(p_s**2, axis=1))  # Variance of source
    # Covariance matrix H
    H = p_s.T @ p_t / n 

    # 3. SVD (Singular Value Decomposition) 수행
    U, D, Vt = np.linalg.svd(H)
    
    # 4. 회전 행렬 (Rotation) R 계산
    d = np.linalg.det(Vt.T @ U.T)
    S = np.eye(3)
    if d < 0: S[2, 2] = -1  # Reflection 방지
    
    R = (Vt.T @ S @ U.T)

    # 5. 스케일 (Scale) s 계산
    # target variance / source variance 비율 고려
    # Trace(D * S) / sigma_s
    trace_val = np.sum(D * np.diag(S)) # np.trace(np.diag(D) @ S)와 동일

    if sigma_s < 1e-12:
        # Degenerate case: all source points are nearly identical
        s = 1.0
    else:
        s = trace_val / sigma_s
    
    # 6. 이동 (Translation) t 계산
    t = mu_t - s * (R @ mu_s)

    return s, R, t