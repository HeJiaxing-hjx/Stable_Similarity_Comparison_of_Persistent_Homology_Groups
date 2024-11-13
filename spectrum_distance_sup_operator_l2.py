import numpy as np
from numpy.linalg import norm
def compute_distance(list1,list2):
    m = len(list1)
    n = len(list2)
    k = max(m, n)
    A = np.zeros((m,m))
    B = np.zeros((n,n))
    spectral_A = np.zeros(k)
    spectral_B = np.zeros(k)
    for i in range(len(list1)):
        for j in range(len(list1)):
            a1 = list1[i][0]
            #print(a1)
            b1 = list1[i][1]
            a2 = list1[j][0]
            b2 = list1[j][1]
            left = max(a1,a2)
            right = min(b1,b2)
            
            if right > left:
                A[i][j] = (right-left)
            else:
                A[i][j] = 0


    for i in range(len(list2)):
        for j in range(len(list2)):
            a1 = list2[i][0]
            #print(a1)
            b1 = list2[i][1]
            a2 = list2[j][0]
            b2 = list2[j][1]
            left = max(a1, a2)
            right = min(b1, b2)
            
            if right > left:
                B[i][j] = (right-left)
            else:
                B[i][j] = 0
    
    
    eigenvalues_A, eigenvectors_A = np.linalg.eigh(A)
    
    sorted_indices_A = np.argsort(-eigenvalues_A)
    sorted_A = eigenvalues_A[sorted_indices_A]
    sorted_A = sorted_A/sorted_A[0]

    spectral_A[0:m]=sorted_A
    
    eigenvalues_B, eigenvectors_B = np.linalg.eigh(B)
    sorted_indices_B = np.argsort(-eigenvalues_B)
    sorted_B = eigenvalues_B[sorted_indices_B]

    sorted_B = sorted_B / sorted_B[0]
    spectral_B[0:n]=sorted_B
    
    dist = 0
    
    """l2 distance"""
    for w in range(k):
        
        dist += np.abs(spectral_A[w] - spectral_B[w])*np.abs(spectral_A[w] - spectral_B[w])
        
    return np.sqrt(dist)

