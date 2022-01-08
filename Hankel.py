import numpy as np

def matrix(x, m, n, p = 1, q = 1):  
  
    """
    Return the block Hankel matrix H, a matrix of size (m*p, n*q),
    
            [ h_1   h_2     ...  h_n       ]
        H = [ h_2   h_3     ...  h_{n+1}   ]
            [  :     :            :        ]
            [ h_m   h_{m+1} ...  h_{m+n-1} ],
    
    where h_i, i=1...(m+n-1), are the Hankel blocks of size (p,q).
    
    INPUT
    x        matrix of size (p*q*(m+n-1),1)  
    
    m       number of block rows
    
    n       number of block columns
    
    p       number of rows of a Hankel block
    
    q       number of columns of a Hankel block    
    """
    
    
    # Check the dimensions 
    if x.shape[0] * x.shape[1] != p*q*(m+n-1):
        print(" ValueError: x.shape[0] * x.shape[1] must equal to p*q*(m+n-1)")
    
    # Reshape the dimensions of the matrix
    x = x.copy().reshape(p, q*(m+n-1))
    
    # Create the Hankel matrix by rows
    H = np.asmatrix(np.zeros((m*p, n*q)))
    
    for ii in range(m):
        H[ii*p :  (ii+1)*p, :] = x[:, ii*q : (ii+n)*q]
    
    return H  
  