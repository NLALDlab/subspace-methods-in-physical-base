import numpy as np
from  LTI_systems import *

def get_parameters(esperimento=0, verbose=False):
    if esperimento == 0:  # condizioni nominali
        M = [27.0, 120.0, 1.1246e4]			# Kg
        if verbose: print("M = ",M)
        K = [18.0e6, 6.0e7, 1.2e8]			# N/m
        if verbose: print("K = ",K)
        C = [5.0e4, 4.6e4, 2.4e5]			# N*s/m
        if verbose: print("C = ",C)
        deltaX = [0.00499999882095, 0.00499999948562, 3.33333076e-02]		# m
        vector_deltacoeff_C3 = [1.0]
        vector_deltacoeff_K2 = [1.0]
        vector_deltacoeff_K1_C1 = [1.0]
        vector_deltacoeff_M1 = [1.0]
    elif esperimento == 1:  # condizioni anomale previste e valori nominali
        M = [27.0, 120.0, 1.1246e4]			# Kg
        if verbose: print("M = ",M)
        K = [18.0e6, 6.0e7, 1.2e8]			# N/m
        if verbose: print("K = ",K)
        C = [5.0e4, 4.6e4, 2.4e5]			# N*s/m
        if verbose: print("C = ",C)
        deltaX = [0.00499999882095, 0.00499999948562, 3.33333076e-02]		# m
        # NB: non e' previsto ceare un vettore di "deltacoeff" per deltaX, perche' la modifica non     
        #     verrebbe poi passata a "simula_sistema_meccanico_ngdl()"
        vector_deltacoeff_C3 = [1.0,1.0,1.0,1.0,1.1,1.2,1.1,1.2]
        vector_deltacoeff_K2 = [0.8,0.9,0.8,0.9,1.0,1.0,1.0,1.0]
        vector_deltacoeff_K1_C1 = [0.8,0.9,0.8,0.9,1.0,1.0,1.0,1.0]
        vector_deltacoeff_M1 = [1.0]
    elif esperimento == 2:  # insieme ridotto di condizioni anomale previste e valori nominali
        M = [27.0, 120.0, 1.1246e4]			# Kg
        if verbose: print("M = ",M)
        K = [18.0e6, 6.0e7, 1.2e8]			# N/m
        if verbose: print("K = ",K)
        C = [5.0e4, 4.6e4, 2.4e5]			# N*s/m
        if verbose: print("C = ",C)
        deltaX = [0.00499999882095, 0.00499999948562, 3.33333076e-02]		# m
        # NB: non e' previsto ceare un vettore di "deltacoeff" per deltaX, perche' la modifica non     
        #     verrebbe poi passata a "simula_sistema_meccanico_ngdl()"
        vector_deltacoeff_C3 = [1.0,1.0,1.1,1.2,1.1,1.2]
        vector_deltacoeff_K2 = [0.8,0.9,0.8,0.9,1.0,1.0]
        vector_deltacoeff_K1_C1 = [1.0]
        vector_deltacoeff_M1 = [1.0]
    elif esperimento == 3:  # condizioni anomale impreviste e valori nominali
        M = [27.0, 120.0, 1.1246e4]			# Kg
        if verbose: print("M = ",M)
        K = [18.0e6, 6.0e7, 1.2e8]			# N/m
        if verbose: print("K = ",K)
        C = [5.0e4, 4.6e4, 2.4e5]			# N*s/m
        if verbose: print("C = ",C)
        deltaX = [0.00499999882095, 0.00499999948562, 3.33333076e-02]		# m
        # NB: non e' previsto ceare un vettore di "deltacoeff" per deltaX, perche' la modifica non     
        #     verrebbe poi passata a "simula_sistema_meccanico_ngdl()"
        vector_deltacoeff_C3 = [1.0,1.5]
        vector_deltacoeff_K2 = [0.5,1.0,1.3]
        vector_deltacoeff_K1_C1 = [0.5,1.0]
        vector_deltacoeff_M1 = [1.0,1.8]
    else:
        print("ERRORE: esperimento non previsto!")
    #endif
    return M,K,C,deltaX,vector_deltacoeff_C3,vector_deltacoeff_K2,vector_deltacoeff_K1_C1,\
           vector_deltacoeff_M1


def IO_config(esperimento):
    if esperimento == 0:
        i_massa_attuata = [1]; # 1,2,3
        ii_variabili_misurate = range(6)
    elif esperimento == 1:
        i_massa_attuata = [1]; # 1,2,3
        ii_variabili_misurate = range(6)
    elif esperimento == 2:
        i_massa_attuata = [1]; # 1,2,3
        ii_variabili_misurate = range(6)
    elif esperimento == 3:
        i_massa_attuata = [1]; # 1,2,3
        ii_variabili_misurate = range(6)
    elif esperimento == 4:
        i_massa_attuata = [1]; # 1,2,3
        ii_variabili_misurate = range(2)
    else:
        print("ERRORE: esperimento non previsto!")
    #endif
    return ii_variabili_misurate,i_massa_attuata
  
def build(esperimento,parameters=None):
    if parameters == None:
        M,K,C,deltaX,_,_,_,_ = get_parameters(esperimento)
    else:
        M,K,C,deltaX = parameters['M'],parameters['K'],parameters['C'],parameters['deltaX']
    #endif
    variabili_misurate,config_attuatori = IO_config(esperimento)
    ndof = len(M)
    A = np.matrix(np.zeros((2*ndof,2*ndof)))
    for i in range(ndof):
        A[ndof+i,i] = 1.0
    #endfor
    for i in range(ndof-1):
        A[i,i] -= C[i]/M[i]
        A[i,i+1] = C[i]/M[i]
        A[i+1,i] = C[i]/M[i+1]
        A[i+1,i+1] -= C[i]/M[i+1]
        A[i,ndof+i] -= K[i]/M[i]
        A[i,ndof+i+1] = K[i]/M[i]
        A[i+1,ndof+i] = K[i]/M[i+1]
        A[i+1,ndof+i+1] -= K[i]/M[i+1]
    #endfor
    A[ndof-1,ndof-1] -= (0.0+C[ndof-1])/M[ndof-1]  
    A[ndof-1,ndof+ndof-1] -= (0.0+K[ndof-1])/M[ndof-1]  
    #print("A = ",A)
    B = np.matrix(np.zeros((2*ndof,2)))
    #B = np.matrix([[Bf[0],   K1*deltaX1/M1                ], \
    #               [Bf[1],   (K2*deltaX2-K1*deltaX1)/M2   ], \
    #               [Bf[2],   (K3*deltaX3-K2*deltaX2)/M3   ], \
    #               [0.,                 0.                ], \
    #               [0.,                 0.                ], \
    #               [0.,                 0.                ]])
    for i in range(ndof):
        if i==0:
            B[i,1] = (K[i]*deltaX[i])/M[i]
        else:
            B[i,1] = (K[i]*deltaX[i] - K[i-1]*deltaX[i-1])/M[i]
        #endif
    #endfor
    for i in config_attuatori:
        B[i-1,0] = 1./M[i-1]  
    #endfor   
    #print("B = ",B)
    D = np.zeros((2*ndof,2))
    C = np.eye(2*ndof) 
    C = np.asmatrix(C[variabili_misurate,:])
    D = np.asmatrix(D[variabili_misurate,:])
    return A, B, C, D

def simula(esperimento, Ac, Bc, Cc, Dc, load, Ts, theta=1.0):
    if parameters == None:
        _,_,_,deltaX,_,_,_,_ = get_parameters(esperimento)
    else:
        _,_,_,deltaX = parameters['M'],parameters['K'],parameters['C'],parameters['deltaX']
    #endif
    ndof = len(deltaX)
    N = load.shape[load.ndim-1]
    #x0 = np.matrix([0., 0., 0., deltaX1+deltaX2+deltaX3, deltaX2+deltaX3, deltaX3]).T
    x0 = np.asmatrix(np.zeros(2*ndof)).T
    for i in range(ndof):
        x0[ndof+i,0] = np.sum(deltaX[i:ndof])
    #endfor   
    #print("x0 = ",x0)
    u = np.zeros([2,N])
    u[0,:] = load[0,0:N]
    u[1,:] = np.ones(N)
    u = np.matrix(u)
    #
    y,X_hist,Ad,Adx = LTI.simula_CLTI_StateSpace(Ac,Bc,Cc,Dc,u,x0,Ts,theta,calc_diff_da_x0=False)
    response = y #- Cc*x0
    return response, X_hist, x0, Ad, Adx
