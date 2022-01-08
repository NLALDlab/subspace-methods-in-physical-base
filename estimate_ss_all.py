import numpy as np
import scipy as sc
from scipy.linalg import schur
import itertools
import math
from cvxopt import base, blas, lapack
from cvxopt.base import matrix, spmatrix

def estimate_ss(Obs, y, u, Tsc, stabilize_A=True, C_f=[], A_vera=[], A_vera_est=[], metodo_def_M=1, matrix_type=1,\
                  verbose=False, permutazioni_su_tutte_le_variabili = True, delete_permutations = False):
    
    # Estimate the state space model
    #INPUT:
    #      - Obs : extended observability matrix
    #      - y : output
    #      - u : input
    #      - metodo_def_M  # 0=A_u^s , 1=C_compl_ort

    #OUTPUT:
    #      - A, B, C, D : state-space matrices
    #      - x0 : initial state x(0)
    #      - n : model order
    Obs = np.asmatrix(Obs)

    prendo_parte_reale = True
    
    _, n = np.shape(Obs)
    nx = n #nx = C.shape[1]
    p = y.shape[0]
    N = y.shape[1]
    m = u.shape[0]
    
    use_A_vera = True if len(A_vera)>0 else False 

    if use_A_vera: autovalori_Avera,tmpV = np.linalg.eig(A_vera)

    #Estimate A and C
    impongo_C_fisica = np.size(C_f) > 0
    C = np.asarray( Obs[:p,:] ) 
    C_s_prima_stima = C.copy()
    
    Als = Obs[:-p,:]
    Bls = Obs[p:,:]
    #print("Als.shape = ",np.asarray(Als).shape," , Bls.shape = ",np.asarray(Bls).shape)
    Bls, _, _, _ = sc.linalg.lstsq(Als, Bls)
    Bls = np.asmatrix(Bls)

    A = Bls[:n,:]
    
    Bls2 = Obs[p:,:]
    Aerr = np.linalg.norm(Als@A - Bls2)
    A = np.asarray(A)
 
    tmpDA,tmpVA = np.linalg.eig(A)  
    #tmpU,tmpS,tmpV = np.linalg.svd(A,full_matrices=True); tmpV=tmpV.T; 
    #if verbose: print("valori singolari di As prima stima = ",tmpS)

    eigs_A_vera,eigvecs_A_vera = np.linalg.eig(A_vera)
    I_eigs_A_vera = np.argsort(eigs_A_vera)
    eigs_A_vera = eigs_A_vera[I_eigs_A_vera]
    eigvecs_A_vera = eigvecs_A_vera[:,I_eigs_A_vera]
    if verbose:
        if use_A_vera:
            print("A_s prima stima = ",A)
            err_comp_prima_stima = np.abs(A_vera - A)/np.abs(A_vera)
            print("relative error: np.abs(A_vera - As_prima_stima)/np.abs(A_vera) = ",err_comp_prima_stima)
            print("C_s prima stima = ",C)
            print("As prima stima eigenvalues: ",tmpDA)
        else:
            print("As prima stima eigenvalues: ",tmpDA)
            print("A_s prima stima = ",A)
        #endif
    #endif

    if True and stabilize_A:
        #Stabilize A, i.e. we transform A in a similar matrix, but with the absolute value of the eigenvalues less than 1

        Sc = np.asmatrix(np.zeros((n,n)))
        def F(w):
            return (abs(w) < 1.0)

        Sc[:,:] = A.copy()
        Sc, Vs, ns = schur(Sc,sort=F)
        w = np.diag(Sc)
        w = w.copy()
        if ns < n:
            maxiterstab = 100
            tmpniter = 0
            while ns < n and tmpniter < maxiterstab:
                w[ns:] = w[ns:]**(-1)
                Sc[::n+1] = w
                Sc = Vs @ Sc @ Vs.conj().T
                A[:,:] = np.real(Sc)
                Sc[:,:] = A
                Sc, Vs, ns = schur(Sc,sort=F)
                if 1 and verbose: print("ns = ",ns," , n = ",n)
                w = np.diag(Sc)
                w = w.copy()
                tmpniter += 1
            #endwhile
        else:
            if verbose: print("A stabilization is not needed!")
        #endif
    #endif         
    
    A_s_prima_stima = A.copy()
    if use_A_vera: relerr_Asps11 = np.abs(A_vera[0,0] - A_s_prima_stima[0,0])/np.abs(A_vera[0,0])
    
    # impongo la matrice "C" fisica, "C_f":
    if impongo_C_fisica:
        C_sps = C.copy()
        if 0 or verbose: print("C_sps = ",C_sps)
        autovalori_Asps,Tx_eigv = np.linalg.eig(A_s_prima_stima)
        I_autovalori_Asps = np.argsort(autovalori_Asps)
        ordered_autovalori_Asps = autovalori_Asps[I_autovalori_Asps]
        ordered_Tx_eigv = Tx_eigv[:,I_autovalori_Asps]
        if verbose: print("np.abs((autovalori_Asps - eigs_A_vera)/eigs_A_vera) = ",np.abs((ordered_autovalori_Asps - eigs_A_vera)/eigs_A_vera))
        maxrelerr_eigenvalues_A = np.max(np.abs((ordered_autovalori_Asps[0:p] - eigs_A_vera[0:p])/eigs_A_vera[0:p]))

        
        if matrix_type == 2:
            U_s, Tx_Q_s = schur(A_s_prima_stima) # "The Schur decomposition is: A = Z T Z^H"
            if verbose: print("U_s = ",U_s)
            if 0: Tx_Q_s = Tx_eigv.copy()

            A_r = np.linalg.inv(Tx_Q_s)  @ A @ Tx_Q_s

            C_r = C_sps @ Tx_Q_s     
        else:
            A_r = np.linalg.inv(Tx_eigv) @ A @ Tx_eigv

            C_r = C_sps @ Tx_eigv        
        #endif
        
        if 0 and nx == 3:
            permutations = np.array([[0,1,2],[1,2,0],[2,0,1],[2,1,0],[1,0,2],[0,2,1]])
        else:
            if permutazioni_su_tutte_le_variabili:
                permutations = np.asarray(list(itertools.permutations(np.arange(nx))))
            else:
                permutations = np.asarray(list(itertools.permutations(np.arange(nx),C.shape[0])))
            #endif
        #endif
        if verbose: print("total number of permutations = ",permutations.shape[0])
        
        # N.B.: permutations reduction can be applied iff metodo_def_M == 2
        if 1 and metodo_def_M == 2 and delete_permutations == True:
            
            if permutazioni_su_tutte_le_variabili:
                permutations1 = np.flip(np.asarray(list(itertools.permutations(np.arange(nx)))))
            else:
                permutations1 = np.flip(np.asarray(list(itertools.permutations(np.arange(nx),C.shape[0]))))
            #endif

            dim_C = C.shape[0]
            index = np.arange((math.factorial(nx)/(math.factorial(nx-dim_C)*math.factorial(dim_C)))-1)
            index_delete = np.arange(1,math.factorial(dim_C))
            index2 = np.arange(1,math.factorial(dim_C))
            for ii in index:
                index2 = index2 + math.factorial(dim_C)
                index_delete = np.append(index_delete, index2 )
            #endfor
            permutations = np.delete(permutations1,index_delete, axis=0)
            
        #endif
        nperm = permutations.shape[0]; print("number of effective permutations = ",nperm)
        
        # if complex-conjugated eigenvalues:
        Icc = []
        if matrix_type == 2:
            i = 0
            while i < nx-1:
                if np.abs(U_s[i+1,i]) > 0:
                    Icc.append(i)
                    Icc.append(i+1)
                    i = i+2
                else:
                    i = i+1
                #endif
            #endwhile
        else:
            for i in np.arange(nx):
                if np.imag(autovalori_Asps[i]) > 0.0:
                    Icc.append(i)
                #endif
            #endfor
        #endif
        if verbose: print("Icc = ",Icc)
        
        min_perm_index = -1; min_fro = np.Inf; fro_hist = np.zeros(nperm) 
        min_A = []; min_C = []; min_B = []; min_D = []; A_no_perm=[]
        min_relerr_A11 = np.Inf; min_relerr_est_A11 = np.Inf; min_autov_spost = -1 #np.Inf
        min_perm_diag_index = -1; min_frodiag = np.Inf; frodiag_hist = np.zeros(nperm)
        min_sigma_uo_index = -1; min_sigma_uo = np.Inf; sigma_uo_hist = np.zeros(nperm)
        err_comp_list = []; Tx_cond_list = []; S_Tx_list = []
        usato_autovettore_relativo_a_modo_non_misurato = []
        tmpMu_hist = []; tmpMm_hist = []; tmp_2err_Ac_est_sub_hist = []
        for jr in range(nperm):
            Jp = permutations[jr,:]; #if True: print("Jp ridotto = ",Jp)
            ncinJp = 0
            for i in np.arange(len(Jp)):
                if Jp[i] in Icc:
                    ncinJp += 1
                #endif
            #endfor
            if ncinJp % 2 > 0: continue 
            if matrix_type == 2:
                for i in np.arange(len(Jp)):
                    if ((Jp[i] in Icc) and ((Jp[i]-1 not in Jp) and (Jp[i]+1 not in Jp))):
                        pass #continue
                    #endif
                #endfor
            else:
                ppp = autovalori_Asps[Jp]
                Iccvals = autovalori_Asps[Icc]
                for i in np.arange(len(Jp)):
                    if (ppp[i] in Iccvals and np.conjugate(ppp[i]) not in Iccvals) or (np.conjugate(ppp[i]) in Iccvals and ppp[i] not in Iccvals):
                        continue
                    #endif
                #endfor
            #endif
            if not permutazioni_su_tutte_le_variabili:
                ny,nx = C.shape
                tmpJp = range(nx)
                tmpperm = list(Jp) + list(-1+np.zeros(nx-ny)); #print(tmpperm)
                jj = ny
                for j in range(nx):
                    if tmpJp[j] not in tmpperm:
                        tmpperm[jj] = tmpJp[j]
                        jj += 1
                    #nedif
                #endfor
                Jp = tmpperm.copy()
            #endif
            if 1 or verbose: print("###\n### permutation n.",str(jr),": ",Jp,"\n###")
            if False:
                Tx_perm = Tx_eigv[:,Jp]
                A = A_r[Jp,:]; A = A[:,Jp]
                C = C_r[:,Jp]
            else:
                M_r = np.zeros((nx,nx))
                for ir in range(nx):
                    M_r[ir,Jp[ir]] = 1.0
                #endfor
                #print("M_r = ",M_r)
                #print("M_r.T = ",M_r.T)
                #print("M_r.T @ M_r = ",M_r.T @ M_r)
                if matrix_type == 2:
                    A = M_r @ A_r @ M_r.T

                    Tx_perm = Tx_Q_s @ M_r.T

                    C = C_r @ M_r.T #C_r[:,Jp]

                    if 1:
                        tmpU = M_r @ np.linalg.inv(U_s) @ M_r.T
                        U1 = tmpU[0:p,0:p]
                        U3 = tmpU[p:,0:p]
                        tmpMu = C[0:p,p:] @ U3 @ np.linalg.inv(C[0:p,0:p])
                        #print("tmpMu = ",tmpMu)
                        tmpMu_hist.append(np.linalg.norm(np.diag(tmpMu),np.inf))
                        tmpMm = C[0:p,0:p] @ U1 @ np.linalg.inv(C[0:p,0:p])
                        tmpMm_hist.append(np.linalg.norm(np.diag(tmpMm),np.inf))
                        tmpr = np.max(np.max(np.abs(tmpMu / (tmpMu + tmpMm))))
                        #print("tmpr = ",tmpr)
                    #endif
                else:
                    A = M_r @ A_r @ M_r.T

                    Tx_perm = Tx_eigv @ M_r.T
                    
                    C = C_r @ M_r.T #C_r[:,Jp]
                #endif
            #endif
            if prendo_parte_reale:
                A_s_perm = np.real(Tx_perm @ A @ np.linalg.inv(Tx_perm))
            else:
                A_s_perm = Tx_perm @ A @ np.linalg.inv(Tx_perm)
            #endif
            if prendo_parte_reale:
                C_s_perm = np.real(C @ np.linalg.inv(Tx_perm))
            else:
                C_s_perm = C @ np.linalg.inv(Tx_perm)
            #endif
            if prendo_parte_reale:
                A = np.real(A)
                C = np.real(C)
            #endif
            if verbose: print("C_f = ",C_f)
            
            niter = 1 
            if True:
                iter = 0 
                tmpUCsps,tmpSCsps,tmpVCsps = np.linalg.svd(C_sps,full_matrices=True); tmpVCsps=tmpVCsps.T; 
                tmpUC,tmpSC,tmpVC = np.linalg.svd(C,full_matrices=True); tmpVC=tmpVC.T; 
                prev_tmpVC = tmpVC.copy()
                I = np.eye(nx)
                M = np.eye(nx) #, dtype=np.cfloat)
                G = np.zeros((nx,nx))
                jNSC = 0
                for j in range(nx):
                    jNSC = j
                    if metodo_def_M == 0: # dummy
                        M[j,:] = tmp[0,0:nx]
                    elif metodo_def_M == 1:
                        M[j,:] = tmpVC[:,jNSC].T
                    elif metodo_def_M == 2:
                        M[j,:] = I[jNSC,:]
                    elif metodo_def_M == 3:
                        M[j,:] = tmpVC[:,jNSC].T
                    elif metodo_def_M == 4:
                        if matrix_type == 2:
                            M[j,:] = tmpVCsps[:,jNSC].T @ Tx_Q_s
                        else:
                            M[j,:] = tmpVCsps[:,jNSC].T @ Tx_eigv
                        #endif
                    elif metodo_def_M == 5:
                        if matrix_type == 2:
                            M[j,:] = tmpVCsps[:,jNSC].T @ Tx_Q_s
                        else:
                            M[j,:] = tmpVCsps[:,jNSC].T @ Tx_eigv
                        #endif
                    elif metodo_def_M == 6: # NO: Tx viene singolare!
                        M[j,:] = I[jNSC,:]
                    #endif
                    if j < p:
                        M[j,:] = C[j,:]
                        G[j,:] = C_f[j,:]
                    else:
                        cmul = 1.0
                        if j==2: cmul = 1.0
                        if metodo_def_M == 3:
                            G[j,:] = cmul * I[jNSC,:]
                        elif metodo_def_M == 5:
                            G[j,:] = I[jNSC,:]  # NO, VIENE MALE COME "4"! @ Tx_eigv
                        elif metodo_def_M == 6:
                            pass # queste righe di G rimangono nulle
                        else:
                            G[j,:] = cmul * M[j,:]
                        #endif
                    #endif
                #endfor            
                Tx = np.linalg.solve(M, G)

                A = np.linalg.inv(Tx) @ A @ Tx
                C = C @ Tx
                
                autovalori_Asps,_ = np.linalg.eig(A_s_prima_stima)
                autovalori_A,_ = np.linalg.eig(A)
                
                if jr==0 or len(A_no_perm)==0:
                    A_no_perm = A.copy()
                #endif
                
                #
                # Estimate B, D and x0 
                F1 = matrix(0.0,(p*N,n))
                F1[:p,:] = C
                for ii in range(1,N):
                    F1[ii*p:(ii+1)*p,:] = F1[(ii-1)*p:ii*p,:] @ A
                #endfor
                if 0: print("len(np.where(np.isnan(F1)==True)) = ",len(np.where(np.isnan(F1)==True)[0]))

                F2 = matrix(0.0,(p*N,p*m))
                ut = u.T
                for ii in range(p):
                    F2[ii::p,ii::p] = ut
                #endfor
                if 0: print("len(np.where(np.isnan(F2)==True)) = ",len(np.where(np.isnan(F2)==True)[0]))

                F3 = matrix(0.0,(p*N,n*m))
                F3t = matrix(0.0,(p*(N-1),n*m))
                for ii in range(1,N):
                    for jj in range(p):
                        for kk in range(n):
                            F3t[jj:jj+(N-ii)*p:p,kk::n] = ut[:N-ii,:]*F1[(ii-1)*p+jj,kk]
                        #endfor
                    #endfor
                    F3[ii*p:,:] = F3[ii*p:,:] + F3t[:(N-ii)*p,:]
                #endfor
                if 0: print("len(np.where(np.isnan(F3)==True)) = ",len(np.where(np.isnan(F3)==True)[0]))
                if 0: print("len(np.where(np.isnan(F3t)==True)) = ",len(np.where(np.isnan(F3t)==True)[0]))

                F = matrix([[F1],[F2],[F3]])
                len_F_isnan = len(np.where(np.isnan(F)==True)[0])
                if False: print("len(np.where(np.isnan(F)==True)) = ",len(np.where(np.isnan(F)==True)[0]))
                if False: print("cond(F) = ",np.linalg.cond(F)," , F.shape = ",F.shape)
                if len_F_isnan > 0: 
                    print("attenzione: all'iterazione ",jr," F ha componenti NaN !")
                    continue
                #endif

                y = matrix(y)
                y_col = y[:]

                Sls = matrix(0.0,(F.size[1],1))
                Uls = matrix(0.0,(F.size[0],F.size[1]))
                Vtls = matrix(0.0,(F.size[1],F.size[1]))

                lapack.gesvd(F, Sls, jobu='S', jobvt='S', U=Uls, Vt=Vtls)

                Frank=len([ii for ii in range(Sls.size[0]) if Sls[ii] >= 1E-6])

                xx = matrix(0.0,(F.size[1],1))
                xx[:Frank] = Uls.T[:Frank,:] * y_col
                xx[:Frank] = base.mul(xx[:Frank],Sls[:Frank]**-1)
                xx[:] = Vtls.T[:,:Frank] * xx[:Frank]

                blas.gemv(F, xx, y_col, beta=-1.0)
                xerr = blas.nrm2(y_col)


                x0 = xx[:n]

                D = xx[n:n+p*m]
                D.size = (p,m)
                D = np.asarray(D)

                B = xx[n+p*m:]
                B.size = (n,m)
                B = np.asarray(B)
                
                #
                if True:
                    autovalori_Acf,eigV_Acf = np.linalg.eig(A)
                    autovalori_A_vera,eigV_A_vera = np.linalg.eig(A_vera)
                    tmp_relerr_A11 = np.abs(A_vera[0,0] - A[0,0])/np.abs(A_vera[0,0])
                    tmp_relerr_Asubmeas = np.linalg.norm(np.abs(A_vera[0:p,0:p] - A[0:p,0:p])/np.abs(A_vera[0:p,0:p]),np.inf)
                    tmp_relerr_est_A11 = np.abs(A_vera_est[0,0] - A[0,0])/np.abs(A_vera_est[0,0])
                    tmp_2err_eigV = np.linalg.norm(eigV_Acf[:,0:p] - eigV_A_vera[:,0:p])
                    Ac_vera = (1/Tsc) * (np.identity(nx) - np.linalg.inv(A_vera)) 
                    Ac_vera_est = (1/Tsc) * (np.identity(nx) - np.linalg.inv(A_vera_est)) 
                    tmp_Ac_est = (1/Tsc) * (np.identity(nx) - np.linalg.inv(A))
                    #teo_est = (1/Tsc)*np.diag((np.eye(p)-(tmpMm+tmpMu)))
                    teo_est = (1/Tsc)*np.diag((np.eye(p)-(tmpMm)))
                    relerr_unmeas = np.max(np.abs((teo_est - np.diag(tmp_Ac_est[0:p,0:p]))/np.diag(tmp_Ac_est[0:p,0:p])))
                    print("max(abs(diag(tmpMu)/diag(A_c[0:p,0:p]))) = ",relerr_unmeas)
                    if 0:
                        #tmp_2err_Ac_vera_sub = np.linalg.norm(Ac_vera[0:p,0:p] - tmp_Ac_est[0:p,0:p],2)
                        tmp_2err_Ac_vera_sub = np.linalg.norm(Ac_vera[0:p,0:p] - tmp_Ac_est[0:p,0:p],'fro')
                        tmp_2err_Ac_est_sub = np.linalg.norm(Ac_vera_est[0:p,0:p] - tmp_Ac_est[0:p,0:p],'fro')
                    else:
                        #tmp_2err_Ac_vera_sub = np.linalg.norm(Ac_vera[0:p,0:p] - tmp_Ac_est[0:p,0:p],np.inf)
                        #tmp_2err_Ac_est_sub = np.linalg.norm(Ac_vera_est[0:p,0:p] - tmp_Ac_est[0:p,0:p],np.inf)
                        #tmp_2err_Ac_est_sub = np.max(np.max(np.abs(Ac_vera_est[0:p,0:p] - tmp_Ac_est[0:p,0:p])))
                        #tmp_2err_Ac_est_sub = np.max(np.diag(np.abs(Ac_vera_est[0:p,0:p] - tmp_Ac_est[0:p,0:p])))
                        tmp_2err_Ac_est_sub = np.max(np.diag(np.abs((Ac_vera_est[0:p,0:p] - tmp_Ac_est[0:p,0:p])/Ac_vera_est[0:p,0:p])))
                        if verbose: print("|Ac_vera_est[0:p,0:p] - tmp_Ac_est[0:p,0:p]| = ",np.abs(Ac_vera_est[0:p,0:p] - tmp_Ac_est[0:p,0:p]))
                        tmp_relerr_A_sub = np.max(np.abs(np.diag(A_vera[0:p,0:p]) - np.diag(A[0:p,0:p]))/np.abs(np.diag(A_vera[0:p,0:p])))
                        #pessima: tmp_relerr_A_sub = np.max(np.abs(np.diag(A_vera) - np.diag(A)) / np.abs(np.diag(A_vera)))
                    #endif
                    tmp_2err_Ac_est_sub_hist.append(tmp_2err_Ac_est_sub)
                    if tmp_2err_Ac_est_sub < min_relerr_est_A11: 
                        min_relerr_est_A11 = tmp_2err_Ac_est_sub  # memorizzo l'errore su A_vera! 
                        relerr_unmeas_at_min = relerr_unmeas
                        if 1:
                            tmpMu_at_min = np.linalg.norm(np.diag(tmpMu),np.inf)
                            tmpMm_at_min = np.linalg.norm(np.diag(tmpMm),np.inf)
                        else:
                            tmpMu_at_min = np.linalg.norm(tmpMu,np.inf)
                            tmpMm_at_min = np.linalg.norm(tmpMm,np.inf)
                        #endif
                        min_perm_index = jr
                        min_A = A.copy()
                        min_C = C.copy()
                        min_B = B.copy()
                        min_D = D.copy()
                        max_Tx_col_norms = np.zeros(1)
                        # le norme delle righe di Tx non danno un indicatore significativo: max_Tx_col_norms = np.max(Tx_row_norms[p:]) / np.max(Tx_row_norms[0:p])
                        # le norme delle colonne di Tx non danno un indicatore significativo: max_Tx_col_norms = np.max(Tx_col_norms[p:]) / np.max(Tx_col_norms[0:p])                    
                    #endif
                    if tmp_relerr_A11 < min_relerr_A11: 
                        min_relerr_A11 = tmp_relerr_A11
                        usato_autovettore_relativo_a_modo_non_misurato = np.argmin(autovalori_Asps[Jp]-autovalori_Avera[-1])==nx-1
                    #endif
                #endif
                if iter < niter-1:  
                    tmpD,tmpV = np.linalg.eig(A)
                    if False:
                        Tx = tmpV.copy()
                    else:
                        tmpQ,tmpR = np.linalg.qr(tmpV)
                        Tx = np.linalg.solve(np.linalg.inv(tmpV),tmpQ.T) # tmpV^{-1} Tx = tmpQ.T -> Tx = tmpV tmpQ.T
                    #endif
                    #  tmpQ tmpV^{-1} A tmpV tmpQ.T
                    A = np.linalg.inv(Tx) @ A @ Tx
                    C = C @ Tx
        
                    if verbose:
                        print("As riortogonalizzata = ",A)
                        err_comp_riortogonalizzata = np.abs(A_vera - A)/np.abs(A_vera)
                        print("ed il suo errore relativo nella stima dei coefficienti = ",err_comp_riortogonalizzata)
                        print("Cs riortogonalizzata = ",C)
                        tmpD,tmpV = np.linalg.eig(A)
                        print("autovalori As dopo riortogonalizzazione: ",tmpD)
                        print("ortogonalità autovettori di As dopo riortogonalizzazione: ",np.linalg.norm(np.eye(nx)-tmpV.T@tmpV))
                        tmpU,tmpS,tmpV = np.linalg.svd(A,full_matrices=True); tmpV=tmpV.T; 
                        if verbose: print("valori singolari di As dopo riortogonalizzazione = ",tmpS)
                    #endif
                #endif
                if iter > 0:
                    if jr == min_perm_index:
                        min_A = A.copy()
                        min_C = C.copy()
                        min_B = B.copy()
                        min_D = D.copy()                    
                    #endif                
                #endif
            ##endfor
        #endfor
        if verbose: print("chosen min_perm_index = ",min_perm_index)
        if verbose: print("       min error = ",min_relerr_est_A11)
        A = min_A.copy()
        C = min_C.copy()
        B = min_B.copy()
        D = min_D.copy()

        relerr_vero_Af_ps_00 = np.abs((A_vera[0,0] - A_s_prima_stima[0,0])/A_vera[0,0])
        relerr_vero_Af_perm_00 = np.abs((A_vera[0,0] - A[0,0])/A_vera[0,0])
        maxrelerr_vero_Af_ps_sub = np.max(np.diag(np.abs((A_vera[0:p,0:p] - A_s_prima_stima[0:p,0:p])/A_vera[0:p,0:p])))
        maxrelerr_vero_Af_perm_sub = np.max(np.diag(np.abs((A_vera[0:p,0:p] - A[0:p,0:p])/A_vera[0:p,0:p])))
        maxrelerr_vero_Af_ps_alldiag = np.max(np.diag(np.abs((A_vera - A_s_prima_stima)/A_vera)))
        maxscalerr_vero_Af_ps_all = np.max(np.max(np.abs(np.diag(1./np.diag(A_vera)) @ (A_vera - A_s_prima_stima))))
        maxscalerr_vero_Af_perm_all = np.max(np.max(np.abs(np.diag(1./np.diag(A_vera)) @ (A_vera - A))))
        
        Ac_vera = (1/Tsc) * (np.identity(nx) - np.linalg.inv(A_vera)) 
        Ac_prima_stima = (1/Tsc) * (np.identity(nx) - np.linalg.inv(A_s_prima_stima))
        Ac_no_perm = (1/Tsc) * (np.identity(nx) - np.linalg.inv(A_no_perm))
        Ac_dopo_perm = (1/Tsc) * (np.identity(nx) - np.linalg.inv(A))
        Ac_vera_est = (1/Tsc) * (np.identity(nx) - np.linalg.inv(A_vera_est)) 
        
        maxrelerr_vero_Ac_ps_sub = np.max(np.diag(np.abs((Ac_vera[0:p,0:p] - Ac_prima_stima[0:p,0:p])/Ac_vera[0:p,0:p])))
        maxrelerr_vero_Ac_no_perm = np.max(np.diag(np.abs((Ac_vera[0:p,0:p] - Ac_no_perm[0:p,0:p])/Ac_vera[0:p,0:p])))
        maxrelerr_vero_Ac_perm_sub = np.max(np.diag(np.abs((Ac_vera[0:p,0:p] - Ac_dopo_perm[0:p,0:p])/Ac_vera[0:p,0:p])))
        minrelerr_vero_Ac_ps_sub = np.min(np.diag(np.abs((Ac_vera[0:p,0:p] - Ac_prima_stima[0:p,0:p])/Ac_vera[0:p,0:p])))
        minrelerr_vero_Ac_no_perm = np.min(np.diag(np.abs((Ac_vera[0:p,0:p] - Ac_no_perm[0:p,0:p])/Ac_vera[0:p,0:p])))
        minrelerr_vero_Ac_perm_sub = np.min(np.diag(np.abs((Ac_vera[0:p,0:p] - Ac_dopo_perm[0:p,0:p])/Ac_vera[0:p,0:p])))


        if verbose :
            print('Ac_vera = ', Ac_vera)
            print('Ac_no_perm = ', Ac_no_perm)
            print('Ac_dopo_perm = ', Ac_dopo_perm)
            print('Ac_prima_stima = ', Ac_prima_stima)
            print('Ac_vera_est = ', Ac_vera_est)
        #endif
        
        errore_Ac11_prima_stima = np.abs(Ac_vera[0,0] - Ac_prima_stima[0,0])#/np.abs(Ac_vera[0,0])
        errore_Ac11_dopo_perm = np.abs(Ac_vera[0,0] - Ac_dopo_perm[0,0])#/np.abs(Ac_vera[0,0])
        errore_Ac12_prima_stima = np.abs(Ac_vera[0,1] - Ac_prima_stima[0,1])#/np.abs(Ac_vera[0,1])
        errore_Ac12_dopo_perm = np.abs(Ac_vera[0,1] - Ac_dopo_perm[0,1])#/np.abs(Ac_vera[0,1])
        errore_Ac21_prima_stima = np.abs(Ac_vera[1,0] - Ac_prima_stima[1,0])#/np.abs(Ac_vera[1,0])
        errore_Ac21_dopo_perm = np.abs(Ac_vera[1,0] - Ac_dopo_perm[1,0])#/np.abs(Ac_vera[1,0])
        errore_Ac22_prima_stima = np.abs(Ac_vera[1,1] - Ac_prima_stima[1,1])#/np.abs(Ac_vera[1,1])
        errore_Ac22_dopo_perm = np.abs(Ac_vera[1,1] - Ac_dopo_perm[1,1])#/np.abs(Ac_vera[1,1])
        
        if verbose: 
            print("min_sigma_uo_index = ",min_sigma_uo_index)
            print("min_perm_index = ",min_perm_index)
            print("min_perm_diag_index = ",min_perm_diag_index)
            for jr in range(0): # RIPRISTINARE ! nperm):
                print("sigma*fro, sigma, fro, frodiag, frodiag/fro = ",sigma_uo_hist[jr]*fro_hist[jr]," , ",sigma_uo_hist[jr]," , ",fro_hist[jr]," , ",frodiag_hist[jr]," , ",frodiag_hist[jr]/fro_hist[jr])
                print("S(Tx) = ",S_Tx_list[jr])
                print("il numero di condizionamento di Tx = ",Tx_cond_list[jr])
                print("ed il suo errore relativo nella stima dei coefficienti = ",err_comp_list[jr])
            #endfor
            print("cond(Tx_eigv) = ",np.linalg.cond(Tx_eigv))
        #endif
        tmpU,tmpS,tmpV = np.linalg.svd(Tx_eigv,full_matrices=True); tmpV=tmpV.T; 
        if verbose: print("valori singolari di Tx_eigv = ",tmpS)
        # Estimate Bs, Ds and x0s 
        As = A_s_prima_stima.copy()
        Cs = C_s_prima_stima.copy()
        F1 = matrix(0.0,(p*N,n))
        F1[:p,:] = Cs
        for ii in range(1,N):
            F1[ii*p:(ii+1)*p,:] = F1[(ii-1)*p:ii*p,:] @ As
        #endfor
        if 0: print("len(np.where(np.isnan(F1)==True)) = ",len(np.where(np.isnan(F1)==True)[0]))

        F2 = matrix(0.0,(p*N,p*m))
        ut = u.T
        for ii in range(p):
            F2[ii::p,ii::p] = ut
        #endfor
        if 0: print("len(np.where(np.isnan(F2)==True)) = ",len(np.where(np.isnan(F2)==True)[0]))

        F3 = matrix(0.0,(p*N,n*m))
        F3t = matrix(0.0,(p*(N-1),n*m))
        for ii in range(1,N):
            for jj in range(p):
                for kk in range(n):
                    F3t[jj:jj+(N-ii)*p:p,kk::n] = ut[:N-ii,:]*F1[(ii-1)*p+jj,kk]
                #endfor
            #endfor
            F3[ii*p:,:] = F3[ii*p:,:] + F3t[:(N-ii)*p,:]
        #endfor
        if 0: print("len(np.where(np.isnan(F3)==True)) = ",len(np.where(np.isnan(F3)==True)[0]))
        if 0: print("len(np.where(np.isnan(F3t)==True)) = ",len(np.where(np.isnan(F3t)==True)[0]))

        F = matrix([[F1],[F2],[F3]])
        len_F_isnan = len(np.where(np.isnan(F)==True)[0])
        if False: print("len(np.where(np.isnan(F)==True)) = ",len(np.where(np.isnan(F)==True)[0]))
        if False: print("cond(F) = ",np.linalg.cond(F)," , F.shape = ",F.shape)
        if len_F_isnan > 0: 
            print("attenzione: Fs ha componenti NaN !")
        #endif

        y = matrix(y)
        y_col = y[:]

        Sls = matrix(0.0,(F.size[1],1))
        Uls = matrix(0.0,(F.size[0],F.size[1]))
        Vtls = matrix(0.0,(F.size[1],F.size[1]))

        lapack.gesvd(F, Sls, jobu='S', jobvt='S', U=Uls, Vt=Vtls)

        Frank=len([ii for ii in range(Sls.size[0]) if Sls[ii] >= 1E-6])

        xx = matrix(0.0,(F.size[1],1))
        xx[:Frank] = Uls.T[:Frank,:] * y_col
        xx[:Frank] = base.mul(xx[:Frank],Sls[:Frank]**-1)
        xx[:] = Vtls.T[:,:Frank] * xx[:Frank]

        blas.gemv(F, xx, y_col, beta=-1.0)
        xerr = blas.nrm2(y_col)


        x0s = xx[:n]

        Ds = xx[n:n+p*m]
        Ds.size = (p,m)
        Ds = np.asarray(Ds)

        Bs = xx[n+p*m:]
        Bs.size = (n,m)
        Bs = np.asarray(Bs)

    #endif
    relerr_Anp11 = None
    check_ortog_Tx = None
    cond_M = None
    cond_Tx = None
    relerr_A11_min_autov_spost = None

    return {'A': A, 'B': B, 'C': C, 'D': D, 'x0': x0, 'As': As, 'Bs': Bs, 'Cs': Cs, 'Ds': Ds, 'x0s': x0s, 'n': n, 'Aerr': Aerr, 'xerr': xerr, 'min_relerr_A11':min_relerr_A11, 'min_relerr_est_A11':min_relerr_est_A11,\
'check_ortog_Tx':check_ortog_Tx,'cond_M':cond_M,'cond_Tx':cond_Tx,\
'maxrelerr_vero_Ac_ps_sub':maxrelerr_vero_Ac_ps_sub,'maxrelerr_vero_Ac_perm_sub':maxrelerr_vero_Ac_perm_sub,\
'minrelerr_vero_Ac_ps_sub':minrelerr_vero_Ac_ps_sub,'minrelerr_vero_Ac_perm_sub':minrelerr_vero_Ac_perm_sub,'relerr_unmeas':relerr_unmeas,'tmpMu_hist':tmpMu_hist,'tmpMm_hist':tmpMm_hist,\
'tmp_2err_Ac_est_sub_hist':tmp_2err_Ac_est_sub_hist,'tmpMu_at_min':tmpMu_at_min,'tmpMm_at_min':tmpMm_at_min,\
'minrelerr_vero_Ac_no_perm':minrelerr_vero_Ac_no_perm, 'maxrelerr_vero_Ac_no_perm':maxrelerr_vero_Ac_no_perm, 'relerr_Asps11':relerr_Asps11,'max_Tx_col_norms':max_Tx_col_norms,\
'usato_autovett_non_mis':usato_autovettore_relativo_a_modo_non_misurato,'maxrelerr_eigenvalues_A':maxrelerr_eigenvalues_A,\
            'relerr_A11_min_autov_spost':relerr_A11_min_autov_spost, \
'errore_Ac11_prima_stima':errore_Ac11_prima_stima,'errore_Ac12_prima_stima':errore_Ac12_prima_stima,\
'errore_Ac21_prima_stima':errore_Ac21_prima_stima,'errore_Ac22_prima_stima':errore_Ac22_prima_stima,\
'errore_Ac11_dopo_perm':errore_Ac11_dopo_perm,'errore_Ac12_dopo_perm':errore_Ac12_dopo_perm,\
'errore_Ac21_dopo_perm':errore_Ac21_dopo_perm,'errore_Ac22_dopo_perm':errore_Ac22_dopo_perm,\
'relerr_vero_Af_ps_00':relerr_vero_Af_ps_00,'relerr_vero_Af_perm_00':relerr_vero_Af_perm_00,\
'maxrelerr_vero_Af_ps_sub':maxrelerr_vero_Af_ps_sub,'maxrelerr_vero_Af_perm_sub':maxrelerr_vero_Af_perm_sub,\
'maxrelerr_vero_Af_ps_alldiag':maxrelerr_vero_Af_ps_alldiag,\
'maxscalerr_vero_Af_ps_all':maxscalerr_vero_Af_ps_all,'maxscalerr_vero_Af_perm_all':maxscalerr_vero_Af_perm_all}
