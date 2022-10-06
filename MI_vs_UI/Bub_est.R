#BUB estimator for discrete entropy
#Mathilde GAILLARD
#Ref. Liam PANINSKI Estimation of Entropy and Mutual Information

library(pracma)
library(corpcor)


bub_opti = function(N,m,k_max){
  #m = nber of bins
  #N = nber of data
  #k_max = nber of coefficients a_j for which bub_opti is using the BUB method otherwise -> compute MM estimator
  #k_max doit être inférieur ou egal à N
  
  p = (0:N)/N 
  B = mat_bernouilli_polynom(p,N)
  g = sapply(p,f,m)
  g_col = matrix(g, ncol = 1)
  h = c(0, sapply(p[-1],H))
  Y_tot = matrix(h, ncol = 1)
  
  #every a_j initialized with the MM estimator
  a = (0:N)/N
  a = -a*log(a)+((1-a)/(2*N))
  
  best_MM = Inf
  for (i in (1:k_max)){
    #print(i)
    h_exp = a[(i+1):length(a)]%*%(B[(i+1):length(B[,1]),])#part of the entropy already explained by the coeff
    Y_non_pond = Y_tot-t(h_exp) #withdraw the part explained
    
    G = repmat(g,i,1)
    X = t(G*B[1:i,])
    D = Diag(rep(-1,i),0)+Diag(rep(1,i-1),1)
    U = (t(X)%*%X)+(N/4)*(t(D)%*%D)
    U[i,i] = U[i,i]+(N/4)
    
    Y = g_col*Y_non_pond
    XY = (t(X)%*%Y)#-h_exp
    XY[i] = XY[i] + (N/4)*a[i+1]
    
    a[1:i] = pseudoinverse(U)%*%XY
    
    #make a 1 col matrix
    a_mat_1col = matrix(a, ncol = 1)
    
    #compare the performance of this specific set f a_j
    biais = 2*g*abs((t(Y_tot)-a%*%B))
    maxbiais = max(abs(biais))
    borne_var = max((a-c(a[2:length(a)],0))^2)
    MM = sqrt(maxbiais^2+N*borne_var)
    
    if(MM<best_MM){
      #choose the best set of a_j
      best_MM = MM
      best_a = a
      best_biais = maxbiais
    }
  }
  return(best_a)
}

mat_bernouilli_polynom = function(p,N){
  #compute the matrix of bernouilli polynomials
  fa = lgamma(1:(N+1))
  Ni = fa[N+1]-fa[1:(N+1)]-rev(fa)
  
  B = zeros(n =  N+1, m = N+1)
  p = p[-1]
  p = p[-length(p)]
  
  lp = log(p)
  lq = rev(lp)
  
  for (i in 0:N){
    B[i+1,2:N] = Ni[i+1]+i*lp+(N-i)*lq
  }
  B = exp (B)
  B[2:(N+1),1]=zeros(N,1)
  B[1:N,N+1]=zeros(N,1)
  return (B)
}

H = function(x){
  #entropy function
  nz = (x>0)
  return (-sum(x[nz]*log(x[nz])))
}

f = function(x,m){
  #weight function
  if(x<(1/m)){
    return(m)
  }
  else{
    return(1/x)
  }
}

estimateur_bub_entropie_discrete = function(X){
  #compute the discrete entropy with the bub_opti function
  N = length(X)
  h = tabulate (X+1)
  m = length(h)
  hist_eff = c()
  for (i in 1:(N+1)){
    hist_eff[i]=sum((h==i-1))
  }
  A = bub_opti(N,m,k_max = 11) #k_max = 11 works when N<10^6
  H_BUB = sum(A*hist_eff)
  return(H_BUB)
}