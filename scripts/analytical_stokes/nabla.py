from sympy import *

class Nabla:
    #
    # nabla dimension is the covariant dimension
    # nabla applies to something of contravariant dimension
    #
    def __init__(self,X):
        self.X = X
        self.dim_cov = self.X.shape[0]

    def apply_to(self,u):
        dim_contra = u.shape[0]
        grad_u = zeros(dim_contra,self.dim_cov)

        for i in range(0,dim_contra):
            for j in range(0,self.dim_cov):
                grad_u[i,j] = diff(u[i,0],self.X[j,0])

        grad_u = simplify(grad_u)

        return grad_u
    
    def curl(self,u):
        dim_contra = u.shape[0]

        curl_u = zeros(dim_contra,1)

        if dim_contra == 3 & self.dim_cov == 3 :
            curl_u[0] = diff(u[2,0],self.X[1,0]) - diff(u[1,0],self.X[2,0])
            curl_u[1] = diff(u[0,0],self.X[2,0]) - diff(u[2,0],self.X[0,0])
            curl_u[2] = diff(u[1,0],self.X[0,0]) - diff(u[0,0],self.X[1,0])

        else:
            assert False,'Nabla.curl not implemented for this dimensions'

        curl_u = simplify(curl_u)

        return curl_u
        
    def cdot(self,u):
        dim_contra = u.shape[0]
        div_u = zeros(1,1)
                        
        for i in range(0,dim_contra):
            div_u[0,0] += diff(u[i,0],self.X[i,0])
        
        return div_u
        

    def square(self,u):
        dim_contra = u.shape[0]
        
        square_u = zeros(dim_contra,1)
        
        x = self.X[0,0]
                    
        for i in range(0,dim_contra):
            component = 0*x
            for j in range(0,self.dim_cov):
                component += diff(u[i,0],self.X[j,0],2)
            square_u[i] = component

        return simplify(square_u)