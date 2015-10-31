#! /usr/bin/env python
# -*- coding: utf-8 -*-
from sympy import *

from nabla import Nabla

import io_tools


x,y,z = symbols('x y z')

init_printing(use_unicode=True)

X = Matrix([[x],[y],[z]])

print X

nabla = Nabla(X)


psi = zeros(3,1)

# nico sol
#psi[0,0] = x**2*(1-x)**2*(sin(2*pi*y))**2*(sin(2*pi*z))**2
#psi[2,0] = (sin(2*pi*x))**2*(sin(2*pi*y))**2*z**2*(1-z)**2

#nico-evans sol
#psi[0,0] = sin(2*pi*x)*y**2*(1-y)**2*z**2*(1-z)**2
#psi[2,0] = x**2*(1-x)**2*y**2*(1-y)**2*sin(2*pi*z)

#testing sol
psi[0,0] = x**2*(1-x)**2*y**2*(1-y)**2*(sin(pi*z))**2
psi[2,0] = x**2*(1-x)**2*y**2*(1-y)**2*z**2*(1-z)**2

#evans sol
#psi[0,0] = x*(1-x)*y**2*(1-y)**2*z**2*(1-z)**2
#psi[2,0] = x**2*(1-x)**2*y**2*(1-y)**2*z*(1-z)

print psi

u = nabla.curl(psi)

print u

div_u = diff(u[0],X[0])+diff(u[1],X[1])+diff(u[2],X[2])

print 'div_u = '
print simplify(div_u)

print 'boundary conditions:'
print u.subs(x,0)
print u.subs(x,1)
print u.subs(y,0)
print u.subs(y,1)
print u.subs(z,0)
print u.subs(z,1)

u_p = u.subs(x,.1)
u_p = u_p.subs(y,.3)
u_p = u_p.subs(z,.2)

print u_p.evalf()

p = zeros(1,1)

p[0] = sin(pi*x)*sin(pi*y)*sin(pi*z)-8/(pi**3)
print p
p_avg = integrate(p,(x,0,1),(y,0,1),(z,0,1))
print p_avg

p_force = nabla.apply_to(p)
p_force = p_force.transpose()
print p_force.shape
print p_force

laplacian_u = zeros(3,1)

for i in range (0,3):
    laplacian_u[i,0] = diff(u[i],X[0,0],2)+diff(u[i],X[1,0],2)+diff(u[i],X[2,0],2)

laplacian_u = simplify(laplacian_u)

#laplacian_u = nabla.square(u)
#print laplacian_u

r = symbols('r')

force = -1/r*laplacian_u + p_force
#force = -1/r*laplacian_u
#force = p_force
force = simplify(force)

print '-------------------'
print 'pressure:'
print ccode(p[0])
print '-------------------'


for i in range(0,u.shape[0]):
    print '-------------------'
    print 'velocity component '+str(i)+':'
    print ccode(u[i,0])
print '-------------------'

for i in range(0,u.shape[0]):
    print '-------------------'
    print 'force component '+str(i)+':'
    print ccode(force[i,0])
print '-------------------'

filename = 'stk_force_3d'
classname = 'StkForce3d'
directory = './'
func = force
io_tools.function_to_cpp_source(func,directory,filename,classname)

dim_domain = 3
filename = 'stk_vel_3d'
classname = 'StkVelocity3d'
func = u
io_tools.function_to_cpp_header(dim_domain,func,directory,filename,classname)

dim_domain = 3
filename = 'stk_prex_3d'
classname = 'StkPressure3d'
func = p
io_tools.function_to_cpp_header(dim_domain,func,directory,filename,classname)

