

from rockit import *

import matplotlib.pyplot as plt
import numpy as np

from numpy import pi, cos, sin, 
from casadi import vertcat, sumsqr

ocp = Ocp(T=FreeTime(10.0))

# Unicycle model

x     = ocp.state()
y     = ocp.state()
theta = ocp.state()

V = ocp.control()
omega     = ocp.control()

ocp.set_der(x, V*cos(theta))
ocp.set_der(y, V*sin(theta))
ocp.set_der(theta,omega)

# Initial constraints
ocp.subject_to(ocp.at_t0(x)==0)
ocp.subject_to(ocp.at_t0(y)==0)
ocp.subject_to(ocp.at_t0(theta)==0)

# Final constraint
ocp.subject_to(ocp.at_tf(x)==3)
ocp.subject_to(ocp.at_tf(y)==4)

ocp.set_initial(x,0)
ocp.set_initial(y,ocp.t)
ocp.set_initial(theta, 0)
ocp.set_initial(V,1)
ocp.set_initial(omega,0)

ocp.subject_to(0 <= (V<=1))
ocp.subject_to( -pi <= (omega<= pi))

# Minimal time
ocp.add_objective(ocp.T)
ocp.add_objective(ocp.integral( (x)**2 + (y)**2))

# Pick a solution method
ocp.solver('ipopt')

# Make it concrete for this ocp
ocp.method(MultipleShooting(N=20,M=4,intg='rk'))

# solve
sol = ocp.solve()

from pylab import *
figure()

ts, xs = sol.sample(x, grid='control')
ts, ys = sol.sample(y, grid='control')

plot(xs, ys,'bo')

ts, xs = sol.sample(x, grid='integrator')
ts, ys = sol.sample(y, grid='integrator')

plot(xs, ys, 'b.')

ts, xs = sol.sample(x, grid='integrator',refine=10)
ts, ys = sol.sample(y, grid='integrator',refine=10)

plot(xs, ys, '-')

axis('equal')
show(block=True)
