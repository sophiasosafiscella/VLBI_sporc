from numpy import *
from numpy.linalg import *
import sys

#Angles in radians, info from Table 2
alpha0437VLBI = (4*3600 + 37*60 + 15 + 0.883250)*(2*pi/86400);
delta0437VLBI = -(47*3600 + 15*60 + 9 + 0.031863)*(2*pi/1296000);
alpha0437Timing = (4*3600 + 37*60 + 15 + 0.883186)*(2*pi/86400);
delta0437Timing = -(47*3600 + 15*60 + 9 + 0.034033)*(2*pi/1296000);
##propagate the 1713 values forward accounting for proper motion between epochs (use Hotan et al. values)
properMotionCorrect = 0
alpha1713VLBI = (17*3600 + 13*60 + 49 + 0.5306 + 0.001653*properMotionCorrect)*(2*pi/86400);
delta1713VLBI = (7*3600 + 47*60 + 37 + 0.519 - 0.0185*properMotionCorrect)*(2*pi/1296000);
alpha1713Timing = (17*3600 + 13*60 + 49 + 0.53077 + 0.001653*properMotionCorrect)*(2*pi/86400);
delta1713Timing = (7*3600 + 47*60 + 37 + 0.5228 - 0.0185*properMotionCorrect)*(2*pi/1296000);

'''
#propagate J1713 angles forward to J0437 epoch using proper motion from Hotan et al. (2006)
alpha1713VLBI += 0.00497*((54100-52275)/365)*(2*pi/1296000)
alpha1713Timing += 0.00497*((54100-52275)/365)*(2*pi/1296000)
delta1713VLBI += -0.0037*((54100-52275)/365)*(2*pi/1296000)
delta1713Timing += -0.0037*((54100-52275)/365)*(2*pi/1296000)
'''

#Form position unit vectors
nhat0437VLBI = array([cos(delta0437VLBI)*cos(alpha0437VLBI),cos(delta0437VLBI)*sin(alpha0437VLBI),sin(delta0437VLBI)]);
nhat0437Timing = array([cos(delta0437Timing)*cos(alpha0437Timing),cos(delta0437Timing)*sin(alpha0437Timing),sin(delta0437Timing)]);
nhat1713VLBI = array([cos(delta1713VLBI)*cos(alpha1713VLBI),cos(delta1713VLBI)*sin(alpha1713VLBI),sin(delta1713VLBI)]);
nhat1713Timing = array([cos(delta1713Timing)*cos(alpha1713Timing),cos(delta1713Timing)*sin(alpha1713Timing),sin(delta1713Timing)]);

#Form difference vectors
d0437 = nhat0437Timing - nhat0437VLBI
d1713 = nhat1713Timing - nhat1713VLBI

#Form M matrix
M = [[0,-nhat0437VLBI[2],nhat0437VLBI[1]],
     [nhat0437VLBI[2],0,-nhat0437VLBI[0]],
     [-nhat0437VLBI[1],nhat0437VLBI[0],0],
     [0,-nhat1713VLBI[2],nhat1713VLBI[1]],
     [nhat1713VLBI[2],0,-nhat1713VLBI[0]],
     [-nhat1713VLBI[1],nhat1713VLBI[0],0]]

#Form Sigma matrix (hack version),info from Table 2, see just above start of section 4
#quantities in radians
sigma0437VLBI = sqrt(cos(delta0437VLBI)**2*(0.000003*(2*pi/86400))**2+(0.000037*(2*pi/1296000))**2)
sigma0437Timing = sqrt(cos(delta0437Timing)**2*(0.000006*(2*pi/86400))**2+(0.000070*(2*pi/1296000))**2)
sigma1713VLBI = sqrt(cos(delta1713VLBI)**2*(0.0001*(2*pi/86400))**2+(0.002*(2*pi/1296000))**2)
sigma1713Timing = sqrt(cos(delta1713Timing)**2*(0.00001*(2*pi/86400))**2+(0.0002*(2*pi/1296000))**2)

Sigma = diag([sigma0437VLBI**2+sigma0437Timing**2,
              sigma0437VLBI**2+sigma0437Timing**2,
              sigma0437VLBI**2+sigma0437Timing**2,
              sigma1713VLBI**2+sigma1713Timing**2,
              sigma1713VLBI**2+sigma1713Timing**2,
              sigma1713VLBI**2+sigma1713Timing**2])

print(Sigma)
#Form D
D = [d0437[0],d0437[1],d0437[2],d1713[0],d1713[1],d1713[2]];

print(D)

#Do Equation 12 computation
cov = dot(transpose(M),dot(inv(Sigma),M))
Ahat = dot(dot(dot(inv(cov),transpose(M)),inv(Sigma)),D)

#Convert Ahat from radians to mas
Ahat *= 1296000000/(2*pi)
print(Ahat)
