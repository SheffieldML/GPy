#
# Puma forward dynamics -- 32nm = 32 inputs, high nonlinearity, med noise
#
#
Origin: simulated

Usage: assessment

Order: uninformative

Attributes:
  1  theta1	      u  [-3.1416,3.1416]	# ang position of joint 1 in radians
  2  theta2	      u  [-3.1416,3.1416]	# ang position of joint 2 in radians
  3  theta3	      u  [-3.1416,3.1416]	# ang position of joint 3 in radians
  4  theta4	      u  [-3.1416,3.1416]	# ang position of joint 4 in radians
  5  theta5	      u  [-3.1416,3.1416]	# ang position of joint 5 in radians
  6  theta6	      u  [-3.1416,3.1416]	# ang position of joint 6 in radians
  7  thetad1      u  (-Inf,Inf)	# ang vel of joint 1 in rad/sec
  8  thetad2      u  (-Inf,Inf)	# ang vel of joint 2 in rad/sec
  9  thetad3      u  (-Inf,Inf)	# ang vel of joint 3 in rad/sec
  10 thetad4      u  (-Inf,Inf)	# ang vel of joint 4 in rad/sec
  11 thetad5      u  (-Inf,Inf)	# ang vel of joint 5 in rad/sec
  12 thetad6      u  (-Inf,Inf)	# ang vel of joint 6 in rad/sec
  13 tau1     u  (-Inf,Inf)	# torque on jt 1 
  14 tau2     u  (-Inf,Inf)	# torque on jt 2
  15 tau3     u  (-Inf,Inf)	# torque on jt 3
  16 tau4     u  (-Inf,Inf)	# torque on jt 4
  17 tau5     u  (-Inf,Inf)	# torque on jt 5
  18 dm1   u	 [0,Inf)	# proportion change in mass of link 1
  19 dm2   u	 [0,Inf)	# prop change in mass of link 2
  20 dm3   u	 [0,Inf)	# prop change in mass of link 3
  21 dm4   u	 [0,Inf)	# prop change in mass of link 4
  22 dm5   u	 [0,Inf)	# prop change in mass of link 5
  23 da1    u	 [0,Inf)	# prop change in length of link 1
  24 da2    u	 [0,Inf)	# prop change in length of link 2
  25 da3    u	 [0,Inf)	# prop change in length of link 3
  26 da4    u	 [0,Inf)	# prop change in length of link 4
  27 da5    u	 [0,Inf)	# prop change in length of link 5
  28 db1      u	 [0,Inf)	# prop change in visc friction of link 1
  29 db2      u	 [0,Inf)	# prop change in visc friction of link 2
  30 db3      u	 [0,Inf)	# prop change in visc friction of link 3
  31 db4      u	 [0,Inf)	# prop change in visc friction of link 4
  32 db5      u	 [0,Inf)	# prop change in visc friction of link 5
  33 thetadd6     u  (-Inf,Inf)	# ang acceleration of joint 6
