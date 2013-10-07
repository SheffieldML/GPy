#include <math.h> 

 double k_uu(t1,t2,theta1,theta2,sig1,sig2)
 {
  double kern=0;
  double dist=0;
  
  dist = sqrt(t2*t2-t1*t1) 
 
  kern = sig1*(1+theta1*dist)*exp(-theta1*dist)

 return kern;
 }



 double k_yy(t1, t2, theta1,theta2,sig1,sig2)
 {
  double kern=0;
  double dist=0;
  
  dist = sqrt(t2*t2-t1*t1) 
 
  kern = sig1*sig2 * (  exp(-theta1*dist)*(theta2-2*theta1+theta1*theta2*dist-theta1*theta1*dist) +
  	exp(-dist)  ) / ((theta2-theta1)*(theta2-theta1))

  return kern;
 } 






	



