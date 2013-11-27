#include "Python.h"
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <iostream>
#include <stdexcept>
double DiracDelta(double x){
  // TODO: this doesn't seem to be a dirac delta ... should return infinity. Neil
    if((x<0.000001) & (x>-0.000001))//go on, laugh at my c++ skills
        return 1.0;
    else
        return 0.0;
};
double DiracDelta(double x,int foo){
    return 0.0;
};

double sinc(double x){
  // compute the sinc function
  if (x==0)
    return 1.0;
  else 
    return sin(x)/x;
}

double sinc_grad(double x){
  // compute the gradient of the sinc function.
  if (x==0)
    return 0.0;
  else 
    return (x*cos(x) - sin(x))/(x*x);
}
double erfcx(double x){
  // Based on code by Soren Hauberg 2010 for Octave.
  // compute the scaled complex error function.
  //return erfc(x)*exp(x*x);
  double xneg=-sqrt(log(DBL_MAX/2));
  double xmax = 1/(sqrt(M_PI)*DBL_MIN);
  xmax = DBL_MAX<xmax ? DBL_MAX : xmax;
  // Find values where erfcx can be evaluated
  double t = 3.97886080735226 / (fabs(x) + 3.97886080735226);
  double u = t-0.5;
  double y = (((((((((u * 0.00127109764952614092 + 1.19314022838340944e-4) * u 
		     - 0.003963850973605135)   * u - 8.70779635317295828e-4) * u 
		   + 0.00773672528313526668) * u + 0.00383335126264887303) * u 
		 - 0.0127223813782122755)  * u - 0.0133823644533460069)  * u 
	       + 0.0161315329733252248)  * u + 0.0390976845588484035)  * u + 0.00249367200053503304;
  y = ((((((((((((y * u - 0.0838864557023001992) * u -		       
		 0.119463959964325415) * u + 0.0166207924969367356) * u + 
	       0.357524274449531043) * u + 0.805276408752910567)  * u + 
	     1.18902982909273333)  * u + 1.37040217682338167)   * u +	
	   1.31314653831023098)  * u + 1.07925515155856677)   * u +	
	 0.774368199119538609) * u + 0.490165080585318424)  * u +	
       0.275374741597376782) * t;

  if (x<xneg)
    return -INFINITY;
  else if (x<0)
    return 2.0*exp(x*x)-y;
  else if (x>xmax)
    return 0.0;
  else 
    return y;
}

double ln_diff_erf(double x0, double x1){
  // stably compute the log of difference between two erfs.
  if (x1>x0){
    PyErr_SetString(PyExc_RuntimeError,"second argument must be smaller than or equal to first in ln_diff_erf");
    throw 1;
  }
  if (x0==x1){
    PyErr_WarnEx(PyExc_RuntimeWarning,"divide by zero encountered in log", 1);
    return -INFINITY;
  }
  else if(x0<0 && x1>0 || x0>0 && x1<0) //x0 and x1 have opposite signs
    return log(erf(x0)-erf(x1));
  else if(x0>0) //x0 positive, x1 non-negative
    return log(erfcx(x1)-erfcx(x0)*exp(x1*x1- x0*x0))-x1*x1; 
  else //x0 and x1 non-positive
    return log(erfcx(-x0)-erfcx(-x1)*exp(x0*x0 - x1*x1))-x0*x0;
}
// TODO: For all these computations of h things are very efficient at the moment. Need to recode sympykern to allow the precomputations to take place and all the gradients to be computed in one function. Not sure of best way forward for that yet. Neil
double h(double t, double tprime, double d_i, double d_j, double l){
  // Compute the h function for the sim covariance.
  double half_l_di = 0.5*l*d_i;
  double arg_1 = half_l_di + tprime/l;
  double arg_2 = half_l_di - (t-tprime)/l;
  double ln_part_1 = ln_diff_erf(arg_1, arg_2);
  arg_2 = half_l_di - t/l;
  double sign_val = 1.0;
  if(t/l==0)
    sign_val = 0.0;
  else if (t/l < 0)
    sign_val = -1.0;
  arg_2 = half_l_di - t/l;
  double ln_part_2 = ln_diff_erf(half_l_di, arg_2);
  // if either ln_part_1 or ln_part_2 are -inf, don't bother computing rest of that term.
  double part_1 = 0.0;
  if(isfinite(ln_part_1))
    part_1 = sign_val*exp(half_l_di*half_l_di - d_i*(t-tprime) + ln_part_1 - log(d_i + d_j));
  double part_2 = 0.0;
  if(isfinite(ln_part_2))
    part_2 = sign_val*exp(half_l_di*half_l_di - d_i*t - d_j*tprime + ln_part_2 - log(d_i + d_j));
  return part_1 - part_2;
}


double dh_dd_i(double t, double tprime, double d_i, double d_j, double l){
  double diff_t = (t-tprime);
  double l2 = l*l;
  double hv = h(t, tprime, d_i, d_j, l);
  double half_l_di = 0.5*l*d_i;
  double arg_1 = half_l_di + tprime/l;
  double arg_2 = half_l_di - (t-tprime)/l;
  double ln_part_1 = ln_diff_erf(arg_1, arg_2);
  arg_1 = half_l_di;
  arg_2 = half_l_di - t/l;
  double sign_val = 1.0;
  if(t/l==0)
    sign_val = 0.0;
  else if (t/l < 0)
    sign_val = -1.0;
  double ln_part_2 = ln_diff_erf(half_l_di, half_l_di - t/l);
  double base = (0.5*d_i*l2*(d_i+d_j)-1)*hv;
  if(isfinite(ln_part_1))
    base -= diff_t*sign_val*exp(half_l_di*half_l_di
				-d_i*diff_t
				+ln_part_1);
  if(isfinite(ln_part_2))
    base += t*sign_val*exp(half_l_di*half_l_di
			   -d_i*t-d_j*tprime
			   +ln_part_2);
  base += l/sqrt(M_PI)*(-exp(-diff_t*diff_t/l2)
			+exp(-tprime*tprime/l2-d_i*t)
			+exp(-t*t/l2-d_j*tprime)
			-exp(-(d_i*t + d_j*tprime)));
  return base/(d_i+d_j);

}

double dh_dd_j(double t, double tprime, double d_i, double d_j, double l){
  double half_l_di = 0.5*l*d_i;
  double hv = h(t, tprime, d_i, d_j, l);
  double sign_val = 1.0;
  if(t/l==0)
    sign_val = 0.0;
  else if (t/l < 0)
    sign_val = -1.0;
  double ln_part_2 = ln_diff_erf(half_l_di, half_l_di - t/l);
  double base = -hv;
  if(isfinite(ln_part_2))
    base += tprime*sign_val*exp(half_l_di*half_l_di-(d_i*t+d_j*tprime)+ln_part_2);
  return base/(d_i+d_j);
}

double dh_dl(double t, double tprime, double d_i, double d_j, double l){
  // compute gradient of h function with respect to lengthscale for sim covariance
  // TODO a lot of energy wasted recomputing things here, need to do this in a shared way somehow ... perhaps needs rewrite of sympykern.
  double half_l_di = 0.5*l*d_i;
  double arg_1 = half_l_di + tprime/l;
  double arg_2 = half_l_di - (t-tprime)/l;
  double ln_part_1 = ln_diff_erf(arg_1, arg_2);
  arg_2 = half_l_di - t/l;
  double ln_part_2 = ln_diff_erf(half_l_di, arg_2);
  double diff_t = t - tprime;
  double l2 = l*l;
  double hv = h(t, tprime, d_i, d_j, l);
  return 0.5*d_i*d_i*l*hv + 2/(sqrt(M_PI)*(d_i+d_j))*((-diff_t/l2-d_i/2)*exp(-diff_t*diff_t/l2)+(-tprime/l2+d_i/2)*exp(-tprime*tprime/l2-d_i*t)-(-t/l2-d_i/2)*exp(-t*t/l2-d_j*tprime)-d_i/2*exp(-(d_i*t+d_j*tprime)));
}

double dh_dt(double t, double tprime, double d_i, double d_j, double l){
  // compute gradient of h function with respect to t.
  double diff_t = t - tprime;
  double half_l_di = 0.5*l*d_i;
  double arg_1 = half_l_di + tprime/l;
  double arg_2 = half_l_di - diff_t/l;
  double ln_part_1 = ln_diff_erf(arg_1, arg_2);
  arg_2 = half_l_di - t/l;
  double ln_part_2 = ln_diff_erf(half_l_di, arg_2);
  
  return (d_i*exp(ln_part_2-d_i*t - d_j*tprime) - d_i*exp(ln_part_1-d_i*diff_t) + 2*exp(-d_i*diff_t - pow(half_l_di - diff_t/l, 2))/(sqrt(M_PI)*l) - 2*exp(-d_i*t - d_j*tprime - pow(half_l_di - t/l,2))/(sqrt(M_PI)*l))*exp(half_l_di*half_l_di)/(d_i + d_j);
}

double dh_dtprime(double t, double tprime, double d_i, double d_j, double l){
  // compute gradient of h function with respect to tprime.
  double diff_t = t - tprime;
  double half_l_di = 0.5*l*d_i;
  double arg_1 = half_l_di + tprime/l;
  double arg_2 = half_l_di - diff_t/l;
  double ln_part_1 = ln_diff_erf(arg_1, arg_2);
  arg_2 = half_l_di - t/l;
  double ln_part_2 = ln_diff_erf(half_l_di, arg_2);

  return (d_i*exp(ln_part_1-d_i*diff_t) + d_j*exp(ln_part_2-d_i*t - d_j*tprime) + (-2*exp(-pow(half_l_di - diff_t/l,2)) + 2*exp(-pow(half_l_di + tprime/l,2)))*exp(-d_i*diff_t)/(sqrt(M_PI)*l))*exp(half_l_di*half_l_di)/(d_i + d_j);
}
