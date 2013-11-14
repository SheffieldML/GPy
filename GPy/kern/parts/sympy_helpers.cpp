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
  // compute the scaled complex error function.
  double xneg=-sqrt(log(DBL_MAX/2));
  double xmax = 1/(sqrt(M_PI)*DBL_MIN);
  xmax = DBL_MAX<xmax ? DBL_MAX : xmax;
  // Find values where erfcx can be evaluated
  double t = 3.97886080735226 / (abs(x) + 3.97886080735226);
  double u = t-0.5;
  double y = (((((((((u * 0.00127109764952614092 + 1.19314022838340944e-4) * u 
	      - 0.003963850973605135)   * u - 8.70779635317295828e-4) * u 
	    + 0.00773672528313526668) * u + 0.00383335126264887303) * u 
	  - 0.0127223813782122755)  * u - 0.0133823644533460069)  * u 
	+ 0.0161315329733252248)  * u + 0.0390976845588484035)  * u + 0.00249367200053503304;
  if (x<xneg)
    return -INFINITY;
  else if (x<0)
    return 2*exp(x*x)-y;
  else if (x>xmax)
    return 0.0;
  else 
    return y;
}

double ln_diff_erf(double x0, double x1){
  // stably compute the log of difference between two erfs.
  if (x1>x0)
    throw std::runtime_error("Error: second argument must be smaller than first in ln_diff_err");
  return log(erf(x0) - erf(x1));
  if (x0==x1)
    return -INFINITY;
  else if(x0<0 && x1>0 || x0>0 && x1<0)
    return log(erf(x0)-erf(x1));
  else if(x1>0)
    return log(erfcx(x1)-erfcx(x0)*exp(x1*x1- x0*x0))-x1*x1;
  else 
    return log(erfcx(-x0)-erfcx(-x1)*exp(x0*x0 - x1*x1))-x0*x0;
}

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
  double ln_part_2 = ln_diff_erf(half_l_di, arg_2);
  
  return sign_val*exp(half_l_di*half_l_di - d_i*(t-tprime) + ln_part_1 - log(d_i + d_j)) - sign_val*exp(half_l_di*half_l_di - d_i*t - d_j*tprime + ln_part_2 - log(d_i + d_j));
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

  double base = ((0.5*d_i*l2*(d_i+d_j)-1)*hv 
		 + (-diff_t*sign_val*exp(half_l_di*half_l_di
					 -d_i*diff_t
					 +ln_part_1)
		    +t*sign_val*exp(half_l_di*half_l_di
				    -d_i*t-d_j*tprime
				    +ln_part_2))
		 + l/sqrt(M_PI)*(-exp(-diff_t*diff_t/l2)
			       +exp(-tprime*tprime/l2-d_i*t)
			       +exp(-t*t/l2-d_j*tprime)
			       -exp(-(d_i*t + d_j*tprime))));
  return base/(d_i+d_j);
}

double dh_dd_j(double t, double tprime, double d_i, double d_j, double l){
  double diff_t = (t-tprime);
  double l2 = l*l;
  double half_l_di = 0.5*l*d_i;
  double hv = h(t, tprime, d_i, d_j, l);
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
  double base = tprime*sign_val*exp(half_l_di*half_l_di-(d_i*t+d_j*tprime)+ln_part_2)-hv;
  return base/(d_i+d_j);
}


double dh_dt(double t, double tprime, double d_i, double d_j, double l){
  return 0.0;
}

double dh_dtprime(double t, double tprime, double d_i, double d_j, double l){
  return 0.0;
}
