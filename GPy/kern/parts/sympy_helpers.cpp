#include <math.h>
#include <float.h>
#include <stdlib.h>

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
  if (x==0)
    return 1.0;
  else 
    return sin(x)/x;
}

double sinc_grad(double x){
  if (x==0)
    return 0.0;
  else 
    return (x*cos(x) - sin(x))/(x*x);
}

double erfcx(double x){
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
  if (x0==x1)
    return INFINITY;
  else if(x0<0 && x1>0 || x0>0 && x1<0)
    return log(erf(x0)-erf(x1));
  else if(x1>0)
    return log(erfcx(x1)-erfcx(x0)*exp(x1*x1)- x0*x0)-x1*x1;
  else 
    return log(erfcx(-x0)-erfcx(-x1)*exp(x0*x0 - x1*x1))-x0*x0;
}
