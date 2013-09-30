#include <math.h>
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
