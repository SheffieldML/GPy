#include <math.h>
double DiracDelta(double x);
double DiracDelta(double x, int foo);

double sinc(double x);
double sinc_grad(double x);

double erfcx(double x);
double ln_diff_erf(double x0, double x1);

double h(double t, double tprime, double d_i, double d_j, double l);
double dh_dl(double t, double tprime, double d_i, double d_j, double l);
double dh_dd_i(double t, double tprime, double d_i, double d_j, double l);
double dh_dd_j(double t, double tprime, double d_i, double d_j, double l);
double dh_dt(double t, double tprime, double d_i, double d_j, double l);
double dh_dtprime(double t, double tprime, double d_i, double d_j, double l);
