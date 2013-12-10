## Copyright (C) 2010 Soren Hauberg
##
## Copyright James Hensman 2011
## 
## This program is free software; you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or (at
## your option) any later version.
##
## This program is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
## General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program; see the file COPYING.  If not, see
## <http://www.gnu.org/licenses/>.

import numpy as np

def erfcx (arg):
	arg = np.atleast_1d(arg)
	assert(np.all(np.isreal(arg)),"erfcx: input must be real")

	## Get precision dependent thresholds -- or not :p
	xneg = -26.628;
	xmax = 2.53e+307;

	## Allocate output
	result = np.zeros (arg.shape)

	## Find values where erfcx can be evaluated
	idx_neg = (arg < xneg);
	idx_max = (arg > xmax);
	idx = ~(idx_neg | idx_max);

	arg = arg [idx];

	## Perform the actual computation
	t = 3.97886080735226 / (np.abs (arg) + 3.97886080735226);
	u = t - 0.5;
	y = (((((((((u * 0.00127109764952614092 + 1.19314022838340944e-4) * u \
	    - 0.003963850973605135)   * u - 8.70779635317295828e-4) * u +     \
	      0.00773672528313526668) * u + 0.00383335126264887303) * u -     \
	      0.0127223813782122755)  * u - 0.0133823644533460069)  * u +     \
	      0.0161315329733252248)  * u + 0.0390976845588484035)  * u +     \
	      0.00249367200053503304;
	y = ((((((((((((y * u - 0.0838864557023001992) * u -           \
	      0.119463959964325415) * u + 0.0166207924969367356) * u + \
	      0.357524274449531043) * u + 0.805276408752910567)  * u + \
	      1.18902982909273333)  * u + 1.37040217682338167)   * u + \
	      1.31314653831023098)  * u + 1.07925515155856677)   * u + \
	      0.774368199119538609) * u + 0.490165080585318424)  * u + \
	      0.275374741597376782) * t;

	y [arg < 0] = 2 * np.exp (arg [arg < 0]**2) - y [arg < 0];

	## Put the results back into something with the same size is the original input
	result [idx] = y;
	result [idx_neg] = np.inf;
	## result (idx_max) = 0; # not needed as we initialise with zeros
	return(result)

