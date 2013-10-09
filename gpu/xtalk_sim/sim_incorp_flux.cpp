#include "xtalk_sim.h"

#include "sim_2mer_flux_data.h"

DATA_TYPE GetIncorpFlux(double time)
{
	double time_last = sim_incorp_flux[(sim_incorp_flux_len-1)*2];
	
	if (time < sim_incorp_flux[0])
		return(0.0);
	
	double dt = (time_last - sim_incorp_flux[0])/(sim_incorp_flux_len-1);
	
	int ileft = (int)((time - sim_incorp_flux[0])/dt);
	int iright = ileft + 1;
	double frac = ((time - sim_incorp_flux[0]) - ileft*dt)/dt;
	
	if (iright >= sim_incorp_flux_len)
		return(0.0);

	DATA_TYPE lval = sim_incorp_flux[ileft*2+1];
	DATA_TYPE rval = sim_incorp_flux[iright*2+1];
	
	DATA_TYPE ret = lval*(1-frac)+rval*frac;
	return(ret);
}























