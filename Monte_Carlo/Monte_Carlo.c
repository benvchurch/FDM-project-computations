/*
 * Monte Carlo.c
 * 
 * Copyright 2017 Benjamin Church <ben@U430>
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301, USA.
 * 
 * 
 */


#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_hyperg.h>
#include <gsl/gsl_sf_gamma.h>

#define min(x,y) (x < y? x : y)
#define sign(x) (x >= 0? 1.0 : -1.0)

gsl_rng *RNG;

double pi = 3.14159265;
double crit_density = 1.3211775*0.0000001, 
	f = 0.1,
	g = 0.0001,
	p = 1.9,
	c = 10.0,
	G = 0.0045,
	k = 2,
	Mprimary = 1000000000000,
	T_age = 10000;
	
double m_max, m_min, expected_num, R_core_prim, R_max_prim;

double MaxRadius(double M)
{
    return pow(3.0*M/(4.0 * pi * 200.0 * crit_density), 1.0/3.0);
}


typedef struct
{
	double R, theta;
	double M, r_core, r_max, alpha, beta, c;
	double v_r, v_theta, v_phi;
} halo;

double MFreeNFW(double r, double M)
{
    double Rmax = MaxRadius(M);
    double Rc = Rmax/c;
    if(r < Rmax)
        return M*(log(1+r/Rc)-r/(r+Rc))/(log(1+c) - c/(1+c));
    else
        return M;
}

double DFreeNFW(double r, double M)
{
    double Rmax = MaxRadius(M);
    double Rc = Rmax/c;
    if(r < Rmax)
        return 200.0/3.0 * crit_density / (log(1+c) - c/(1+c))*pow(c, 3.0) * 1.0/(r/Rc*pow(1+r/Rc, 2.0));
    else
        return 0.0;
}

double PhiFreeNFW(double r, double M)
{
    double Rmax = MaxRadius(M);
    double Rc = Rmax/c;
    if(r < Rmax)
        return -M*G*((Rmax/r * log(1+r/Rc) - log(1+c))/(log(1+c) - c/(1+c)) + 1)/Rmax;
    else
        return -M*G/r;
}

double TidalRadius(double M, double R)
{
	return R*pow(M/(2*MFreeNFW(R, Mprimary)), 1.0/3.0);
}

double func(double x, double r)
{
    return (log(1.0+x) - x/(1.0+x))/(log(1.0+c)-c/(1.0+c)) - r;
}

double df (double x)
{
    return x/pow(1.0+x, 2.0) * 1.0/(log(1.0+c)-c/(1.0+c));
}

double newton(double r)
{
    int itr, maxmitr = 100;
    double h, x0 = 0.5, x1, allerr = 0.00001;
    
    for (itr = 0; itr < maxmitr; itr++)
    {
        h = func(x0, r)/df(x0);
        x1 = x0 - h;

        if (abs(h) < allerr)
        {
           return x1;
        }
        x0 = x1;
    }
    return x1;
}

double get_R()
{
	return newton(gsl_rng_uniform(RNG)) * R_core_prim;
}

double get_theta()
{
	return acos(2*gsl_rng_uniform(RNG)-1);
}

double get_M()
{
	double r = gsl_ran_flat(RNG, 0, 1);
	return Mprimary*pow(pow(m_max/Mprimary, 1.0-p)*r + pow(m_min/Mprimary, 1.0-p)*(1.0-r), 1.0/(1.0-p));
}

void set_shape(halo *ptr, double M, double R)
{
	ptr->r_core = pow(3*M/(4 * pi * 200.0 * crit_density), 1.0/3.0)/c;
	ptr->r_max = min(pow(3*M/(4 * pi * 200.0 * crit_density), 1.0/3.0), TidalRadius(M, R));
	ptr->alpha = 1.0;
	ptr->beta = 3.0;
	ptr->c = c;
}

void set_velocity(halo *ptr, double R)
{
	double v_sigma = sqrt(1.0/3.0)*sqrt(-PhiFreeNFW(R, Mprimary));
	
	ptr->v_r = gsl_ran_gaussian_ziggurat(RNG, v_sigma);
	ptr->v_theta = gsl_ran_gaussian_ziggurat(RNG, v_sigma);
	ptr->v_phi = gsl_ran_gaussian_ziggurat(RNG, v_sigma);
}

double enclosed_mass(halo *ptr, double r)
{
	double x = r/(ptr->r_core);
	double c_halo = ptr->c;
	if(x > c_halo) 
		return ptr->M;
	else
		return ptr->M * (log(1+x)-x/(1+x))/(log(1+c_halo) - c_halo/(1+c_halo));
}


void truncate(halo *ptr, double R)
{
	double Rt = TidalRadius(ptr->M, R); 
	double R_max = min(ptr->r_max, Rt);
	ptr->M = enclosed_mass(ptr, R_max);
	
	ptr->r_max = R_max;
	ptr->c = R_max/ptr->r_core;
}

halo *make_halo()
{
	halo *ptr = (halo *) malloc(sizeof(halo));
	ptr->R = get_R();
	ptr->theta = get_theta();
	
	ptr->M = get_M();
	set_shape(ptr, ptr->M, ptr->R);
	set_velocity(ptr, ptr->R);
	truncate(ptr, ptr->R);
	
	return ptr;
}
 
void print_halo_basic(halo *ptr)
{
	printf("\n R = %3f \n theta = %3f \n M = %3f \n", ptr->R, ptr->theta, ptr->M);
}


double Fluc(halo **halos, int num_halos, double D)
{
	double sum = 0;
	int i;
	for(i = 0; i < num_halos; i++)
	{
		double R = halos[i]->R;
		double r = sqrt(R*R + D*D - 2.0*R*D*cos(halos[i]->theta));
		double s = sin(halos[i]->theta);
		sum += pow(enclosed_mass(halos[i], r) * G /(r*r), 2.0) * (pow(halos[i]->v_r, 2.0) + pow(halos[i]->v_theta, 2.0) + 2*(halos[i]->v_r)*(halos[i]->v_theta) * D*s/r * sqrt(1 - pow(D*s/r, 2.0)) * sign(D-R));
		//printf("%f ", (pow(halos[i]->v_r, 2.0) + pow(halos[i]->v_theta, 2.0) + 2*(halos[i]->v_r)*(halos[i]->v_theta) * D*s/r * sqrt(1 - pow(D*s/r, 2.0)) * sign(D-R)));
	}
	return sum;
}

double H_Density(halo **halos, int num_halos, double D, double dD)
{
	double sum = 0;
	int i;
	for(i = 0; i < num_halos; i++)
	{
		if(halos[i]->R < D && halos[i]->R > D - dD)
			sum += halos[i]->M;
	}
	return sum/(4.0*pi*D*D*dD);
}

int hist(halo **halos, int num_halos, double D, double dD)
{
	double sum = 0;
	int i;
	for(i = 0; i < num_halos; i++)
	{
		if(halos[i]->R  < D && halos[i]->R > D - dD)
			sum ++;
	}
	return sum;
}

int main(int argc, char **argv)
{
	
	//printf("%f\n", log(0.0001));
	//getchar();

	const gsl_rng_type *T;
	R_max_prim = MaxRadius(Mprimary);
	R_core_prim = R_max_prim/c; 
	m_max = f*Mprimary;
	m_min = g*Mprimary;
	expected_num = (2.0-p)/(1.0-p)*f*(pow(m_max/Mprimary, 1.0-p) - pow(m_min/Mprimary, 1.0-p))/(pow(m_max/Mprimary, 2.0-p) - pow(m_min/Mprimary, 2.0-p));

	gsl_rng_env_setup();

	T = gsl_rng_default;
	RNG = gsl_rng_alloc (T);
	
	int num_trials = 10000;
	int num_points = 100;
	double Flucs[num_points], Dens[num_points], Hist[num_points];
	double mass = 0, avg_mass = 0;
	int i,j;
	
	for(i = 0; i < num_points; i++)
	{
		Flucs[i] = 0;
		Dens[i] = 0;
		Hist[i] = 0;
	}
	
	for(i = 0; i < num_trials; i++)
	{
		if(i % 100 == 0)printf("%d out of %d \n", i/100, num_trials/100);
		unsigned long num_halos = gsl_ran_poisson(RNG, expected_num);
		
		mass = 0;
		//printf("Expect: %f   Have: %lu \n", expected_num, num_halos);
		
		halo *halolist[num_halos];
		for(j = 0; j < num_halos; j++)
		{
			halolist[j] = make_halo();
			mass += halolist[j]->M;
			//print_halo_basic(halolist[j]);
		}

		for(j = 0; j < num_points; j++)
		{
			Flucs[j] += Fluc(halolist, num_halos, pow(10, 5.0*j/num_points))/num_trials;
			Dens[j] += H_Density(halolist, num_halos, R_max_prim*j/(double)num_points, R_max_prim/(double)num_points)/num_trials;
			Hist[j] += hist(halolist, num_halos, R_max_prim*(j+1)/(double)num_points, R_max_prim/(double)num_points)/num_trials;
		}
		//printf("Mass frac: %f \n", mass/Mprimary);
		avg_mass += mass/num_trials;
		
		/*for(j = 0; j < num_points; j++)
		{
			//printf("%f : %f\n", pow(10, 5.0*j/num_points), log10(Flucs[j]));
			//printf("%f : %f\n", pow(10, 5.0*j/num_points), log10(Dens[j]));
			//printf("%f : %f\n", pow(10, 5.0*j/num_points), Hist[j]);
		}*/
		
		for(j = 0; j < num_halos; j++)
			free(halolist[j]);
	}
	for(j = 0; j < num_points; j++)
	{
		printf("%f : %f\n", pow(10, 5.0*j/num_points), log10(Flucs[j]));
		//printf("%f : %f\n", log10(R_max_prim*(j+1)/(double)num_points), log10(Dens[j]));
	}
	printf("Mass frac: %f \n", avg_mass/Mprimary);
	gsl_rng_free (RNG);
	return 0;
}

