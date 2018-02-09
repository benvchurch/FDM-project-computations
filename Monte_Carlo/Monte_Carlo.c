/*
 * Monte Carlo.cpp
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
#include <string.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_hyperg.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_integration.h>
#include <time.h>

#define INTEGRATION_POINTS 81
#define INTEGRAL_FUDGE_FACTOR 1.0125
#define INTEGRATION_SECTIONS 8
#define CUTOFF_SCALE 0
#define min(x,y) (x < y? x : y)
#define sign(x) (x >= 0? 1.0 : -1.0)
#define dot_macro(A, B) ((A).x * (B).x + (A).y * (B).y + (A).z * (B).z)
#define mag_sq(A) dot_macro(A, A)
#define mag(A) sqrt(dot_macro(A, A))
#define assign_vec(A, x_val, y_val, z_val) {(A).x = x_val; (A).y = y_val; (A).z = z_val;}
#define assign_vec_diff(d, A, B) {(d).x = (A).x - (B).x; (d).y = (A).y - (B).y; (d).z = (A).z - (B).z;}
#define make_unit(A) {double len = mag(A); assign_vec(A, (A.x)/len, (A.y)/len, (A.z)/len);}
#define rho_profile(x, a, b) (pow(x, 2 - a)*pow((1 + x), a - b))

gsl_rng *RNG;
gsl_integration_workspace * w;

const int default_log_num_trials = 6;
int num_trials, num_points = 100;

const double pi = 3.14159265;
double crit_density = 1.3211775E-7,
	f = 0.1,
	g = 1E-3,
	p = 1.9,
	G = 0.0045,
	k = 2,
	M_prim = 1E12,
	T_age = 1E4;

double m_max, m_min, expected_num, R_core_prim, R_max_prim, c_prim;

double MaxRadius(double M)
{
    return pow(3.0*M/(4.0 * pi * 200.0 * crit_density), 1.0/3.0);
}

typedef struct
{
	double x, y, z;
} vector;

typedef struct
{
	double m, s, max_val;
}data_cell;

void update_cell(data_cell *ptr, double new_data, int k)
{
	double old_m = ptr->m;
	ptr->m += (new_data - old_m)/(double) k;
	ptr->s += (new_data - old_m)*(new_data - ptr->m);
	if(ptr->max_val < new_data) ptr->max_val = new_data;
}

void reset_cell(data_cell *ptr)
{
	ptr->m = 0;
	ptr->s = 0;
	ptr->max_val = 0;
}

typedef struct
{
	double R, theta, phi;
	double cos_theta;
	double M, r_core, r_max, alpha, beta, c, normalization;
	double integs[INTEGRATION_SECTIONS];
	vector v, position;
} halo;

double Power(double x, int y)
{
	if(y > 0)
	{
		double result = 1;
		while(y > 0)
		{
			if(y % 2 == 0)
			{
				x = x*x;
				y = y/2;
			}
			else
			{
				result = x*result;
				y--;
			}
		}
		return result;
	}
	else if(y < 0)
		return 1/Power(x, -y);
	else
		return 1;
}

void print_vector(vector *A)
{
	printf("(%.3f, %.3f, %.3f)\n", A->x, A->y, A->z);
}

double perp_mag_sq(vector *A, vector *u)
{
	double dot_prod = dot_macro(*A, *u);
	return mag_sq(*A) - dot_prod*dot_prod/mag_sq(*u);
}

double MFreeNFW(double r)
{
    if(r < R_max_prim)
        return M_prim*(log(1 + r/R_core_prim)-r/(r + R_core_prim))/(log(1 + c_prim) - c_prim/( 1 + c_prim));
    else
        return M_prim;
}

double DFreeNFW(double r)
{
    if(r < R_max_prim)
        return 200.0/3.0 * crit_density / (log(1 + c_prim) - c_prim/(1 + c_prim))*pow(c_prim, 3) * 1.0/(r/R_core_prim*pow(1+r/R_core_prim, 2.0));
    else
        return 0.0;
}

double PhiFreeNFW(double r)
{
    if(r < R_max_prim)
        return -M_prim*G*((R_max_prim/r * log(1 + r/R_core_prim) - log(1 + c_prim))/(log(1 + c_prim) - c_prim/(1 + c_prim)) + 1)/R_max_prim;
    else
        return -M_prim*G/r;
}

double TidalRadius(double M, double R)
{
	return R*pow(M/(2*MFreeNFW(R)), 1.0/3.0);
}

double NFW_func(double x, double r)
{
    return (log(1.0 + x) - x/(1.0 + x))/(log(1.0 + c_prim) - c_prim/(1.0 + c_prim)) - r;
}

double df (double x)
{
    return x/((1.0 + x)*(1.0 + x)) * 1.0/(log(1.0 + c_prim) - c_prim/(1.0 + c_prim));
}

double newton(double r)
{
    int itr, maxmitr = 100;
    double h, x0 = 0.1, x1, allerr = pow(10, -6);

    for (itr = 0; itr < maxmitr; itr++)
    {
        h = NFW_func(x0, r)/df(x0);
        x1 = x0 - h;

        if (fabs(h) < allerr)
        {
            return x1;
        }
        x0 = x1;
    }
    return x1;
}

double rho_profile_func(double x, void *params)
{
	double *param_ptr = (double *)params;
	return pow(x, 2 - param_ptr[0])*pow((1 + x), param_ptr[0] - param_ptr[1]);
}

double integ_profile(double alpha, double beta, double r)
{
	/*if(alpha > 3)
		return 1;
	double dist = 0, step = r/INTEGRATION_POINTS, sum = 0;
	int i;

	for(i = 0; i < INTEGRATION_POINTS; i++)
	{
		if(i == 0 && alpha > 2 && alpha < 3)
			sum += 8.0/(3.0*step) * 1/(3.0 - alpha) * pow(step, 3 - alpha);
		else if(i == 0 || i == INTEGRATION_POINTS - 1)
			sum += rho_profile(dist, alpha, beta);
		else if(i%3 == 0)
			sum += 2*rho_profile(dist, alpha, beta);
		else
			sum += 3*rho_profile(dist, alpha, beta);
		dist += step;
	}

	return 3.0*step/8.0*sum * INTEGRAL_FUDGE_FACTOR;*/
	//printf("int %f\n", 3*step/8*sum);

	double result, error;
	double arr[2];
	arr[0] = alpha;
	arr[1] = beta;

	gsl_function F;
	F.function = &rho_profile_func;
  	F.params = arr;

  	gsl_integration_qags (&F, 0, r, 0, 1e-7, 1000, w, &result, &error);
	if(error/result > 1e-7)printf("LARGE INTEG ERROR WARNING %f for result %f", error, result);

	//printf("alpha: %.3f beta: %.3f r: %.3f err %f\n", alpha, beta, r, (result - quick_result)/result);

	return result;
}

double worst_thing_ever(int lr, double a, double b, double r)
{
	switch(lr)
	{
		case -2: return pow(2,-2 + b)*pow(3,-4 + a - b)*(-20.25 - ((8*a*a*a + 12*a*a*(-2 + b) + b*(38 + (-15 + b)*b) +
					2*a*(8 + 3*(-7 + b)*b))*(-0.0078125 + Power(1 - 2*r,4)/8.))/2. +
					81*r - 27*(-6 + 2*a + b)*(0.1875 + (-1 + r)*r) + (3*(4*a*a + 4*a*(-4 + b) + (-9 + b)*(-2 + b))*(-1 + 4*r)*(7 + 4*r*(-5 + 4*r)))/32.);
		break;

		case -1: return (pow(2,-4 + a - b)*(-24 + 48*r - ((-2 + a + b)*(a*a + (-7 + b)*b + a*(-1 + 2*b))*(-3 + 2*r)*(-1 + 2*r)*(5 + 4*(-2 + r)*r))/64. -
					3*(-4 + a + b)*(3 - 8*r + 4*r*r) + 6*(8 + a*a - 7*b + b*b + a*(-5 + 2*b))*(-0.2916666666666667 + (r*(3 + (-3 + r)*r))/3.)))/3.0;
		break;

		case 0: return pow(2,-4 - a)*pow(3,-4 + a - b)*(-(a*a*a*(-1 + Power(-2 + r,4))) -
					16*(27 + (3*b)/4. + (b*b*b*(-1 + Power(-2 + r,4)))/2. + 40*b*r - 48*b*r*r - 27*r*r*r + 4*b*r*r*r +
					(13*b*r*r*r*r)/4. - 3*b*b*(1 + Power(-2 + r,3)*r)) - 3*a*a*(-1 + r)*(-41 + 2*b*(-3 + r)*(5 + (-4 + r)*r) - r*(-23 + r + r*r)) -
					2*a*(-1 + r)*(-75 + 6*b*b*(-3 + r)*(5 + (-4 + r)*r) + r*(-187 + r*(77 + r)) - 3*b*(37 + r*(5 + r*(-19 + 5*r)))));
		break;

		case 1: return  -(pow(2,-5 - 2*a)*pow(5,-3 + a - b)*(-2 + r)*(a*a*a*(-120 + 68*r - 14*r*r + r*r*r) +
					a*a*(-1880 + 596*r - 38*r*r - 3*r*r*r + 12*b*(-120 + 68*r - 14*r*r + r*r*r)) +
					2*a*(-2200 - 1932*r + 426*r*r + r*r*r + b*(-3920 + 344*r + 268*r*r - 42*r*r*r) + 24*b*b*(-120 + 68*r - 14*r*r + r*r*r)) +
					8*(-500*(4 + 2*r + r*r) + 8*b*b*b*(-120 + 68*r - 14*r*r + r*r*r) - 4*b*b*(40 + 212*r - 86*r*r + 9*r*r*r) +
					b*(-200 - 1892*r + 206*r*r + 31*r*r*r))))/3.0;
		break;

		case 2: return  pow(2,-4 - 3*a)*pow(3,-7 + 2*a - 2*b)*(-8957952 - ((-2 + a)*(-1 + a)*a + 8*(182 + 3*(-11 + a)*a)*b + 192*(-10 + a)*b*b + 512*b*b*b)*
					(-64 + Power(-8 + r,4)/4.) + 2239488*r - 15552*(-18 + a + 8*b)*(48 - 16*r + r*r) +
					72*(162 + a*a - 224*b + 64*b*b + a*(-19 + 16*b))*(-448 + r*(192 + (-24 + r)*r)));

		break;

		case 3: return (pow(2,-5 - 4*a)*pow(17,-3 + a - b)*(-965935104 - (a*a*a + a*a*(-3 + 48*b) + 32*b*(307 - 432*b + 128*b*b) + a*(2 - 912*b + 768*b*b))*
					(-1024 + Power(-16 + r,4)/4.) + 120741888*r - 221952*(-34 + a + 16*b)*(192 - 32*r + r*r) +
					272*(578 + a*a - 832*b + 256*b*b + a*(-35 + 32*b))*(-3584 + r*(768 + (-48 + r)*r))))/3.0;

		break;

		case 4: return pow(2,-6 - 5*a)*pow(33,-4 + a - b)*11*(-113048027136 -
				(a*a*a + a*a*(-3 + 96*b) + 64*b*(1123 - 1632*b + 512*b*b) + a*(2 - 3360*b + 3072*b*b))*(-16384 + Power(-32 + r,4)/4.) + 7065501696*r -
				3345408*(-66 + a + 32*b)*(768 - 64*r + r*r) + 1056*(2178 + a*a - 3200*b + 1024*b*b + a*(-67 + 64*b))*(-28672 + r*(3072 + (-96 + r)*r)));

		break;

		case 5: return (pow(2,-7 - 6*a)*pow(65,-3 + a - b)*
     			(-13822328832000 + 431947776000*r - (((-2 + a)*(-1 + a)*a + 64*(8582 + 3*(-67 + a)*a)*b +
            	12288*(-66 + a)*b*b + 262144*b*b*b)*(-96 + r)*(-32 + r)*
          		(5120 + (-128 + r)*r))/4. - 51916800*(-130 + a + 64*b)*(3072 - 128*r + r*r) + 4160*(8450 + a*a - 12544*b + 4096*b*b + a*(-131 + 128*b))*
        		(-229376 + r*(12288 + (-192 + r)*r))))/3.0;
		break;
	};

	return 1;
}

double sec_integ_profile(halo *ptr, double r)
{
	double a = ptr->alpha, b = ptr->beta;
	int lr = floor(log2(r)); //MAKE BETTER!
	if(lr < -2) return pow(r, 3-a)*(1/(3-a) + r*(a-b)/(4-a) + (a*a - a + b - 2*a*b + b*b)/(5-a)*r*r/2.0);
	if(lr >= INTEGRATION_SECTIONS - 2) return integ_profile(a, b, r); //MAKE BETTER
	return ptr->integs[lr + 2] + worst_thing_ever(lr, a, b, r);
}

double get_M()
{
	double r = gsl_rng_uniform(RNG);
	return M_prim*pow(pow(m_max/M_prim, 1.0-p)*r + pow(m_min/M_prim, 1.0-p)*(1.0-r), 1.0/(1.0-p));
}

double c_bar(double M)
{
	return pow(10.0, 1.0 - 0.1 * (log10(M) - 12));
}

void set_shape(halo *ptr, double M, double R)
{
	ptr->r_max = pow(3*M/(4 * pi * 200.0 * crit_density), 1.0/3.0);
	ptr->alpha = gsl_ran_gaussian_ziggurat(RNG, 0.2) + 1.25; //suggested by J. Ostriker
	ptr->beta = 3; //probalby need to change
	ptr->c = gsl_ran_lognormal(RNG, log(c_bar(M)), 0.25); //Ludlow el. al. (2013) and  Frank van den Bosch
	ptr->r_core = ptr->r_max/ptr->c;
	int i;
	double r = 1.0/4.0, a = ptr->alpha, b = ptr->beta;
	ptr->integs[0] = pow(r, 3-a)*(1/(3-a) + r*(a-b)/(4-a) + (a*a - a + b - 2*a*b + b*b)/(5-a)*r*r/2.0);
	r *= 2;
	for(i = 0; i < INTEGRATION_SECTIONS - 1; i++)
	{
		ptr->integs[i + 1] = ptr->integs[i]  + worst_thing_ever(i - 2, ptr->alpha, ptr->beta, r);
		r *= 2;
	}
	ptr->normalization = sec_integ_profile(ptr, ptr->c);

}

void set_velocity(halo *ptr, double R)
{
	double v_sigma = sqrt(1.0/3.0)*sqrt(-PhiFreeNFW(R));

	ptr->v.x = gsl_ran_gaussian_ziggurat(RNG, v_sigma);
	ptr->v.y = gsl_ran_gaussian_ziggurat(RNG, v_sigma);
	ptr->v.z = gsl_ran_gaussian_ziggurat(RNG, v_sigma);
}

double enclosed_mass(halo *ptr, double r)
{
	double x = r/(ptr->r_core);
	if(x > ptr->c)
		return ptr->M;
	else
	{
	/*	double sec = sec_integ_profile(ptr, x), integ = integ_profile(ptr->alpha, ptr->beta, x);
		double lerr = log(fabs(sec - integ));
		if(lerr > -4)
		{
			printf("alpha: %f beta: %f r: %f \n", ptr->alpha, ptr->beta, x);
			printf("lerr: %f sec: %f real: %f r: \n", lerr, sec, integ);
		}*/
		return ptr->M * sec_integ_profile(ptr, x)/ptr->normalization;
	}
}


void truncate(halo *ptr, double R)
{
	double l2 = R*R*perp_mag_sq(&(ptr->v), &(ptr->position));
	double r0 = l2/(M_prim * G);
	double E = 0.5*mag_sq(ptr->v) + PhiFreeNFW(R);
	double ecc =  sqrt(1.0 + 2.0*E*l2/(M_prim * G * M_prim * G));
	double R_min = min(fabs(r0/(1.0 + ecc)), fabs(r0/(1.0 - ecc)));

	/*printf("velocity: ");
	print_vector(ptr->v);
	printf("position ");
	print_vector(halo_pos(ptr));
	printf("R = %.3f theta = %.3f phi = %.3f\n l2 = %.3f e = %.3f \n", ptr->R, ptr->theta, ptr->phi, l2, ecc);*/

	if(E > 0 || R_min < 0.1)
	{
		ptr->M = 0;
	}
	else
	{
		double Rt = TidalRadius(ptr->M, R_min);
		double R_max = min(ptr->r_max, Rt);
		double new_M = enclosed_mass(ptr, R_max);

		ptr->r_max = R_max;
		ptr->c = R_max/ptr->r_core;
		ptr->normalization *= (new_M/ptr->M);
		ptr->M = new_M;
	}
}

halo *make_halo(halo *ptr)
{
	ptr->R = newton(gsl_rng_uniform(RNG)) * R_core_prim;
	ptr->cos_theta = 2*gsl_rng_uniform(RNG) - 1;
	ptr->theta = acos(ptr->cos_theta);
	ptr->phi = 2*pi*gsl_rng_uniform(RNG);
	assign_vec(ptr->position, ptr->R * sin(ptr->theta) * cos(ptr->phi), ptr->R * sin(ptr->theta) * sin(ptr->phi), ptr->R * ptr->cos_theta);

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

double Fluc(halo *halos, int num_halos, double D)
{
	double sum = 0;
	int i;
	vector my_pos = {0, 0, D}, diff;
	for(i = 0; i < num_halos; i++)
	{
		double R = halos[i].R;
		double r = sqrt(R*R + D*D - 2.0*R*D*halos[i].cos_theta);

		assign_vec_diff(diff, halos[i].position, my_pos);
		make_unit(diff)
		double v_r = dot_macro(halos[i].v, diff);

		double produced_fluc = ((r > CUTOFF_SCALE) ? (enclosed_mass(halos + i, r) * G /(r*r) * v_r) : 0);
		sum += produced_fluc; //KILLEM
		// kill heating which is adiabtic
		/*double natural_fluc = 2 * pi* pow(D, 3.0/2.0)/sqrt(MFreeNFW(D) * G) * 1/(PhiFreeNFW(D));
		if(produced_fluc/natural_fluc > 1)
			sum += produced_fluc;
		printf("%.3f\n", D);
		printf("velocity: ");
		print_vector(halos[i]->v);
		printf("position ");
		print_vector(halo_pos(halos[i]));
		printf("R = %.3f theta = %.3f phi = %.3f\n v_r = %.3f \n", halos[i]->R, halos[i]->theta, halos[i]->phi, v_r);*/
	}
	return sum*sum;
}

double H_Density(halo *halos, int num_halos, double D, double dD)
{
	double sum = 0;
	int i;
	for(i = 0; i < num_halos; i++)
	{
		if(halos[i].R < D && halos[i].R > D - dD)
			sum += halos[i].M;
	}
	return sum/(4.0*pi*D*D*dD);
}


int print_out = 0;

void init(int argc, char **argv)
{
	const gsl_rng_type *T;
	gsl_rng_env_setup();
	T = gsl_rng_default;
	RNG = gsl_rng_alloc (T);
	w = gsl_integration_workspace_alloc (1000);

	if(argc >= 2)
		num_trials = (int)pow(10, atoi(argv[1]));
	else
		num_trials = (int)pow(10, default_log_num_trials);

	if(argc > 2)
		print_out = 1;

	R_max_prim = MaxRadius(M_prim);
	c_prim = c_bar(M_prim);
	R_core_prim = R_max_prim/c_prim;
	m_max = f*M_prim;
	m_min = g*M_prim;
	expected_num = (2.0-p)/(1.0-p)*f*(pow(m_max/M_prim, 1.0-p) - pow(m_min/M_prim, 1.0-p))
		/ (pow(m_max/M_prim, 2.0-p) - pow(m_min/M_prim, 2.0-p));
}

double std_err_mean(double sum_of_squares)
{
	return sqrt(sum_of_squares/((double) num_trials*(num_trials - 1)));
}

void sq_root_data(data_cell *data, int num)
{
	int i;
	for(i = 0; i < num; i++)
	{
		data[i].m = sqrt(data[i].m);
		data[i].s *= 1.0/(2.0*data[i].m);
		data[i].max_val = sqrt(data[i].max_val);
	}
}

void print_to_file(char *name, double *Ds, data_cell *data)
{
	int j;
	char path[100], filename[50], snum[10], addendum[10];
	strcpy(path, "data_files/");
	strcpy(filename, name);
	strcpy(addendum, ".txt");
	snprintf(snum, 10, "%d_%d", (int)gsl_rng_default_seed, (int)log10(num_trials));
	strcat(path, filename);
	strcat(path, snum);
	strcat(path, addendum);

	FILE *f = NULL;

	if(!print_out)
		f = fopen(path, "w");

	if(!print_out && f != NULL)
	{
		for(j = 0; j < num_points; j++)
		{
			double sig = std_err_mean(data[j].s)/data[j].m/log(10);
			printf("%f, %f, %f, %fs,\n", data[j].m, data[j].s, sig, log10(data[j].m));
			if(data[j].m > 0)
				fprintf(f, "%f %f %f %f\n", log10(Ds[j]), log10(data[j].m), sig, log10(data[j].max_val));
		}
		fclose(f);
	}
	else
	{
		for(j = 0; j < num_points; j++)
		{
			double sig = std_err_mean(data[j].s)/data[j].m/log(10);
			printf("%f, %f, %f, %f,\n", data[j].m, data[j].s, sig, log10(data[j].m));
			if(data[j].m > 0)
				printf("%f : %f +/- %f\n", log10(Ds[j]), log10(data[j].m), sig);
		}
	}
}

int main(int argc, char **argv)
{
	init(argc, argv);
	unsigned int max_allowed_halos = (int)(10*expected_num);

	double Ds[num_points];
	data_cell Flucs[num_points], Dens[num_points];
	halo *halolist = malloc(max_allowed_halos*sizeof(halo));

	double mass = 0, avg_mass = 0;
	int i,j;

	for(i = 0; i < num_points; i++)
	{
		Ds[i] = pow(10, 5.0*i/(double) num_points);
		reset_cell(&Flucs[i]);
		reset_cell(&Dens[i]);
	}

	for(i = 0; i < num_trials; i++)
	{
		if(i % 100 == 0)printf("%d out of %d \n", i/100, num_trials/100);
		unsigned int num_halos = gsl_ran_poisson(RNG, expected_num);
		if(num_halos > max_allowed_halos)
		{
			max_allowed_halos *= 2;
			free(halolist);
			halolist = malloc(max_allowed_halos*sizeof(halo));
			printf("ALLOCATING\n");
		}
		mass = 0;
		//printf("Expect: %f   Have: %lu \n", expected_num, num_halos);

		for(j = 0; j < num_halos; j++)
		{
			make_halo(halolist + j);
			//print_halo_basic(halolist + j);
			mass += halolist[j].M;

			if(mass != mass)
			{
				print_halo_basic(halolist + j);
				printf("%f %f %d %f\n", mass, halolist[j].M, j, halolist[j].alpha);
				return -1;
			}
		}

		for(j = 0; j < num_points; j++)
		{
			update_cell(Flucs + j, Fluc(halolist, num_halos, Ds[j]), i + 1);
			update_cell(Dens + j, (j == 0 ? H_Density(halolist, num_halos, Ds[j], Ds[j]) : H_Density(halolist, num_halos, Ds[j], Ds[j] - Ds[j-1])), i + 1);
		}
		//printf("Mass frac: %f \n", mass/M_prim);
		avg_mass += mass/num_trials;

		/*for(j = 0; j < num_points; j++)
		{
			//printf("%f : %f\n", pow(10, 5.0*j/num_points), log10(Flucs[j]));
			//printf("%f : %f\n", pow(10, 5.0*j/num_points), log10(Dens[j]));
			//printf("%f : %f\n", pow(10, 5.0*j/num_points), Hist[j]);
		}*/

	}
	print_to_file("Density", Ds, Dens);
	sq_root_data(Flucs, num_points);
	print_to_file("Flucs", Ds, Flucs);

	printf("Mass frac: %f \n", avg_mass/M_prim);
	gsl_rng_free (RNG);
	gsl_integration_workspace_free (w);
	return 0;
}
