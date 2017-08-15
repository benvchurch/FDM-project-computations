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

#define min(x,y) (x < y? x : y)
#define sign(x) (x >= 0? 1.0 : -1.0)
#define dot_macro(A, B) ((A).x * (B).x + (A).y * (B).y + (A).z * (B).z)
#define mag_sq(A) dot_macro(A, A)
#define mag(A) sqrt(dot_macro(A, A))
#define assign_vec(A, x_val, y_val, z_val) {(A).x = x_val; (A).y = y_val; (A).z = z_val;}

gsl_rng *RNG;
gsl_integration_workspace * w;

const int default_log_num_trials = 6;
int num_trials, num_points = 100;

const double pi = 3.14159265;
double crit_density = 1.3211775*0.0000001,
	f = 0.1,
	g = 0.0001,
	p = 1.9,
	G = 0.0045,
	k = 2,
	M_prim = 1000000000000,
	T_age = 10000;

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
	double m, s;
}data_cell;

void update_cell(data_cell *ptr, double new_data, int k)
{
	double old_m = ptr->m;
	ptr->m += (new_data - old_m)/k;
	ptr->s += (new_data - old_m)*(new_data - ptr->m);
}

void reset_cell(data_cell *ptr)
{
	ptr->m = 0;
	ptr->s = 0;
}

typedef struct
{
	double R, theta, phi;
	double M, r_core, r_max, alpha, beta, c, normalization;
	vector v, position;
} halo;

void print_vector(vector *A)
{
	printf("<%.3f, %.3f, %.3f>\n", A->x, A->y, A->z);
}

vector *add_vec(vector *res, vector *A, vector *B)
{
	assign_vec(*res, A->x + B->x, A->y + B->y, A->z + B->z);
	return res;
}

vector *sub_vec(vector *res, vector *A, vector *B)
{
	assign_vec(*res, A->x - B->x, A->y - B->y, A->z - B->z);
	return res;
}

double dot(vector *A,  vector *B)
{
	return dot_macro(*A, *B);
}

vector *scale(vector *res, double c, vector *A)
{
	assign_vec(*res, c*(A->x), c*(A->y), c*(A->z));
	return res;
}

vector *unit(vector *unit_vec, vector *diff, vector *A, vector *B)
{
	sub_vec(diff, B, A);
	return scale(unit_vec, 1/mag(*diff), diff);
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
    return x/pow(1.0 + x, 2.0) * 1.0/(log(1.0 + c_prim) - c_prim/(1.0 + c_prim));
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

double rho_profile(double x, void *params)
{
	double *param_ptr = (double *)params;
	return pow(x, 2 - param_ptr[0])*pow((1 + x), param_ptr[0] - param_ptr[1]);
}

double integ_profile(double alpha, double beta, double r)
{
	double result, error;
	double arr[2];
	arr[0] = alpha;
	arr[1] = beta;

  	gsl_function F;
  	F.function = &rho_profile;
  	F.params = arr;

  	gsl_integration_qags (&F, 0, r, 0, 1e-6, 1000, w, &result, &error);
	if(error/result > 1e-6)printf("LARGE INTEG ERROR WARNING %f for result %f", error, result);
	return result;
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
	ptr->normalization = integ_profile(ptr->alpha, ptr->beta, ptr->c);
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
		return ptr->M * integ_profile(ptr->alpha, ptr->beta, x)/ptr->normalization;
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

halo *make_halo()
{
	halo *ptr = (halo *) malloc(sizeof(halo));
	ptr->R = newton(gsl_rng_uniform(RNG)) * R_core_prim;
	ptr->theta = acos(2*gsl_rng_uniform(RNG) - 1);
	ptr->phi = 2*pi*gsl_rng_uniform(RNG);
	assign_vec(ptr->position, ptr->R*sin(ptr->theta)*cos(ptr->phi), ptr->R*sin(ptr->theta)*sin(ptr->phi), ptr->R*cos(ptr->theta));

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
	vector my_pos = {0, 0, D}, unit_vec, diff;
	for(i = 0; i < num_halos; i++)
	{
		double R = halos[i]->R;
		double r = sqrt(R*R + D*D - 2.0*R*D*cos(halos[i]->theta));
		double v_r = dot(&(halos[i]->v), unit(&unit_vec, &diff, &my_pos, &(halos[i]->position)));
		sum += enclosed_mass(halos[i], r) * G /(r*r) * v_r;
		/*printf("%.3f\n", D);
		printf("velocity: ");
		print_vector(halos[i]->v);
		printf("position ");
		print_vector(halo_pos(halos[i]));
		printf("R = %.3f theta = %.3f phi = %.3f\n v_r = %.3f \n", halos[i]->R, halos[i]->theta, halos[i]->phi, v_r);*/
	}
	return pow(sum, 2.0);
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
	return sqrt(sum_of_squares/(num_trials*(num_trials - 1)));
}

void sq_root_data(data_cell *data, int num)
{
	int i;
	for(i = 0; i < num; i++)
	{
		data[i].m = sqrt(data[i].m);
		data[i].s *= 1.0/(2.0*data[i].m);
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
			if(data[j].m != 0 && sig < 100)
				fprintf(f, "%f %f %f\n", log10(Ds[j]), log10(data[j].m), sig);
		}
		fclose(f);
	}
	else
	{
		for(j = 0; j < num_points; j++)
		{
			double sig = std_err_mean(data[j].s)/data[j].m/log(10);
			if(data[j].m != 0 && sig < 100)
				printf("%f : %f +/- %f\n", log10(Ds[j]), log10(data[j].m), sig);
		}
	}
}

int main(int argc, char **argv)
{
	init(argc, argv);
	double Ds[num_points];
	data_cell Flucs[num_points], Dens[num_points];
	double mass = 0, avg_mass = 0;
	int i,j;

	for(i = 0; i < num_points; i++)
	{
		Ds[i] = pow(10, 5.0*i/num_points);
		reset_cell(&Flucs[i]);
		reset_cell(&Dens[i]);
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
			update_cell(&Flucs[j], Fluc(halolist, num_halos, Ds[j]), j + 1);
			update_cell(&Dens[j], (j == 0 ? H_Density(halolist, num_halos, Ds[j], Ds[j]) : H_Density(halolist, num_halos, Ds[j], Ds[j] - Ds[j-1])), j + 1);
		}
		//printf("Mass frac: %f \n", mass/M_prim);
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
	print_to_file("Density", Ds, Dens);
	sq_root_data(Flucs, num_points);
	print_to_file("Flucs", Ds, Flucs);

	printf("Mass frac: %f \n", avg_mass/M_prim);
	gsl_rng_free (RNG);
	gsl_integration_workspace_free (w);
	return 0;
}
