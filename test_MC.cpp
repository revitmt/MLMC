/*
 * test_Problem.cpp
 *
 *  Created on: May 14, 2015
 *      Author: viktor
 */


#include <iostream>

#include "DiscretizationData.h"
#include "InputData.h"
#include "Stokes.h"
#include "Navier_Stokes.h"
#include "MonteCarlo.h"
#include "SharedData.h"
#include "Output.h"


using namespace MonteCarlo;
using namespace std;



int main(int argc, char **argv)
{
	dealii::deallog.depth_console (2);


	const int dim = 2;

	int n_of_levels = 1;
	int init_level = 0;
	int level = 0;
	double error = 1e-6;
	int samples = 1000000;

	// generate discretization data
	DiscretizationData<dim>::generate( n_of_levels, init_level );


	// initialize input functions
	Coefficient<dim>	coefficient_function;
	BoundaryValues<dim>	BV_function;
	RightHandSide<dim>	rhs_function;

	// initialize the problems
//	MonteCarloSolver<dim> problem_1( level, error, samples, MonteCarloSolver<dim>::test );
	MonteCarloSolver<dim> problem_2( level, error, samples, MonteCarloSolver<dim>::base );

	// initialize solution vectors
	solution_type<dim>	solution(level);
	solution_type<dim>	variance(level);

	Output<dim> to_file;


//	std::ofstream myfile("spars_pattern.txt");
//	DiscretizationData<dim>::sparsity_patterns_ptr[1]->print_gnuplot(myfile);

	// initialize random field for boundary value function
	double cor_length = 14.0 / sqrt(6.0);
	BV_function.initialize( cor_length );


	// MC solution
//	problem_1.run( solution, variance, 0 );
	problem_2.run( solution, variance, 0 );


	to_file.print( solution, "MC_Navier_Stokes_solution" );
	to_file.print( variance, "MC_variance_Navier_Stokes_solution" );

//	cerr << problem_2.info.iterations << endl;
//	cerr << problem_3.info.iterations << endl;

	return 0;
}
