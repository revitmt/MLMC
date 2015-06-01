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

	// generate discretization data
	DiscretizationData<dim>::generate( n_of_levels, init_level );

	// initialize input functions
	Coefficient<dim>	coefficient_function;
	BoundaryValues<dim>	BV_function;
	RightHandSide<dim>	rhs_function;

	// initialize the problems
	Stokes<dim>			Stokes_problem( level,   error );
	Navier_Stokes<dim>	problem_0( level,   error );
	Navier_Stokes<dim>	modified_problem( level,   error );

	// initialize solution vectors
	solution_type<dim>	Stokes_solution(level);
	solution_type<dim>	solution_0(level);
	solution_type<dim>	modified_solution(level);

	Output<dim> to_file;


	// initialize random field for boundary value function
	double cor_length = 14.0 / sqrt(6.0);
//	double cor_length = 100;
	BV_function.initialize( cor_length );

	// vector of parameters
	std::vector<double> params( BV_function.get_stoch_dim() );
	RandomGenerator random_generator( Normal );


	// generate boundary value function
	random_generator.generate( params );
	BV_function.set_random_vector( params );

	std::vector<double> coefficients;
	coefficients = BV_function.get_coefficients();

//	for (int i=0; i<BV_function.get_stoch_dim(); i++)
//		cerr << coefficients[i] << " " << endl;

	// Stokes solution
	Stokes_problem.run( coefficient_function, rhs_function, BV_function, Stokes_solution );

	// Navier-Stokes solution
	problem_0.run( coefficient_function, rhs_function, BV_function, solution_0 );  cerr << problem_0.info.init_residual << " " << problem_0.info.total_iterations << endl;
	problem_0.run( coefficient_function, rhs_function, BV_function, solution_0 );  cerr << problem_0.info.init_residual << " " << problem_0.info.total_iterations << endl;

	// modify params
	random_generator.generate( params );
//	for ( int i = 0; i < BV_function.get_stoch_dim(); i++ )
//		params[i] = params[i] + BV_function.get_random_vector()[i];
	BV_function.set_random_vector( params );
	modified_solution.vector = solution_0.vector;
	modified_problem.run( coefficient_function, rhs_function, BV_function, modified_solution );


	cerr << modified_problem.info.init_residual << " "  << modified_problem.info.total_iterations << endl;


	to_file.print( Stokes_solution, "Stokes_solution" );
	to_file.print( solution_0, "Navier_Stokes_solution" );
	to_file.print( modified_solution, "Navier_Stokes_solution_modified" );


	return 0;
}

