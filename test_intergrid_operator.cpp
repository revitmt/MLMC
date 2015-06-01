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

	int n_of_levels = 5;
	int init_level = 0;
	int level = 0;
	double error = 1e-5;

	// generate discretization data
	DiscretizationData<dim>::generate( n_of_levels, init_level );

	// initialize input functions
	Coefficient<dim>	coefficient_function;
	BoundaryValues<dim>	BV_function;
	RightHandSide<dim>	rhs_function;

	// initialize the problems
	Stokes<dim>			Stokes_problem( level,   error );
	Navier_Stokes<dim>	problem_0( level,   error );
	Navier_Stokes<dim>	problem_1( level+1, error );
	Navier_Stokes<dim>	problem_2( level+2, error );
	Navier_Stokes<dim>	problem_3( level+3, error );
	Navier_Stokes<dim>	problem_4( level+4, error );
	Navier_Stokes<dim>	modified_problem( level,   error );

	// initialize solution vectors
	solution_type<dim>	Stokes_solution(level);
	solution_type<dim>	solution_0(level);
	solution_type<dim>	solution_1(level+1);
	solution_type<dim>	solution_2(level+2);
	solution_type<dim>	solution_3(level+3);
	solution_type<dim>	solution_4(level+4);
	solution_type<dim>	modified_solution(level);

	Output<dim> to_file;


	// initialize random field for boundary value function
	double cor_length = 14.0 / sqrt(6.0);
	BV_function.initialize( cor_length );

	// vector of parameters
	std::vector<double> params( BV_function.get_stoch_dim() );
	RandomGenerator random_generator( Normal );


	// generate boundary value function
	random_generator.generate( params );
	BV_function.set_random_vector( params );

	// Stokes solution
	Stokes_problem.run( coefficient_function, rhs_function, BV_function, Stokes_solution );

	// Navier-Stokes solution
	solution_0.vector = 0.0;					problem_0.run( coefficient_function, rhs_function, BV_function, solution_0 );
	solution_1.interpolate_from( solution_0 );  problem_1.run( coefficient_function, rhs_function, BV_function, solution_1 );
	solution_2.interpolate_from( solution_0 );  problem_2.run( coefficient_function, rhs_function, BV_function, solution_2 );
	solution_3.interpolate_from( solution_0 );  problem_3.run( coefficient_function, rhs_function, BV_function, solution_3 );
	solution_4.interpolate_from( solution_0 );  problem_4.run( coefficient_function, rhs_function, BV_function, solution_4 );
	cerr << "0->0 " << problem_0.info.iterations << endl;
	cerr << "0->1 " << problem_1.info.iterations << endl;
	cerr << "0->2 " << problem_2.info.iterations << endl;
	cerr << "0->3 " << problem_3.info.iterations << endl;
	cerr << "0->4 " << problem_4.info.iterations << endl;

	solution_0.interpolate_from( solution_1 );  problem_0.run( coefficient_function, rhs_function, BV_function, solution_0 );
	solution_1.vector = 0.0;  					problem_1.run( coefficient_function, rhs_function, BV_function, solution_1 );
	solution_2.interpolate_from( solution_1 );  problem_2.run( coefficient_function, rhs_function, BV_function, solution_2 );
	solution_3.interpolate_from( solution_1 );  problem_3.run( coefficient_function, rhs_function, BV_function, solution_3 );
	solution_4.interpolate_from( solution_1 );  problem_4.run( coefficient_function, rhs_function, BV_function, solution_4 );
	cerr << "1->0 " << problem_0.info.iterations << endl;
	cerr << "1->1 " << problem_1.info.iterations << endl;
	cerr << "1->2 " << problem_2.info.iterations << endl;
	cerr << "1->3 " << problem_3.info.iterations << endl;
	cerr << "1->4 " << problem_4.info.iterations << endl;

	solution_0.interpolate_from( solution_2 );  problem_0.run( coefficient_function, rhs_function, BV_function, solution_0 );
	solution_1.interpolate_from( solution_2 );  problem_1.run( coefficient_function, rhs_function, BV_function, solution_1 );
	solution_2.vector = 0.0;					problem_2.run( coefficient_function, rhs_function, BV_function, solution_2 );
	solution_3.interpolate_from( solution_2 );  problem_3.run( coefficient_function, rhs_function, BV_function, solution_3 );
	solution_4.interpolate_from( solution_2 );  problem_4.run( coefficient_function, rhs_function, BV_function, solution_4 );
	cerr << "2->0 " << problem_0.info.iterations << endl;
	cerr << "2->1 " << problem_1.info.iterations << endl;
	cerr << "2->2 " << problem_2.info.iterations << endl;
	cerr << "2->3 " << problem_3.info.iterations << endl;
	cerr << "2->4 " << problem_4.info.iterations << endl;

	solution_0.interpolate_from( solution_3 );  problem_0.run( coefficient_function, rhs_function, BV_function, solution_0 );
	solution_1.interpolate_from( solution_3 );  problem_1.run( coefficient_function, rhs_function, BV_function, solution_1 );
	solution_2.interpolate_from( solution_3 );  problem_2.run( coefficient_function, rhs_function, BV_function, solution_2 );
	solution_3.vector = 0.0;					problem_3.run( coefficient_function, rhs_function, BV_function, solution_3 );
	solution_4.interpolate_from( solution_3 );  problem_4.run( coefficient_function, rhs_function, BV_function, solution_4 );
	cerr << "3->0 " << problem_0.info.iterations << endl;
	cerr << "3->1 " << problem_1.info.iterations << endl;
	cerr << "3->2 " << problem_2.info.iterations << endl;
	cerr << "3->3 " << problem_3.info.iterations << endl;
	cerr << "3->4 " << problem_4.info.iterations << endl;

	solution_0.interpolate_from( solution_4 );  problem_0.run( coefficient_function, rhs_function, BV_function, solution_0 );
	solution_1.interpolate_from( solution_4 );  problem_1.run( coefficient_function, rhs_function, BV_function, solution_1 );
	solution_2.interpolate_from( solution_4 );  problem_2.run( coefficient_function, rhs_function, BV_function, solution_2 );
	solution_3.interpolate_from( solution_4 );  problem_3.run( coefficient_function, rhs_function, BV_function, solution_3 );
	solution_4.vector = 0.0;					problem_4.run( coefficient_function, rhs_function, BV_function, solution_4 );
	cerr << "4->0 " << problem_0.info.iterations << endl;
	cerr << "4->1 " << problem_1.info.iterations << endl;
	cerr << "4->2 " << problem_2.info.iterations << endl;
	cerr << "4->3 " << problem_3.info.iterations << endl;
	cerr << "4->4 " << problem_4.info.iterations << endl;


	// modify params
	for ( int i = 0; i < BV_function.get_stoch_dim(); i++ )
		params[i] = params[i] + 0.01*BV_function.get_random_vector()[i];
	BV_function.set_random_vector( params );
	modified_problem.run( problem_0.Stokes_operator_matrix, problem_0.Stokes_operator_rhs, BV_function, modified_solution );


	to_file.print( solution_1, "Stokes_solution" );
	to_file.print( solution_2, "Navier_Stokes_solution" );
	to_file.print( solution_3, "Navier_Stokes_solution_interpolated_1" );
	to_file.print( solution_4, "Navier_Stokes_solution_interpolated_2" );
	to_file.print( modified_solution, "Navier_Stokes_solution_modified" );


	return 0;
}

