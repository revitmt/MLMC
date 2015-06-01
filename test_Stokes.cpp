/*
 * test_Problem.cpp
 *
 *  Created on: May 14, 2015
 *      Author: viktor
 */


#include <iostream>

#include "DiscretizationData.h"
#include "InputData.h"
//#include "Problem.h"
#include "Stokes.h"
#include "Output.h"


using namespace MonteCarlo;
using namespace std;



int main(int argc, char **argv)
{
	const int dim = 2;

	int init_level = 2;

	DiscretizationData<dim>::add_next_level( init_level );
	DiscretizationData<dim>::add_next_level( init_level );
	DiscretizationData<dim>::add_next_level( init_level );
	DiscretizationData<dim>::add_next_level( init_level );
	DiscretizationData<dim>::add_next_level( init_level );

	DiscretizationData<dim>::finalize( init_level );

	Coefficient<dim>	coefficient_function;
	BoundaryValues<dim>	BV_function;
	RightHandSide<dim>	rhs_function;
	Stokes<dim>			problem_1( 0, 1e-6 );
	Stokes<dim>			problem_2( 4, 1e-6 );
	Stokes<dim>			problem_3( 4, 1e-6 );

	solution_type<dim>	solution_1(0);
	solution_type<dim>	solution_2(4);
	solution_type<dim>	solution_3(4);
	solution_type<dim>	solution_4(4);

	Output<dim> to_file;

	problem_1.run( coefficient_function, rhs_function, BV_function, solution_1 );
	problem_2.run( coefficient_function, rhs_function, BV_function, solution_2 );
	problem_3.run( problem_2.operator_matrix, problem_2.operator_rhs, problem_2.constraints, solution_3 );

	solution_4.interpolate_from(solution_1);

	to_file.print( solution_1, "Stokes_solution_1" );
	to_file.print( solution_2, "Stokes_solution_2" );
	to_file.print( solution_3, "Stokes_solution_3" );
	to_file.print( solution_4, "Stokes_solution_4" );


	cout << solution_1.estimate_posterior_error(0) << endl;
	cout << solution_1.estimate_posterior_error(1) << endl;
	cout << solution_1.estimate_posterior_error(2) << endl;

	return 0;
}
