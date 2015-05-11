/*
 * Project.cpp
 *
 *  Created on: Dec 18, 2014
 *      Author: viktor
 */

#include "MLMC.h"
//#include "MonteCarlo.h"

#include <iostream>
#include <lapacke.h>


using namespace MonteCarlo;
using namespace std;

int main(int argc, char **argv)
{
	const int dim = 2;
	double eps = 0.0005; //[ 0.01 0.005 0.001 0.0005 0.0001 ]
	double cor_length = 0.05;


	Output<2> toFile;
//	Coefficient<dim> mycoef;
//	RightHandSide<dim> mycoef;

//	DiscretizationData<dim>::add_next_level();
//	DiscretizationData<dim>::add_next_level();
//	DiscretizationData<dim>::add_next_level();
//	DiscretizationData<dim>::add_next_level();
//	mycoef.generate();
//	const int level = 3;
//	toFile.print_exact_function( level, mycoef, "coefficient");

//	DiscretizationData<dim>::generate( 1, 11 );
//	DiscretizationData<dim>::generate( 1, 11 );

//	std::cout << DiscretizationData<dim>::triangulations_ptr[0]->memory_consumption() / 1024.0 / 1024.0 / 1024.0 << std::endl;
//	std::cout << DiscretizationData<dim>::dof_handlers_ptr[0]->memory_consumption() / 1024.0 / 1024.0 / 1024.0 << std::endl;
//	std::cout << DiscretizationData<dim>::sparsity_patterns_ptr[0]->memory_consumption() / 1024.0 / 1024.0 / 1024.0 << std::endl;
//	std::cout << DiscretizationData<dim>::hanging_nodes_constraints_ptr[0]->memory_consumption() / 1024.0 / 1024.0 / 1024.0 << std::endl;
//	std::cout << DiscretizationData<dim>::intergrid_maps_ptr[0][0]->memory_consumption() / 1024.0 / 1024.0 / 1024.0 << std::endl;
//	std::cout << DiscretizationData<dim>::fe->memory_consumption() / 1024.0 / 1024.0 / 1024.0 << std::endl;
//	std::cout << DiscretizationData<dim>::quadrature_formula->memory_consumption() / 1024.0 / 1024.0 / 1024.0 << std::endl;
//	std::cout << MemoryConsumption::memory_consumption(*DiscretizationData<dim>::triangulations_ptr[0]) / 1024.0 / 1024.0 / 1024.0  << std::endl;

//	int input;
//	std::cout << "Hello: ";
//	std::cin >> input;

//	DiscretizationData<dim>::generate(6);
//	int level = 5;
//
//	Coefficient<dim>	coefficient_function_1;
//	coefficient_function_1.generate();
//	solution_type<dim> given_function(level);
//	VectorTools::interpolate ( *(DiscretizationData<dim>::dof_handlers_ptr[level]), coefficient_function_1, given_function.vector );
//	toFile.print( given_function, "function_" + std::to_string(0) );
//
//	std::vector<double> param_1 = coefficient_function_1.get_random_vector();
//
//	double min_dist_1 = 10000.0;
//	double min_dist_2 = 10000.0;
//	int min_index_1 = 0;
//	int min_index_2 = 0;
//	for ( int i = 1; i < 5; i++ )
//	{
//		Coefficient<dim>	coefficient_function_2;
//		coefficient_function_2.generate();
//		std::vector<double> param_2 = coefficient_function_2.get_random_vector();
//		solution_type<dim> closest_function(level);
//		VectorTools::interpolate ( *(DiscretizationData<dim>::dof_handlers_ptr[level]), coefficient_function_2, closest_function.vector );
//
//		toFile.print( closest_function, "function_" + std::to_string(i) );
//
//		closest_function.subtract(given_function);
//		double dist_1 = closest_function.Linfty_norm();
//
//		double dist_2 = 0.0;
//		for ( unsigned int j = 0; j < param_2.size(); j++ )
//			dist_2 += ( param_2[j] - param_1[j] ) * ( param_2[j] - param_1[j] ) ;
//		dist_2 = sqrt(dist_2);
//
//		min_index_1  = ( dist_1 < min_dist_1) ? i : min_index_1;
//		min_dist_1 = ( dist_1 < min_dist_1) ? dist_1 : min_dist_1;
//
//		min_index_2  = ( dist_2 < min_dist_2) ? i : min_index_2;
//		min_dist_2 = ( dist_2 < min_dist_2) ? dist_2 : min_dist_2;
//
//		std::cout << i << " " << dist_1 << " " << dist_2 << std::endl;
//	}
//
//	std::cout << std::endl;
//	std::cout << min_index_1 << " " << min_dist_1 << std::endl;
//	std::cout << min_index_2 << " " << min_dist_2 << std::endl;


	MLMC<dim> MLMC_problem( eps, cor_length );
	MLMC_problem.run();



	return 0;
}

