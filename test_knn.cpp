/*
 * test_Problem.cpp
 *
 *  Created on: May 14, 2015
 *      Author: viktor
 */


#include <iostream>

#include "DiscretizationData.h"
#include "DataTypes.h"
#include "SharedData.h"
#include "RandomGenerator.h"


using namespace MonteCarlo;
using namespace std;



int main(int argc, char **argv)
{
	dealii::deallog.depth_console (2);

	const int dim = 2;

	int n_of_levels = 1;
	int init_level = 0;

	int dim_of_param = 15;

	// generate discretization data
	DiscretizationData<dim>::generate( n_of_levels, init_level );


	int num_of_points = 85846;

	std::vector<double> params( dim_of_param );
	std::vector<double> query( dim_of_param );
	solution_type<dim>  result(0);
	std::vector<std::vector<double>> all_params( num_of_points );
	RandomGenerator random_generator( Normal );


	for (int i = 0; i < dim_of_param; i++)
		random_generator.generate( query[i] );


	for ( int i = 0; i < num_of_points; i++ )
	{
		random_generator.generate( params );
		for (int j = 0; j < dim_of_param; j++)
			result.vector[j] = params[j];
		allSolutions<dim>::add( params, result );
		all_params[i] = params;
	}
	allSolutions<dim>::find_closest( query, result );

	double dist_knn = 0;
	for ( int j = 0; j < dim_of_param; j++ )
		dist_knn += (result.vector[j] - query[j])*(result.vector[j] - query[j]);
	dist_knn = sqrt(dist_knn);


	double min = 1.0e5;
	double dist;
	int index;
	for ( int i = 0; i < num_of_points; i++ )
	{
		dist = 0;
		for ( int j = 0; j < dim_of_param; j++ )
			dist += (all_params[i][j] - query[j])*(all_params[i][j] - query[j]);
		dist = sqrt(dist);

		if ( dist < min )
		{
			min = dist;
			index = i;
		}
	}

	cerr << min << " " << dist_knn << endl;
	for (int i = 0; i < dim_of_param; i++)
		cerr << result.vector[i] << " ";
	cerr << endl;
	for (int i = 0; i < dim_of_param; i++)
		cerr << all_params[index][i] << " ";
	cerr << endl;



//	cerr << problem_2.info.iterations << endl;
//	cerr << problem_3.info.iterations << endl;

	return 0;
}
