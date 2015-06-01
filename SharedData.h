/*
 * SharedData.h
 *
 *  Created on: Feb 15, 2015
 *      Author: viktor
 */

#ifndef SHAREDDATA_H_
#define SHAREDDATA_H_

#include "DataTypes.h"

#include <flann/flann.hpp>

using namespace std;
using namespace MonteCarlo;

template< int dim >
class allSolutions
{
	public:
		allSolutions() {};
		virtual ~allSolutions() {};
		static void add( std::vector<double> parameter, solution_type<dim> solution, int sample = 0 )
		{
			num_of_points++;
			solutions.push_back(solution);
			parameters.insert( parameters.end(), parameter.begin(), parameter.end() );


			// build k-d tree after we have generated 100 points
			if ( num_of_points >= threashold_num_of_points && num_of_points >= 1.2 * indexed_num_of_points )
			{
				dim_of_points = parameter.size();
				build_kd_tree();
				indexed_num_of_points = num_of_points;

				std::cerr << "Build k-d tree for " << num_of_points << std::endl;
			}

//			if ( num_of_points < threashold_num_of_points )
//				parameters.insert( parameters.end(), parameter.begin(), parameter.end() );
//			else if ( num_of_points == threashold_num_of_points )
//			{
//				dim_of_points = parameter.size();
//				build_kd_tree();
//				parameters.clear();
//			}
//			else // num_of_points > 100
//			{
//				flann::Matrix<double> point;
//				point = flann::Matrix<double>(&parameter[0], 1, parameter.size() );
//				index.addPoints(point, 1.3);
//			}
		};
		static void build_kd_tree()
		{
			if ( parameters.size() != 0 )
				dataset = flann::Matrix<double>( &parameters[0], num_of_points, dim_of_points );
			else
				std::cerr << "ERROR: No data to build kd-tree" << std::endl;

			// construct an randomized kd-tree index using 4 kd-trees
			index = flann::Index<flann::L2<double>> (dataset, flann::KDTreeIndexParams(4));
			index.buildIndex();
		};
		static void find_closest( std::vector<double> &parameter, solution_type<dim> &solution )
		{
			std::vector<std::vector<int>>	 indices;
			std::vector<std::vector<double>> dists;

			int nn = 1;

			if ( num_of_points > threashold_num_of_points )
			{
				// do a knn search, using 128 checks
				query = flann::Matrix<double>(&parameter[0], 1, parameter.size() );
				index.knnSearch( query, indices, dists, nn, flann::SearchParams(128));

				solution.vector = 0.0;

				// NN estimate
				for ( int i = 0; i < nn; i++)
					solution.add( solutions[indices[0][i]], 1.0/nn );

				// Shepard estimate
//				double sum = 0.0;
//				for ( int i = 0; i < nn; i++)
//					sum += 1.0 / dists[0][i] / dists[0][i];
//				for ( int i = 0; i < nn; i++)
//					solution.add( solutions[indices[0][i]], 1.0 / dists[0][i] / dists[0][i] / sum );
			}
			else
				solution.vector = 0.0;
		};
		static void find_closest_test( std::vector<double> &parameter )
		{
			std::vector< std::vector<int> > indices;
			std::vector<std::vector<double> > dists;

			query = flann::Matrix<double>(&parameter[0], 1, parameter.size() );

			int nn = 1;

			// do a knn search, using 128 checks
			index.knnSearch( query, indices, dists, nn, flann::SearchParams(128));
			std::cout << indices[0][0] << std::endl;
		};
	private:
		static std::vector<double>	parameters;
		static int threashold_num_of_points;
		static int num_of_points, indexed_num_of_points;
		static int dim_of_points;
		static std::vector<solution_type<dim>>	solutions;
		static flann::Matrix<double> dataset;
		static flann::Matrix<double>  query;
		static flann::Index< flann::L2<double> > index;
};

template<int dim>	std::vector<double>					allSolutions<dim>::parameters;
template<int dim>	int 								allSolutions<dim>::threashold_num_of_points = 10;
template<int dim>	int 								allSolutions<dim>::num_of_points;
template<int dim>	int 								allSolutions<dim>::indexed_num_of_points;
template<int dim>	int 								allSolutions<dim>::dim_of_points;
template<int dim>	std::vector<solution_type<dim>>		allSolutions<dim>::solutions;
template<int dim>	flann::Matrix<double> 				allSolutions<dim>::dataset;
template<int dim>	flann::Matrix<double> 				allSolutions<dim>::query;
template<int dim>	flann::Index< flann::L2<double> >	allSolutions<dim>::index(flann::KDTreeIndexParams(4));



//template< int dim >
//class SharedData
//{
//	public:
//		SharedData() {};
//		virtual ~SharedData() {};
//		static std::vector<system_type<dim>>	system;		// TODO: system_type consists of system & rhs; in parallel code system cannot be static
//		static
//};



#endif /* SHAREDDATA_H_ */
