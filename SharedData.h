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
		static void add( std::vector<double> parameter, solution_type<dim> solution, int sample )
		{
			parameters.insert( parameters.end(), parameter.begin(), parameter.end() );
			num_of_points++;
			dim_of_points = parameter.size();
			solutions.push_back(solution);
			solutions.back().sample = sample;

//			if ( parameters.size() == 10*parameter.size() )
//			{
//				build_kd_tree();
//			}
//			else if( parameters.size() > 10*parameter.size() )
//			{
//				flann::Matrix<double> point;
//				point = flann::Matrix<double>(&parameter[0], 1, parameter.size() );
//				index.addPoints(point);
//			}
		};
		static void build_kd_tree()
		{
			if ( parameters.size() != 0 )
			{
				double ppoints[][3] = { {5.0, 3.0, 8.0}, {1.0, 2.0, 3.0} };

				dataset = flann::Matrix<double>( &parameters[0], num_of_points, dim_of_points );
			}
			else
				std::cerr << "ERROR: No data to build kd-tree" << std::endl;

			// construct an randomized kd-tree index using 4 kd-trees
			index = flann::Index<flann::L2<double> > (dataset, flann::KDTreeIndexParams(4));
			index.buildIndex();
		};
		static void find_closest( std::vector<double> &parameter, solution_type<dim> &solution )
		{
			std::vector< std::vector<int> > indices;
			std::vector<std::vector<double> > dists;

			query = flann::Matrix<double>(&parameter[0], 1, parameter.size() );

			int nn = 1;

			// do a knn search, using 128 checks
			index.knnSearch( query, indices, dists, nn, flann::SearchParams(128));

			if ( parameters.size() != 0 )
			{
				solution.vector = 0.0;

				// NN estimate
				for ( int i = 0; i < nn; i++)
					solution.add( solutions[indices[0][i]], 1.0/nn );
				//solution.interpolate_from( solutions[indices[0][0]] );

				// Shepard estimate
//				double sum = 0.0;
//				for ( int i = 0; i < nn; i++)
//					sum += 1.0 / dists[0][i] / dists[0][i];
//				for ( int i = 0; i < nn; i++)
//					solution.add( solutions[indices[0][i]], 1.0 / dists[0][i] / dists[0][i] / sum );
			}
			else
				solution.vector = 0.0;

//			double min_dist = 100000.0;
//			int min_index = 0;
////
////				Coefficient<dim>	coefficient_function_1;
////				coefficient_function_1.set_random_vector( parameter );
////
////				solution_type<dim> given_function(solution.level);
////				solution_type<dim> closest_function(solution.level);
////				VectorTools::interpolate ( *(DiscretizationData<dim>::dof_handlers_ptr[solution.level]), coefficient_function_1, given_function.vector );
//
//			for ( int j = 0; j < parameters.size(); j++ )
//			{
//				double dist = 0.0;
//
////					Coefficient<dim>	coefficient_function_2;
////					coefficient_function_2.set_random_vector( parameters[j] );
////					VectorTools::interpolate ( *(DiscretizationData<dim>::dof_handlers_ptr[solution.level]), coefficient_function_2, closest_function.vector );
////					closest_function.subtract(given_function);
////
////					dist = closest_function.Linfty_norm();
//
//				for ( int i = 0; i < parameters[j].size(); i++ )
//				{
//					dist += ( parameters[j][i] - parameter[i] ) * ( parameters[j][i] - parameter[i] ) ;
//				}
//				dist = sqrt(dist);
//
//				if ( dist < min_dist ) //&& ( solution.level - solutions[min_index].level ) < 3 )
//				{
//					min_index = j;
//					min_dist = dist;
//				}
//			}
//
//			if ( parameters.size() != 0 )
//			{
//				solution.interpolate_from(solutions[min_index]);
////					std::fstream myfile("mylog.txt", ios::out | ios::app);
////					myfile << std::endl
////					       << std::setw(3) << solution.level << " "
////						   << std::setw(7) << solution.sample << " --> "
////						   << std::setw(3) << solutions[min_index].level << " "
////						   << std::setw(7) << solutions[min_index].sample << " "
////						   << std::fixed
////						   << std::setprecision(1) << std::setw(7) << min_dist;
//			}
//			else
//				solution.vector = 0.0;
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
		static int num_of_points;
		static int dim_of_points;
		static std::vector<solution_type<dim>>	solutions;
		static flann::Matrix<double> dataset;
		static flann::Matrix<double>  query;
		static flann::Index< flann::L2<double> > index;
};

template<int dim>	std::vector<double>					allSolutions<dim>::parameters;
template<int dim>	int 								allSolutions<dim>::num_of_points;
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
