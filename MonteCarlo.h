/*
 * MonteCarlo.h
 *
 *  Created on: Dec 13, 2014
 *      Author: viktor
 */

#ifndef MONTECARLO_H_
#define MONTECARLO_H_

//#include "DiscretizationData.h"
#include "DataTypes.h"
#include "Stokes.h"
#include "Navier_Stokes.h"
#include "Output.h"
#include "statistics.h"
#include <vector>
#include <numeric>					// std::accumulate
#include <string>					// std::string, std::to_string

#include "my_timer.h"

#include <fstream>
#include <iostream>

using namespace dealii;



namespace MonteCarlo
{

	struct MC_info
	{
		int zero_guess_minIter = 1000000;
		int zero_guess_maxIter = 0;
		double zero_guess_meanIter = 0.0;
		int zero_guess_sumIter = 0;

		int accelerated_minIter = 1000000;
		int accelerated_maxIter = 0;
		double accelerated_meanIter = 0.0;
		int accelerated_sumIter = 0;

		double zero_guess_min_total_CPU_time = 1000000.0;
		double zero_guess_max_total_CPU_time = 0.0;
		double zero_guess_mean_total_CPU_time = 0.0;
		double zero_guess_mean_sum_CPU_time = 0.0;

		double accelerated_min_total_CPU_time = 1000000.0;
		double accelerated_max_total_CPU_time = 0.0;
		double accelerated_mean_total_CPU_time = 0.0;
		double accelerated_mean_sum_CPU_time = 0.0;

		double min_kdtree_init_guess_CPU_time = 1000000.0;
		double max_kdtree_init_guess_CPU_time = 0.0;
		double mean_kdtree_init_guess_CPU_time = 0.0;
		double sum_kdtree_init_guess_CPU_time = 0.0;
	};

/*---------------------------------------------------------------------------*/
/*                           Class interface                                 */
/*---------------------------------------------------------------------------*/

	template<int dim>
	class MonteCarloSolver
	{
		public:
			enum correction_or_base { base, correction, test };
			typedef Navier_Stokes<dim> problem_type;

		public:
			MonteCarloSolver( int l, double lin_sol_err, int n = 1, correction_or_base mode_ = correction );	// n is the number of MC samples
			virtual ~MonteCarloSolver() {};
			void run( solution_type<dim> &MC_solution, solution_type<dim> &variance, int n_old=0 );				// run MC simulation, n_old is the number of samples of passed by reference MC_solution vector
			MC_info get_coarse_MC_info();
			MC_info get_fine_MC_info();

		private:
			my_timer			timer;
			int num_of_samples;											// number of MC samples
			int level;													// level of discretization
			correction_or_base mode;									// base or correction mode
			problem_type 		coarse_Problem,		fine_Problem;		// coarse & fine problems
//			lin_solver_info		coarse_info,		fine_info;			// solver info
			MC_info				coarse_level_info,	fine_level_info;	// MC info
			solution_type<dim>  coarse_solution, 	fine_solution;		// all sample solutions
			solution_type<dim>  level_solution;							// level correction
			/*---------------------------------------------------------------*/
			Coefficient<dim>	coefficient_function;
			BoundaryValues<dim>	BV_function;
			RightHandSide<dim>	rhs_function;
			/*---------------------------------------------------------------*/
			Output<dim>			toFile;
//			void update_coarse_MC_info( lin_solver_info &coarse_info );
//			void update_fine_MC_info( lin_solver_info &fine_info );
			void compute_mean( solution_type<dim> &MC_solution, const std::vector<solution_type<dim>> &solutions );
			void compute_variance( solution_type<dim> &variance, const solution_type<dim> &MC_solution, const std::vector<solution_type<dim>> &solutions );
	};





/*---------------------------------------------------------------------------*/
/*                           Class implementation                            */
/*---------------------------------------------------------------------------*/


	template<int dim>
	MonteCarloSolver<dim>::MonteCarloSolver( int l, double lin_sol_err, int n, correction_or_base mode_ )
	{
		level = l;
		num_of_samples = n;
		mode = mode_;

		if ( num_of_samples != 0 )
		{
			// there are two modes: base & corrections
			switch (mode)
			{
				case test:
					coarse_Problem.reinit(level, lin_sol_err);
					break;
				case base:
					coarse_Problem.reinit(level, lin_sol_err);
					break;
				case correction:
					coarse_Problem.reinit(level-1, lin_sol_err);
					fine_Problem.reinit(level, lin_sol_err);
					coarse_solution.reinit( level-1 );
					break;
			}

			level_solution.reinit(level);
			fine_solution.reinit(level);

			double cor_length = 14.0 / sqrt(6.0);
			BV_function.initialize( cor_length );
		}
	}












	template<int dim>
	void MonteCarloSolver<dim>::run( solution_type<dim> &MC_solution, solution_type<dim> &variance, int num_of_samples_old )
	{
		if ( num_of_samples != 0 )
		{
			std::ofstream logfile("results.txt", ios::out | ios::app);

			BV_function.generate(Normal);
			coarse_Problem.run( coefficient_function, rhs_function, BV_function, level_solution );
			allSolutions<dim>::add( BV_function.get_coefficients(), level_solution, -(0+num_of_samples_old) );
			update_statistics_v( 0, 1, MC_solution.vector, level_solution.vector, variance.vector, variance.vector );

//			logfile << level << " " << 0+num_of_samples_old << " "; coarse_Problem.info.print(logfile); logfile << " " << std::endl;

			if ( mode == test )
				for ( int i = 1; i < num_of_samples; i++ )
				{
					BV_function.generate(Normal);

					level_solution.sample = -(i+num_of_samples_old);
					coarse_Problem.run( coarse_Problem.Stokes_operator_matrix, coarse_Problem.Stokes_operator_rhs, BV_function, level_solution );

					allSolutions<dim>::add( BV_function.get_coefficients(), level_solution, -(i+num_of_samples_old) );

					update_statistics_v( i, 1, MC_solution.vector, level_solution.vector, variance.vector, variance.vector );

					std::cerr << i << std::endl;
//					logfile << level << " " << i+num_of_samples_old << " "; coarse_Problem.info.print(logfile);	logfile << " " << std::endl;
				}
			else if ( mode == base )
				for ( int i = 1; i < num_of_samples; i++ )
				{
					logfile << level << " " << i+num_of_samples_old << " ";

					BV_function.generate(Normal);

					// zero initial guess
					level_solution.vector = 0.0;
					coarse_Problem.run( coarse_Problem.Stokes_operator_matrix, coarse_Problem.Stokes_operator_rhs, BV_function, level_solution );
					coarse_Problem.info.print(logfile); logfile << " ";

					// estimated initial guess
					double knn_time;
					std::vector<double> params = BV_function.get_coefficients();
					timer.tic();
					allSolutions<dim>::find_closest( params, level_solution );
					knn_time = timer.toc();
					coarse_Problem.run( coarse_Problem.Stokes_operator_matrix, coarse_Problem.Stokes_operator_rhs, BV_function, level_solution );

					allSolutions<dim>::add( BV_function.get_coefficients(), level_solution, i+num_of_samples_old );

					update_statistics_v( i, 1, MC_solution.vector, level_solution.vector, variance.vector, variance.vector );

					coarse_Problem.info.print(logfile);	logfile << std::scientific << std::setprecision(4) <<  " " << knn_time << " " << std::endl;
				}
			else if ( mode == correction ) /* mode == corrections */
				for ( int i = 1; i < num_of_samples; i++ )
				{
					coefficient_function.generate(Normal);

					solution_type<dim> coarse_solution;
					coarse_solution.reinit( level-1 );

					// solve coarse problem
					coarse_solution.sample = i+num_of_samples_old;
//					coarse_Problem.run( &coefficient_function, coarse_solution);
//					coarse_info = coarse_Problem.get_solver_info();

					// interpolate init guess from coarse mesh
					fine_solution.interpolate_from( coarse_solution );
					level_solution.vector = 0.0;
					level_solution.vector -= fine_solution.vector;

					// solve fine problem
					fine_solution.sample = i+num_of_samples_old;
//					fine_Problem.run( &coefficient_function, fine_solution );
//					fine_info = fine_Problem.get_solver_info();

					// compute level correction
					level_solution.vector += fine_solution.vector;

					allSolutions<dim>::add( coefficient_function.get_coefficients(), fine_solution, i+num_of_samples_old );

//					update_coarse_MC_info( coarse_info );
//					update_fine_MC_info( fine_info );
//					update_statistics_v( i, 1, MC_solution.vector, level_solution.vector, variance.vector, variance.vector );
				}
		}
	}







	template<int dim>
	void MonteCarloSolver<dim>::compute_mean( solution_type<dim> &MC_solution, const std::vector<solution_type<dim>> &solutions )
	{
		MC_solution.vector = 0.0;
		for ( int sample = 0; sample < num_of_samples; sample++ )
			MC_solution.vector += solutions[sample].vector;
		MC_solution.vector /= num_of_samples;
	}


	template<int dim>
	void MonteCarloSolver<dim>::compute_variance( solution_type<dim> &variance, const solution_type<dim> &MC_solution, const std::vector<solution_type<dim>> &solutions )
	{
		variance.vector = 0.0;
		for ( int sample = 0; sample < num_of_samples; sample++ )
			for ( int i = 0; i < variance.vector.size(); i++ )
				variance.vector[i] += ( solutions[sample].vector[i] - MC_solution.vector[i] ) * ( solutions[sample].vector[i] - MC_solution.vector[i] );
		variance.vector /= std::max( 1, num_of_samples - 1 );
	}


//	template<int dim>
//	void MonteCarloSolver<dim>::update_coarse_MC_info( lin_solver_info &coarse_info )
//	{
//		coarse_level_info.zero_guess_sumIter +=	coarse_info.zero_guess_iterations;
//		coarse_level_info.zero_guess_minIter = ( coarse_info.zero_guess_iterations < coarse_level_info.zero_guess_minIter ) ? coarse_info.zero_guess_iterations : coarse_level_info.zero_guess_minIter;
//		coarse_level_info.zero_guess_maxIter = ( coarse_info.zero_guess_iterations > coarse_level_info.zero_guess_maxIter ) ? coarse_info.zero_guess_iterations : coarse_level_info.zero_guess_maxIter;
//
//		coarse_level_info.accelerated_sumIter += coarse_info.accelerated_iterations;
//		coarse_level_info.accelerated_minIter = ( coarse_info.accelerated_iterations < coarse_level_info.accelerated_minIter ) ? coarse_info.accelerated_iterations : coarse_level_info.accelerated_minIter;
//		coarse_level_info.accelerated_maxIter = ( coarse_info.accelerated_iterations > coarse_level_info.accelerated_maxIter ) ? coarse_info.accelerated_iterations : coarse_level_info.accelerated_maxIter;
//
//		coarse_level_info.zero_guess_mean_sum_CPU_time += coarse_info.zero_guess_total_CPU_time;
//		coarse_level_info.zero_guess_min_total_CPU_time = ( coarse_info.zero_guess_total_CPU_time < coarse_level_info.zero_guess_min_total_CPU_time ) ? coarse_info.zero_guess_total_CPU_time : coarse_level_info.zero_guess_min_total_CPU_time;
//		coarse_level_info.zero_guess_max_total_CPU_time = ( coarse_info.zero_guess_total_CPU_time > coarse_level_info.zero_guess_max_total_CPU_time ) ? coarse_info.zero_guess_total_CPU_time : coarse_level_info.zero_guess_max_total_CPU_time;
//
//		coarse_level_info.accelerated_mean_sum_CPU_time += coarse_info.accelerated_total_CPU_time;
//		coarse_level_info.accelerated_min_total_CPU_time = ( coarse_info.accelerated_total_CPU_time < coarse_level_info.accelerated_min_total_CPU_time ) ? coarse_info.accelerated_total_CPU_time : coarse_level_info.accelerated_min_total_CPU_time;
//		coarse_level_info.accelerated_max_total_CPU_time = ( coarse_info.accelerated_total_CPU_time > coarse_level_info.accelerated_max_total_CPU_time ) ? coarse_info.accelerated_total_CPU_time : coarse_level_info.accelerated_max_total_CPU_time;
//
//		coarse_level_info.sum_kdtree_init_guess_CPU_time += coarse_info.kdtree_init_guess_CPU_time;
//		coarse_level_info.min_kdtree_init_guess_CPU_time = ( coarse_info.kdtree_init_guess_CPU_time < coarse_level_info.min_kdtree_init_guess_CPU_time ) ? coarse_info.kdtree_init_guess_CPU_time : coarse_level_info.min_kdtree_init_guess_CPU_time;
//		coarse_level_info.max_kdtree_init_guess_CPU_time = ( coarse_info.kdtree_init_guess_CPU_time > coarse_level_info.max_kdtree_init_guess_CPU_time ) ? coarse_info.kdtree_init_guess_CPU_time : coarse_level_info.max_kdtree_init_guess_CPU_time;
//	}
//
//
//	template<int dim>
//	void MonteCarloSolver<dim>::update_fine_MC_info( lin_solver_info &fine_info )
//	{
//		fine_level_info.zero_guess_sumIter +=	fine_info.zero_guess_iterations;
//		fine_level_info.zero_guess_minIter = ( fine_info.zero_guess_iterations < fine_level_info.zero_guess_minIter ) ? fine_info.zero_guess_iterations : fine_level_info.zero_guess_minIter;
//		fine_level_info.zero_guess_maxIter = ( fine_info.zero_guess_iterations > fine_level_info.zero_guess_maxIter ) ? fine_info.zero_guess_iterations : fine_level_info.zero_guess_maxIter;
//
//		fine_level_info.accelerated_sumIter += fine_info.accelerated_iterations;
//		fine_level_info.accelerated_minIter = ( fine_info.accelerated_iterations < fine_level_info.accelerated_minIter ) ? fine_info.accelerated_iterations : fine_level_info.accelerated_minIter;
//		fine_level_info.accelerated_maxIter = ( fine_info.accelerated_iterations > fine_level_info.accelerated_maxIter ) ? fine_info.accelerated_iterations : fine_level_info.accelerated_maxIter;
//
//		fine_level_info.zero_guess_mean_sum_CPU_time += fine_info.zero_guess_total_CPU_time;
//		fine_level_info.zero_guess_min_total_CPU_time = ( fine_info.zero_guess_total_CPU_time < fine_level_info.zero_guess_min_total_CPU_time ) ? fine_info.zero_guess_total_CPU_time : fine_level_info.zero_guess_min_total_CPU_time;
//		fine_level_info.zero_guess_max_total_CPU_time = ( fine_info.zero_guess_total_CPU_time > fine_level_info.zero_guess_max_total_CPU_time ) ? fine_info.zero_guess_total_CPU_time : fine_level_info.zero_guess_max_total_CPU_time;
//
//		fine_level_info.accelerated_mean_sum_CPU_time += fine_info.accelerated_total_CPU_time;
//		fine_level_info.accelerated_min_total_CPU_time = ( fine_info.accelerated_total_CPU_time < fine_level_info.accelerated_min_total_CPU_time ) ? fine_info.accelerated_total_CPU_time : fine_level_info.accelerated_min_total_CPU_time;
//		fine_level_info.accelerated_max_total_CPU_time = ( fine_info.accelerated_total_CPU_time > fine_level_info.accelerated_max_total_CPU_time ) ? fine_info.accelerated_total_CPU_time : fine_level_info.accelerated_max_total_CPU_time;
//
//		fine_level_info.sum_kdtree_init_guess_CPU_time += fine_info.kdtree_init_guess_CPU_time;
//		fine_level_info.min_kdtree_init_guess_CPU_time = ( fine_info.kdtree_init_guess_CPU_time < fine_level_info.min_kdtree_init_guess_CPU_time ) ? fine_info.kdtree_init_guess_CPU_time : fine_level_info.min_kdtree_init_guess_CPU_time;
//		fine_level_info.max_kdtree_init_guess_CPU_time = ( fine_info.kdtree_init_guess_CPU_time > fine_level_info.max_kdtree_init_guess_CPU_time ) ? fine_info.kdtree_init_guess_CPU_time : fine_level_info.max_kdtree_init_guess_CPU_time;
//	}


	template<int dim>
	MC_info MonteCarloSolver<dim>::get_coarse_MC_info()
	{
		if ( num_of_samples != 0 )
		{
			coarse_level_info.zero_guess_meanIter  = double(coarse_level_info.zero_guess_sumIter) / num_of_samples;
			coarse_level_info.accelerated_meanIter = double(coarse_level_info.accelerated_sumIter) / num_of_samples;
			coarse_level_info.zero_guess_mean_total_CPU_time = coarse_level_info.zero_guess_mean_sum_CPU_time / num_of_samples;
			coarse_level_info.accelerated_mean_total_CPU_time = coarse_level_info.accelerated_mean_sum_CPU_time / num_of_samples;
			coarse_level_info.mean_kdtree_init_guess_CPU_time = coarse_level_info.sum_kdtree_init_guess_CPU_time / num_of_samples;
		}
		return coarse_level_info;
	}


	template<int dim>
	MC_info MonteCarloSolver<dim>::get_fine_MC_info()
	{
		if ( num_of_samples != 0 )
		{
			fine_level_info.zero_guess_meanIter  = double(fine_level_info.zero_guess_sumIter) / num_of_samples;
			fine_level_info.accelerated_meanIter = double(fine_level_info.accelerated_sumIter) / num_of_samples;
			fine_level_info.zero_guess_mean_total_CPU_time = fine_level_info.zero_guess_mean_sum_CPU_time / num_of_samples;
			fine_level_info.accelerated_mean_total_CPU_time = fine_level_info.accelerated_mean_sum_CPU_time / num_of_samples;
			fine_level_info.mean_kdtree_init_guess_CPU_time = fine_level_info.sum_kdtree_init_guess_CPU_time / num_of_samples;
		}
		return fine_level_info;
	}


} // namespace MonteCarlo


#endif /* MONTECARLO_H_ */

