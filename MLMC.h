/*
 * MLMC.h
 *
 *  Created on: Dec 13, 2014
 *      Author: viktor
 */

#ifndef MLMC_H_
#define MLMC_H_

#include <fstream>
#include "DataTypes.h"	// contains DiscretizationData.h
#include "MonteCarlo.h"
#include "Output.h"

#include "my_timer.h"

#include <math.h>       /* log2 */
//#include <lapacke.h>
#include <numeric>      // std::accumulate
#include <algorithm>    // std::max

using namespace dealii;

#define delimeter " | "
#define long_space "         "


namespace MonteCarlo
{

/*---------------------------------------------------------------------------*/
/*                           Class interface                                 */
/*---------------------------------------------------------------------------*/

	template <int dim>
	class MLMC
	{
		public:
			MLMC( double error, double corr_length );
			virtual ~MLMC() {};
			void run();

		private:
			int num_of_levels;											// number of levels
			double eps, eps_FE, eps_MC, eps_CG;							// tolerance of the solver
			double alfa, beta, gamma;									// parameters which show growth rate of error components
			std::vector<int>					num_of_samples;			// number of samples used inside while loop
			std::vector<int>					actual_num_of_samples;	// actually computed number of samples
			std::vector<int>					total_num_of_samples;	// final number of samples
			std::vector<solution_type<dim>>		level_corrections;		// solution corrections computed at each level
			std::vector<solution_type<dim>>		variance;				// variance of the level corrections
			std::vector<solution_type<dim>>		local_level_corrections;// solution corrections computed at each level inside while loop
			std::vector<solution_type<dim>>		local_variance;			// variance of the level corrections inside while loop
			std::vector<double>					mesh_diameters;			// mesh diameters at each level
			std::vector<double>					FE_errors;				// FE posterior errors computed for each grid in the hierarchy
			std::vector<double>					FE_costs;				// CPU times to compute FE solution at each grid in the hierarchy
			std::vector<double>					FE_costs_accel;			// CPU times to compute accelerated FE solution at each grid in the hierarchy
			std::vector<double>					averaged_variances;		// L-1 norm of the variance of the level corrections
			std::vector<double>					error_weights;			// weights assigning MC error to each level
			std::vector<MC_info>				info_coarse;			// MC solver info for coarse mesh at each level in the hierarchy
			std::vector<MC_info>				info_fine;				// MC solver info for fine mesh at each level in the hierarchy
			solution_type<dim>					solution;				// final solution vector
			void compute_num_of_samples();
			void compute_num_of_levels ();
			void estimate_parameter( const std::vector<double> &x, const std::vector<double> &y, double &parameter );
			void update_statistics( int level );
			void update_level_solution( int level );
			void update_level_variance( int level );
			void update_coarse_info( int level, MC_info &tmp_info );
			void update_fine_info( int level, MC_info &tmp_info );
			void print_log( int loop );
			void print_stat();
			void print_stat_CPU_line(MC_info &info_coarse, MC_info &info_fine, int level);
			void print_stat_accel_CPU_line(MC_info &info_coarse, MC_info &info_fine, int level);
			Output<2> toFile;
			my_timer 			timer;
	};





/*---------------------------------------------------------------------------*/
/*                           Class implementation                            */
/*---------------------------------------------------------------------------*/

	template <int dim>
	MLMC<dim>::MLMC( double error, double corr_length )
	{
		// tolerance of the FE discretization, MLMC estimator & CG solver
		eps = error;
		eps_FE = 0.8 * eps;
		eps_MC = 0.1 * eps;
		eps_CG = 0.1 * eps;

		// initialize coefficient random field: stoch_dim, type_of_distribution
		Coefficient<dim> coefficient;
		coefficient.initialize( corr_length );

		DiscretizationData<dim>::initialize();
		Problem<dim>::initialize();

		// compute number of levels and generate corresponding meshes
		compute_num_of_levels();

		// resize all vectors
		num_of_samples.resize(num_of_levels);
		total_num_of_samples.resize(num_of_levels);
		actual_num_of_samples.resize(num_of_levels);

		error_weights.resize(num_of_levels);
		averaged_variances.resize(num_of_levels);

		FE_costs_accel.resize(num_of_levels);

		info_coarse.resize(num_of_levels);
		info_fine.resize(num_of_levels);

		level_corrections.resize(num_of_levels);
		variance.resize(num_of_levels);
		local_level_corrections.resize(num_of_levels);
		local_variance.resize(num_of_levels);

		// final solution is defined on the finest mesh
		solution.reinit( num_of_levels-1 );

		// initialize all vectors for level corrections and variance
		for ( int level = 0; level < num_of_levels; level++ )
		{
			total_num_of_samples[level] = 0;
			level_corrections[level].reinit( level );
			variance[level].reinit( level );
			local_level_corrections[level].reinit( level );
			local_variance[level].reinit( level );
		}

		// set up initial guess for number of samples at each level
		compute_num_of_samples();
	}


	template <int dim>
	void MLMC<dim>::compute_num_of_levels()
	{
		deallog.depth_console (0);

		double cor_length = Coefficient<dim>::get_cor_length();

		float	test_error			= 100.0;
		int		init_level			= 5; //ceil( -1.0 * log(cor_length/5.0) / log(2.0) );
		int		test_level			= 0;
		int		test_num_of_samples	= 1;

		MC_info info;

		std::cout << std::endl;
		std::cout << "/********************************************************************************/" << std::endl
		          << "                           Estimate number of levels                              " << std::endl
		          << "/********************************************************************************/" << std::endl;
		std::cout << "---------------------------------------------------" << std::endl
				  << "       |        |   test  |            |           " << std::endl
		          << " level |    h   |    M_l  |  FE error  |  FE cost  " << std::endl
		          << "---------------------------------------------------" << std::endl;
//		while ( test_error > eps_FE )
		while ( test_level + init_level <= 11 )
		{
			DiscretizationData<dim>::add_next_level( init_level );										// generate next level
			Problem<dim> problem;
			problem.add_next_level_data();																// assemble deterministic rhs & boundary conditions

			solution_type<dim> level_solution(test_level), level_variance(test_level);					// initialize solution & variance at this level

			MonteCarloSolver<dim> MC_problem( test_level, std::min(1.0e-6, eps_CG / 5.0), test_num_of_samples, test );		//  1e-6 since we have to measure only FE error
			MC_problem.run( level_solution, level_variance );											// get solution & variance

			test_error	= level_solution.estimate_posterior_error();									// estimate FE error at this level
			info		= MC_problem.get_coarse_MC_info();												// get solver info

			mesh_diameters.push_back( 1.0 / pow(2.0, test_level+init_level) );							// calculate mesh diameters at each level
			FE_errors.push_back( test_error );															//
			FE_costs.push_back( info.zero_guess_mean_total_CPU_time );


			std::cout << std::fixed;
			std::cout << std::setprecision(3);
			std::cout << std::setw(6) << test_level << delimeter
					  << std::setw(6) << mesh_diameters[test_level] << delimeter
					  << std::setw(7) << test_num_of_samples << delimeter;
			std::cout << std::scientific;
			std::cout << std::setw(10) << test_error << delimeter
					  << std::setw(10) << FE_costs[test_level]
					  << std::endl;
			test_level++;
		}
		DiscretizationData<dim>::finalize();															// finalize mesh creation --> generate intergrid maps
		std::cout << "---------------------------------------------------" << std::endl;

		num_of_levels = DiscretizationData<dim>::num_of_levels;
		std::cout << "Initial level:    " << init_level << std::endl
				  << "Number of levels: " << num_of_levels << std::endl
				  << "FE error:         " << eps_FE << std::endl
				  << "MC error:         " << eps_MC << std::endl
				  << "CG error:         " << eps_CG << std::endl
				  << "Actual  FE error: " << FE_errors[num_of_levels-1] << std::endl
				  << "correl. length:   " << cor_length << std::endl
				  << "stoch_dim:        " << Coefficient<dim>::get_stoch_dim() << std::endl;
	}


	template <int dim>
	void MLMC<dim>::compute_num_of_samples()
	{
		num_of_samples[0] = 1000000;
//		eps_CG = 1e-8;
//		eps_CG = 0.01 / sqrt(num_of_samples[0]);
		for ( int level = 1; level < num_of_levels - 2; level++ )
			num_of_samples[level] = 50;
		for ( int level = num_of_levels - 2; level < num_of_levels; level++ )
			num_of_samples[level] = 20;
	}


	template <int dim>
	void MLMC<dim>::estimate_parameter( const std::vector<double> &x, const std::vector<double> &y, double &parameter )
	{
//		int m, n, lda, ldb, nrhs;
//
//		m = x.size();
//		n = 2;
//		nrhs = 1;
//		lda = m;
//		ldb = m;
//
//		double* A = new double[2*m];
//		double* B = new double[m];
//
//		for ( int i = 0; i < m; i++ )
//		{
//			A[i] = 1.0;
//			A[i+m] = log2( x[i] );
//			B[i] = log2( y[i] );
//		}
//
//		LAPACKE_dgels( LAPACK_COL_MAJOR, 'N', m, n, nrhs, A, lda, B, ldb );
//
//		parameter = B[1];
//
//		delete A;
//		delete B;
	}



	template <int dim>
	void MLMC<dim>::run()
	{
		deallog.depth_console (0);

		MC_info	tmp_info;

		std::cout << std::endl << "Start MLMC solver" << std::endl;
		int samples_to_generate = std::accumulate( num_of_samples.begin(), num_of_samples.end(), 0 );
		int loop = 1;
		while ( samples_to_generate != 0 )
//		while ( loop < 2 )
		{

			/************************************************************************************************************************************************************************/
			/*                 compute solution and update variance at base level 0                                                                                                 */
			/************************************************************************************************************************************************************************/
			{
				MonteCarloSolver<dim> MC_problem(0, eps_CG / num_of_levels, num_of_samples[0], base );			// note "base" option
				MC_problem.run( local_level_corrections[0], local_variance[0], total_num_of_samples[0] );		// compute local level corrections and variances

				update_statistics( 0 );

				tmp_info = MC_problem.get_coarse_MC_info();
				update_coarse_info( 0, tmp_info );

				total_num_of_samples[0]	+= num_of_samples[0];													// update total number of samples computed at this level	( M_l )
				averaged_variances[0]	 = variance[0].L2_norm();												// update integrated variance of the solution at this level	( sigma_l )
			}


//			std::cout << "Start building kd-tree" << std::endl;
//			timer.tic();
			allSolutions<dim>::build_kd_tree();
//			std::cout << "Build kd-tree CPU time: " << timer.toc() << std::endl;


			/************************************************************************************************************************************************************************/
			/*                 compute corrections and update variances from other levels                                                                                           */
			/************************************************************************************************************************************************************************/

			for ( int level = 1; level < num_of_levels; level++ )
			{
				MonteCarloSolver<dim> MC_problem(level, eps_CG / num_of_levels, num_of_samples[level], correction);		// note "correction" option
				MC_problem.run( local_level_corrections[level], local_variance[level], total_num_of_samples[level] );	// compute local level corrections and variances

				update_statistics( level );

				tmp_info = MC_problem.get_coarse_MC_info();
				update_coarse_info( level, tmp_info );
				tmp_info = MC_problem.get_fine_MC_info();
				update_fine_info( level, tmp_info );

				total_num_of_samples[level]	+= num_of_samples[level];													// update total number of samples computed at this level	( M_l )
				averaged_variances[level]	 = variance[level].L2_norm();												// update integrated variance of the solution at this level	( sigma_l )

//				std::cout << "Start building kd-tree" << std::endl;
//				timer.tic();
//				allSolutions<dim>::build_kd_tree();
//				std::cout << "Build kd-tree CPU time: " << timer.toc() << std::endl;
			}


			/************************************************************************************************************************************************************************/
			/*                 update error weights and number of samples to be additionally generated                                                                              */
			/************************************************************************************************************************************************************************/

			FE_costs[0] = info_coarse[0].zero_guess_mean_total_CPU_time;
			FE_costs_accel[0] = info_coarse[0].accelerated_mean_total_CPU_time;;
			for ( int level = 1; level < num_of_levels; level++ )
			{
				FE_costs[level] = info_coarse[level].zero_guess_mean_total_CPU_time; // + info_fine[level].zero_guess_mean_total_CPU_time;
				FE_costs_accel[level] = info_coarse[level].accelerated_mean_total_CPU_time; // + info_fine[level].accelerated_mean_total_CPU_time;
			}

			double denom = 0.0;
			for ( int level = 0; level < num_of_levels; level++ )
				denom += pow( FE_costs[level]*averaged_variances[level], 1.0/3.0 );

			for ( int level = 0; level < num_of_levels; level++ )
			{
				error_weights[level] = pow( FE_costs[level] * averaged_variances[level], 1.0/3.0 ) / denom;
				actual_num_of_samples[level] = (int)ceil( averaged_variances[level] / ( eps_MC * eps_MC * error_weights[level] * error_weights[level] ) );
				num_of_samples[level] = std::max( 0, actual_num_of_samples[level] - total_num_of_samples[level] );
			}

			print_log( loop );

			samples_to_generate = std::accumulate( num_of_samples.begin(), num_of_samples.end(), 0 );
			loop++;
		}

		/************************************************************************************************************************************************************************/
		/*                 estimate growth of FE error, FE cost and variance                                                                                                    */
		/************************************************************************************************************************************************************************/

//		estimate_parameter( mesh_diameters, FE_errors, alfa );
//		estimate_parameter( mesh_diameters, averaged_variances, beta );
//		estimate_parameter( mesh_diameters, FE_costs, gamma );
//		gamma = -1.0*gamma;
//
//		std::cout << std::fixed;
//		std::cout << "alfa:  " << alfa  << std::endl;
//		std::cout << "beta:  " << beta  << std::endl;
//		std::cout << "gamma: " << gamma << std::endl;

		print_stat();

		for ( int level = 0; level < num_of_levels; level++ )
			solution.add( level_corrections[level] );


//		for ( int level = 0; level < num_of_levels; level++ )
//		{
//			toFile.print( level_corrections[level], "level-correction-" + std::to_string(level) );
//			toFile.print( variance[level], "level-variance-" + std::to_string(level) );
//		}
//		toFile.print( solution, "final_solution" );
	}


	template <int dim>
	void MLMC<dim>::update_statistics( int lev )
	{
		update_statistics_v(	total_num_of_samples[lev],		num_of_samples[lev],
								level_corrections[lev].vector,	local_level_corrections[lev].vector,
								variance[lev].vector, 			local_variance[lev].vector);
	}


	template <int dim>
	void MLMC<dim>::update_level_solution( int level )
	{
		int loc_samples = num_of_samples[level];
		int lev_samples = total_num_of_samples[level];

		for ( int i = 0; i < level_corrections[level].vector.size(); i++ )
			level_corrections[level].vector[i] = ( lev_samples * level_corrections[level].vector[i] + loc_samples * local_level_corrections[level].vector[i] ) / ( loc_samples + lev_samples );
	}


	template <int dim>
	void MLMC<dim>::update_level_variance( int level )
	{
		int loc_samples = num_of_samples[level];
		int lev_samples = total_num_of_samples[level];

		double b1 = (double)( loc_samples * lev_samples ) / ( loc_samples + lev_samples - 1 ) / ( loc_samples + lev_samples );

		for ( int i = 0; i < variance[level].vector.size(); i++ )
		{
			variance[level].vector[i] = ( std::max(0,lev_samples-1) * variance[level].vector[i] + std::max(0,loc_samples-1) * local_variance[level].vector[i] ) / ( std::max(1,loc_samples + lev_samples - 1) )
			                       + b1 * ( level_corrections[level].vector[i] - local_level_corrections[level].vector[i] ) * ( level_corrections[level].vector[i] - local_level_corrections[level].vector[i] );
		}
	}


	template <int dim>
	void MLMC<dim>::update_coarse_info( int level, MC_info &tmp_info )
	{
		info_coarse[level].zero_guess_minIter = ( tmp_info.zero_guess_minIter < info_coarse[level].zero_guess_minIter ) ? tmp_info.zero_guess_minIter : info_coarse[level].zero_guess_minIter;
		info_coarse[level].zero_guess_maxIter = ( tmp_info.zero_guess_maxIter > info_coarse[level].zero_guess_maxIter ) ? tmp_info.zero_guess_maxIter : info_coarse[level].zero_guess_maxIter;
		info_coarse[level].zero_guess_meanIter = ( total_num_of_samples[level] * info_coarse[level].zero_guess_meanIter + num_of_samples[level] * tmp_info.zero_guess_meanIter ) / ( total_num_of_samples[level] + num_of_samples[level] );
		info_coarse[level].zero_guess_sumIter += tmp_info.zero_guess_sumIter;

		info_coarse[level].accelerated_minIter = ( tmp_info.accelerated_minIter < info_coarse[level].accelerated_minIter ) ? tmp_info.accelerated_minIter : info_coarse[level].accelerated_minIter;
		info_coarse[level].accelerated_maxIter = ( tmp_info.accelerated_maxIter > info_coarse[level].accelerated_maxIter ) ? tmp_info.accelerated_maxIter : info_coarse[level].accelerated_maxIter;
		info_coarse[level].accelerated_meanIter = ( total_num_of_samples[level] * info_coarse[level].accelerated_meanIter + num_of_samples[level] * tmp_info.accelerated_meanIter ) / ( total_num_of_samples[level] + num_of_samples[level] );
		info_coarse[level].accelerated_sumIter += tmp_info.accelerated_sumIter;

		info_coarse[level].zero_guess_min_total_CPU_time = ( tmp_info.zero_guess_min_total_CPU_time < info_coarse[level].zero_guess_min_total_CPU_time ) ? tmp_info.zero_guess_min_total_CPU_time : info_coarse[level].zero_guess_min_total_CPU_time;
		info_coarse[level].zero_guess_max_total_CPU_time = ( tmp_info.zero_guess_max_total_CPU_time > info_coarse[level].zero_guess_max_total_CPU_time ) ? tmp_info.zero_guess_max_total_CPU_time : info_coarse[level].zero_guess_max_total_CPU_time;
		info_coarse[level].zero_guess_mean_total_CPU_time = ( total_num_of_samples[level] * info_coarse[level].zero_guess_mean_total_CPU_time + num_of_samples[level] * tmp_info.zero_guess_mean_total_CPU_time ) / ( total_num_of_samples[level] + num_of_samples[level] );
		info_coarse[level].zero_guess_mean_sum_CPU_time += tmp_info.zero_guess_mean_sum_CPU_time;

		info_coarse[level].accelerated_min_total_CPU_time = ( tmp_info.accelerated_min_total_CPU_time < info_coarse[level].accelerated_min_total_CPU_time ) ? tmp_info.accelerated_min_total_CPU_time : info_coarse[level].accelerated_min_total_CPU_time;
		info_coarse[level].accelerated_max_total_CPU_time = ( tmp_info.accelerated_max_total_CPU_time > info_coarse[level].accelerated_max_total_CPU_time ) ? tmp_info.accelerated_max_total_CPU_time : info_coarse[level].accelerated_max_total_CPU_time;
		info_coarse[level].accelerated_mean_total_CPU_time = ( total_num_of_samples[level] * info_coarse[level].accelerated_mean_total_CPU_time + num_of_samples[level] * tmp_info.accelerated_mean_total_CPU_time ) / ( total_num_of_samples[level] + num_of_samples[level] );
		info_coarse[level].accelerated_mean_sum_CPU_time += tmp_info.accelerated_mean_sum_CPU_time;

		info_coarse[level].min_kdtree_init_guess_CPU_time = ( tmp_info.min_kdtree_init_guess_CPU_time < info_coarse[level].min_kdtree_init_guess_CPU_time ) ? tmp_info.min_kdtree_init_guess_CPU_time : info_coarse[level].min_kdtree_init_guess_CPU_time;
		info_coarse[level].max_kdtree_init_guess_CPU_time = ( tmp_info.max_kdtree_init_guess_CPU_time > info_coarse[level].max_kdtree_init_guess_CPU_time ) ? tmp_info.max_kdtree_init_guess_CPU_time : info_coarse[level].max_kdtree_init_guess_CPU_time;
		info_coarse[level].mean_kdtree_init_guess_CPU_time = ( total_num_of_samples[level] * info_coarse[level].mean_kdtree_init_guess_CPU_time + num_of_samples[level] * tmp_info.mean_kdtree_init_guess_CPU_time ) / ( total_num_of_samples[level] + num_of_samples[level] );
		info_coarse[level].sum_kdtree_init_guess_CPU_time += tmp_info.sum_kdtree_init_guess_CPU_time;
	}


	template <int dim>
	void MLMC<dim>::update_fine_info( int level, MC_info &tmp_info )
	{
		info_fine[level].zero_guess_minIter = ( tmp_info.zero_guess_minIter < info_fine[level].zero_guess_minIter ) ? tmp_info.zero_guess_minIter : info_fine[level].zero_guess_minIter;
		info_fine[level].zero_guess_maxIter = ( tmp_info.zero_guess_maxIter > info_fine[level].zero_guess_maxIter ) ? tmp_info.zero_guess_maxIter : info_fine[level].zero_guess_maxIter;
		info_fine[level].zero_guess_meanIter = ( total_num_of_samples[level] * info_fine[level].zero_guess_meanIter + num_of_samples[level] * tmp_info.zero_guess_meanIter ) / ( total_num_of_samples[level] + num_of_samples[level] );
		info_fine[level].zero_guess_sumIter += tmp_info.zero_guess_sumIter;

		info_fine[level].accelerated_minIter = ( tmp_info.accelerated_minIter < info_fine[level].accelerated_minIter ) ? tmp_info.accelerated_minIter : info_fine[level].accelerated_minIter;
		info_fine[level].accelerated_maxIter = ( tmp_info.accelerated_maxIter > info_fine[level].accelerated_maxIter ) ? tmp_info.accelerated_maxIter : info_fine[level].accelerated_maxIter;
		info_fine[level].accelerated_meanIter = ( total_num_of_samples[level] * info_fine[level].accelerated_meanIter + num_of_samples[level] * tmp_info.accelerated_meanIter ) / ( total_num_of_samples[level] + num_of_samples[level] );
		info_fine[level].accelerated_sumIter += tmp_info.accelerated_sumIter;

		info_fine[level].zero_guess_min_total_CPU_time = ( tmp_info.zero_guess_min_total_CPU_time < info_fine[level].zero_guess_min_total_CPU_time ) ? tmp_info.zero_guess_min_total_CPU_time : info_fine[level].zero_guess_min_total_CPU_time;
		info_fine[level].zero_guess_max_total_CPU_time = ( tmp_info.zero_guess_max_total_CPU_time > info_fine[level].zero_guess_max_total_CPU_time ) ? tmp_info.zero_guess_max_total_CPU_time : info_fine[level].zero_guess_max_total_CPU_time;
		info_fine[level].zero_guess_mean_total_CPU_time = ( total_num_of_samples[level] * info_fine[level].zero_guess_mean_total_CPU_time + num_of_samples[level] * tmp_info.zero_guess_mean_total_CPU_time ) / ( total_num_of_samples[level] + num_of_samples[level] );
		info_fine[level].zero_guess_mean_sum_CPU_time += tmp_info.zero_guess_mean_sum_CPU_time;

		info_fine[level].accelerated_min_total_CPU_time = ( tmp_info.accelerated_min_total_CPU_time < info_fine[level].accelerated_min_total_CPU_time ) ? tmp_info.accelerated_min_total_CPU_time : info_fine[level].accelerated_min_total_CPU_time;
		info_fine[level].accelerated_max_total_CPU_time = ( tmp_info.accelerated_max_total_CPU_time > info_fine[level].accelerated_max_total_CPU_time ) ? tmp_info.accelerated_max_total_CPU_time : info_fine[level].accelerated_max_total_CPU_time;
		info_fine[level].accelerated_mean_total_CPU_time = ( total_num_of_samples[level] * info_fine[level].accelerated_mean_total_CPU_time + num_of_samples[level] * tmp_info.accelerated_mean_total_CPU_time ) / ( total_num_of_samples[level] + num_of_samples[level] );
		info_fine[level].accelerated_mean_sum_CPU_time += tmp_info.accelerated_mean_sum_CPU_time;

		info_fine[level].min_kdtree_init_guess_CPU_time = ( tmp_info.min_kdtree_init_guess_CPU_time < info_fine[level].min_kdtree_init_guess_CPU_time ) ? tmp_info.min_kdtree_init_guess_CPU_time : info_fine[level].min_kdtree_init_guess_CPU_time;
		info_fine[level].max_kdtree_init_guess_CPU_time = ( tmp_info.max_kdtree_init_guess_CPU_time > info_fine[level].max_kdtree_init_guess_CPU_time ) ? tmp_info.max_kdtree_init_guess_CPU_time : info_fine[level].max_kdtree_init_guess_CPU_time;
		info_fine[level].mean_kdtree_init_guess_CPU_time = ( total_num_of_samples[level] * info_fine[level].mean_kdtree_init_guess_CPU_time + num_of_samples[level] * tmp_info.mean_kdtree_init_guess_CPU_time ) / ( total_num_of_samples[level] + num_of_samples[level] );
		info_fine[level].sum_kdtree_init_guess_CPU_time += tmp_info.sum_kdtree_init_guess_CPU_time;
	}


	template <int dim>
	void MLMC<dim>::print_log( int loop )
	{
		std::cout << "/********************************************************************************************************************/" << std::endl
		          << "                                  Loop #:" << loop                                                                      << std::endl
		          << "/********************************************************************************************************************/" << std::endl << std::endl
		          << "----------------------------------------------------------------------------------------------------------------------" << std::endl
		          << "       |        |        |    M_l  |    M_l  |    M_l  |             |   FE cost   |             |    / sigma_l       " << std::endl
		          << " level |    h   |   b_l  |   total |  actual |    add  |   FE cost   | accelerated |    sigma_l  |  l/         / M_l  " << std::endl
		          << "----------------------------------------------------------------------------------------------------------------------" << std::endl;
		double sum1 = 0.0;
		double sum2 = 0.0;
		int sum3 = 0.0;
		int sum4 = 0;
		int sum5 = 0;
		for ( int level = 0; level < num_of_levels; level++ )
		{
			std::cout << std::fixed;
			std::cout << std::setprecision(3);
			std::cout << std::setw(6) << level 							<< delimeter
					  << std::setw(6) << mesh_diameters[level] 			<< delimeter
				      << std::setw(6) << error_weights[level] 			<< delimeter
					  << std::setw(7) << total_num_of_samples[level]	<< delimeter
					  << std::setprecision(1)
					  << std::setw(7) << actual_num_of_samples[level]	<< delimeter
					  << std::setw(7) << num_of_samples[level] 			<< delimeter;
			std::cout.precision(5);
			std::cout << std::scientific;
			std::cout << std::setw(11) << FE_costs[level] 				<< delimeter
					  << std::setw(11) << FE_costs_accel[level]			<< delimeter
					  << std::setw(11) << averaged_variances[level] 	<< delimeter
			          << std::setw(11) << sqrt(averaged_variances[level] / total_num_of_samples[level])
			          << std::endl;
			sum1 += sqrt(averaged_variances[level] / total_num_of_samples[level]);
			sum2 += error_weights[level];
			sum3 += total_num_of_samples[level];
			sum4 += actual_num_of_samples[level];
			sum5 += num_of_samples[level];
		}
		std::cout << "----------------------------------------------------------------------------------------------------------------------" << std::endl;
		std::cout << std::fixed;
		std::cout << std::setprecision(3);
		std::cout << std::setw(6) << " "	<< delimeter
				  << std::setw(6) << " "	<< delimeter
			      << std::setw(6) << sum2 	<< delimeter
				  << std::setw(7) << sum3 	<< delimeter
				  << std::setw(7) << sum4 	<< delimeter
				  << std::setw(7) << sum5 	<< delimeter;
		std::cout << std::setw(11) << " "	<< delimeter
				  << std::setw(11) << " "	<< delimeter
				  << std::setw(11) << " "	<< delimeter
				  << std::scientific
				  << std::setw(11) << sum1 << std::endl << std::endl;
	}


	template <int dim>
	void MLMC<dim>::print_stat()
	{
		std::cout << std::endl << std::endl << std::endl;
		std::cout << "/****************************************************************************************************************************************************************************************/" << std::endl
		          << "                           Statistics                                                                                                                                                     " << std::endl
		          << "/****************************************************************************************************************************************************************************************/" << std::endl << std::endl
		          << "                                             Zero guess                                                                                  Accelerated                                      " << std::endl
		          << "---------------------------------------------------------------------------------------------       --------------------------------------------------------------------------------------" << std::endl
		          << "       |               coarse                                          fine                                       coarse                                          fine                    " << std::endl
		          << "---------------------------------------------------------------------------------------------       --------------------------------------------------------------------------------------" << std::endl
		          << " level |   min  |   max  |   mean  |  total  |         |   min  |   max  |   mean  |  total            min  |   max  |   mean  |  total  |         |   min  |   max  |   mean  |  total   " << std::endl
		          << "---------------------------------------------------------------------------------------------       --------------------------------------------------------------------------------------" << std::endl
		          << "                                             Iterations                                                                                   Iterations                                      " << std::endl
		          << "---------------------------------------------------------------------------------------------       --------------------------------------------------------------------------------------" << std::endl;
		std::cout << std::fixed;
		std::cout << std::setprecision(3);
		std::cout << std::setw(6) << 0   << delimeter
				  << std::setw(6) << " " << delimeter
			      << std::setw(6) << " " << delimeter
				  << std::setw(7) << " " << delimeter
				  << std::setw(7) << " " << delimeter
				  << std::setw(7) << " " << delimeter
				  << std::setw(6) << info_coarse[0].zero_guess_minIter   << delimeter
			      << std::setw(6) << info_coarse[0].zero_guess_maxIter   << delimeter
				  << std::setw(7) << info_coarse[0].zero_guess_meanIter  << delimeter
				  << std::setw(7) << info_coarse[0].zero_guess_sumIter
				  // accelerated
				  << std::setw(9) << "  | |  "
				  << std::setw(6) << " " << delimeter
			      << std::setw(6) << " " << delimeter
				  << std::setw(7) << " " << delimeter
				  << std::setw(7) << " " << delimeter
				  << std::setw(7) << " " << delimeter
				  << std::setw(6) << info_coarse[0].accelerated_minIter  << delimeter
			      << std::setw(6) << info_coarse[0].accelerated_maxIter  << delimeter
				  << std::setw(7) << info_coarse[0].accelerated_meanIter << delimeter
				  << std::setw(7) << info_coarse[0].accelerated_sumIter  << std::endl;
		for ( int level = 1; level < num_of_levels; level++ )
		{
			std::cout << std::setw(6) << level << delimeter
					  << std::setw(6) << info_coarse[level].zero_guess_minIter  << delimeter
				      << std::setw(6) << info_coarse[level].zero_guess_maxIter  << delimeter
					  << std::setw(7) << info_coarse[level].zero_guess_meanIter << delimeter
					  << std::setw(7) << info_coarse[level].zero_guess_sumIter  << delimeter
					  << std::setw(7) << " " << delimeter
					  << std::setw(6) << info_fine[level].zero_guess_minIter  << delimeter
				      << std::setw(6) << info_fine[level].zero_guess_maxIter  << delimeter
					  << std::setw(7) << info_fine[level].zero_guess_meanIter << delimeter
					  << std::setw(7) << info_fine[level].zero_guess_sumIter
					  // accelerated
					  << std::setw(9) << "  | |  "
					  << std::setw(6) << info_coarse[level].accelerated_minIter  << delimeter
				      << std::setw(6) << info_coarse[level].accelerated_maxIter  << delimeter
					  << std::setw(7) << info_coarse[level].accelerated_meanIter << delimeter
					  << std::setw(7) << info_coarse[level].accelerated_sumIter  << delimeter
					  << std::setw(7) << " " << delimeter
					  << std::setw(6) << info_fine[level].accelerated_minIter  << delimeter
				      << std::setw(6) << info_fine[level].accelerated_maxIter  << delimeter
					  << std::setw(7) << info_fine[level].accelerated_meanIter << delimeter
					  << std::setw(7) << info_fine[level].accelerated_sumIter
					  << std::endl;
		}

		std::cout << "-----------------------------------------------------------------------------------------------------------------" << std::endl
                  << "-----------------------------------------------------------------------------------------------------------------" << std::endl
                  << "                                               CPU time                                                          " << std::endl
                  << "-----------------------------------------------------------------------------------------------------------------" << std::endl;
		print_stat_CPU_line(info_coarse[0], info_fine[0], 0);
		for ( int level = 1; level < num_of_levels; level++ )
			print_stat_CPU_line(info_coarse[level], info_fine[level], level);

		std::cout << "-----------------------------------------------------------------------------------------------------------------" << std::endl
                  << "-----------------------------------------------------------------------------------------------------------------" << std::endl
                  << "                                        Accelerated CPU time                                                     " << std::endl
                  << "-----------------------------------------------------------------------------------------------------------------" << std::endl;
		print_stat_accel_CPU_line(info_coarse[0], info_fine[0], 0);
		for ( int level = 1; level < num_of_levels; level++ )
			print_stat_accel_CPU_line(info_coarse[level], info_fine[level], level);


		std::cout << "---------------------------------------------------------------------------------------------       --------------------------------------------------------------------------------------" << std::endl
                  << "---------------------------------------------------------------------------------------------       --------------------------------------------------------------------------------------" << std::endl
                  << "                                                                                                                                       init gues CPU time                                 " << std::endl
                  << "---------------------------------------------------------------------------------------------       --------------------------------------------------------------------------------------" << std::endl;
		std::cout << std::setw(6)  << 0 << delimeter
				  << std::setw(83) << " "
				  // accelerated
				  << std::setw(9) << " "
				  << std::setw(6) << " " << delimeter
			      << std::setw(6) << " " << delimeter
				  << std::setw(7) << " " << delimeter
				  << std::setw(7) << " " << delimeter
				  << std::setw(7) << " " << delimeter
				  << std::setw(6) << info_coarse[0].min_kdtree_init_guess_CPU_time  << delimeter
			      << std::setw(6) << info_coarse[0].max_kdtree_init_guess_CPU_time  << delimeter
				  << std::setw(7) << info_coarse[0].mean_kdtree_init_guess_CPU_time << delimeter
				  << std::setw(7) << info_coarse[0].sum_kdtree_init_guess_CPU_time
				  << std::endl;
		for ( int level = 1; level < num_of_levels; level++ )
		{
			std::cout << std::setw(6) << level << delimeter
					  << std::setw(83) << " "
					  // accelerated
					  << std::setw(9) << " "
					  << std::setw(6) << info_coarse[level].min_kdtree_init_guess_CPU_time  << delimeter
				      << std::setw(6) << info_coarse[level].max_kdtree_init_guess_CPU_time  << delimeter
					  << std::setw(7) << info_coarse[level].mean_kdtree_init_guess_CPU_time << delimeter
					  << std::setw(7) << info_coarse[level].sum_kdtree_init_guess_CPU_time  << delimeter
					  << std::setw(7) << " " << delimeter
					  << std::setw(6) << info_fine[level].min_kdtree_init_guess_CPU_time  << delimeter
				      << std::setw(6) << info_fine[level].max_kdtree_init_guess_CPU_time  << delimeter
					  << std::setw(7) << info_fine[level].mean_kdtree_init_guess_CPU_time << delimeter
					  << std::setw(7) << info_fine[level].sum_kdtree_init_guess_CPU_time
					  << std::endl;
		}
	}


	template <int dim>
	void MLMC<dim>::print_stat_accel_CPU_line(MC_info &info_coarse, MC_info &info_fine, int level)
	{
		if (level == 0)
		{
			std::cout	<< std::fixed; std::cout << std::setprecision(3)
						<< std::setw(6) << level << delimeter
						<< std::setw(6) << " " << delimeter
						<< std::setw(6) << " " << delimeter
						<< std::scientific << std::setprecision(5)
						<< std::setw(11) << " " << delimeter
						<< std::setw(11) << " " << delimeter
						<< std::setw(7) << "    " << delimeter
						<< std::fixed; std::cout << std::setprecision(3)
						<< std::setw(6) << info_coarse.accelerated_min_total_CPU_time << delimeter
						<< std::setw(6) << info_coarse.accelerated_max_total_CPU_time << delimeter
						<< std::scientific << std::setprecision(5)
						<< std::setw(11) << info_coarse.accelerated_mean_total_CPU_time << delimeter
						<< std::setw(11) << info_coarse.accelerated_mean_sum_CPU_time
						<< std::endl;
		}
		else
		{
			std::cout	<< std::fixed; std::cout << std::setprecision(3)
						<< std::setw(6) << level << delimeter
						<< std::setw(6) << info_coarse.accelerated_min_total_CPU_time << delimeter
						<< std::setw(6) << info_coarse.accelerated_max_total_CPU_time << delimeter
						<< std::scientific << std::setprecision(5)
						<< std::setw(11) << info_coarse.accelerated_mean_total_CPU_time << delimeter
						<< std::setw(11) << info_coarse.accelerated_mean_sum_CPU_time << delimeter
						<< std::setw(7) << "    " << delimeter
						<< std::fixed; std::cout << std::setprecision(3)
						<< std::setw(6) << info_fine.accelerated_min_total_CPU_time << delimeter
						<< std::setw(6) << info_fine.accelerated_max_total_CPU_time << delimeter
						<< std::scientific << std::setprecision(5)
						<< std::setw(11) << info_fine.accelerated_mean_total_CPU_time << delimeter
						<< std::setw(11) << info_fine.accelerated_mean_sum_CPU_time
						<< std::endl;
		}
	}


	template <int dim>
	void MLMC<dim>::print_stat_CPU_line(MC_info &info_coarse, MC_info &info_fine, int level)
	{
		if (level == 0)
		{
			std::cout	<< std::fixed; std::cout << std::setprecision(3)
						<< std::setw(6) << level << delimeter
						<< std::setw(6) << " " << delimeter
						<< std::setw(6) << " " << delimeter
						<< std::scientific << std::setprecision(5)
						<< std::setw(11) << " " << delimeter
						<< std::setw(11) << " " << delimeter
						<< std::setw(7) << "    " << delimeter
						<< std::fixed; std::cout << std::setprecision(3)
						<< std::setw(6) << info_coarse.zero_guess_min_total_CPU_time << delimeter
						<< std::setw(6) << info_coarse.zero_guess_max_total_CPU_time << delimeter
						<< std::scientific << std::setprecision(5)
						<< std::setw(11) << info_coarse.zero_guess_mean_total_CPU_time << delimeter
						<< std::setw(11) << info_coarse.zero_guess_mean_sum_CPU_time
						<< std::endl;
		}
		else
		{
			std::cout	<< std::fixed; std::cout << std::setprecision(3)
						<< std::setw(6) << level << delimeter
						<< std::setw(6) << info_coarse.zero_guess_min_total_CPU_time << delimeter
						<< std::setw(6) << info_coarse.zero_guess_max_total_CPU_time << delimeter
						<< std::scientific << std::setprecision(5)
						<< std::setw(11) << info_coarse.zero_guess_mean_total_CPU_time << delimeter
						<< std::setw(11) << info_coarse.zero_guess_mean_sum_CPU_time << delimeter
						<< std::setw(7) << "    " << delimeter
						<< std::fixed; std::cout << std::setprecision(3)
						<< std::setw(6) << info_fine.zero_guess_min_total_CPU_time << delimeter
						<< std::setw(6) << info_fine.zero_guess_max_total_CPU_time << delimeter
						<< std::scientific << std::setprecision(5)
						<< std::setw(11) << info_fine.zero_guess_mean_total_CPU_time << delimeter
						<< std::setw(11) << info_fine.zero_guess_mean_sum_CPU_time
						<< std::endl;
		}
	}

} /* namespace MonteCarlo */


#endif /* MLMC_H_ */


