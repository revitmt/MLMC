/*
 * Problem.h
 *
 *  Created on: Dec 13, 2014
 *      Author: viktor
 */

#ifndef PROBLEM_H_
#define PROBLEM_H_

#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/tensor_base.h>
//#include <deal.II/base/timer.h>					// CPU time


#include <deal.II/lac/full_matrix.h>			// classical rectangular matrix
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_relaxation.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/dofs/dof_accessor.h>						// access information related to dofs while iterating trough cells, faces, etc --> get_dof_indices ( local_dof_indices )

#include <deal.II/fe/fe_values.h>				// finite element functions evaluated in quadrature points of a cell --> FEValues<dim>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/base/work_stream.h>
#include <deal.II/base/multithread_info.h>

//#include "DiscretizationData.h"
#include "DataTypes.h"
#include "InputData.h"
#include "Output.h"

#include "my_timer.h"

using namespace dealii;


namespace deterministic_solver
{

	enum Grid_Resolution { fine, coarse, regular };

	struct lin_solver_info // must be class
	{
		int		zero_guess_iterations = 0;
		int		accelerated_iterations = 0;
		double	zero_guess_total_CPU_time = 0.0;
		double	accelerated_total_CPU_time = 0.0;
		double	brute_init_guess_CPU_time = 0.0;
		double	kdtree_init_guess_CPU_time = 0.0;
		double	assemble_CPU_time = 0.0;
		double	solve_CPU_time = 0.0;
	};

/*---------------------------------------------------------------------------*/
/*                           Class interface                                 */
/*---------------------------------------------------------------------------*/


	template <int dim>
	class Problem
	{
		public:
			struct ScratchData
			{
				std::vector<Vector<double>>	function_values;
				FEValues<dim>	 	fe_values;
				ScratchData () : function_values( n_q_points, Vector<double>(dim+1) ), fe_values ( *fe, *quadrature_formula, update_values | update_gradients | update_quadrature_points | update_JxW_values  ) {};
				ScratchData (const ScratchData &scratch) : function_values(scratch.function_values),
															 fe_values(	scratch.fe_values.get_fe(),
																		scratch.fe_values.get_quadrature(),
																		scratch.fe_values.get_update_flags()) {}
			};
			struct PerTaskRhsData
			{
				Vector<double>            cell_rhs;
				std::vector<unsigned int> local_dof_indices;
				PerTaskRhsData ( const int &dof_per_cell ) : cell_rhs(dof_per_cell), local_dof_indices(dof_per_cell){};
			};
			struct PerTaskMatrixData
			{
				FullMatrix<double>        cell_matrix;
				std::vector<unsigned int> local_dof_indices;
				PerTaskMatrixData ( const int &dof_per_cell ) : cell_matrix(dof_per_cell,dof_per_cell), local_dof_indices(dof_per_cell){};
			};

		public:
			Problem() {};
			Problem ( Grid_Resolution isfine, int level, double lin_sol_error );
			virtual ~Problem() {};
			static void initialize();
			void add_next_level_data();
			void reinit ( Grid_Resolution isfine, int level, double lin_sol_error );
			void run(	Coefficient<dim>	*coefficient_function,
						BoundaryValues<dim> *bound_val_function,
						RightHandSide<dim>  *RHS_function,
						solution_type<dim>	&solution);
			solution_type<dim> get_init_guess();
			lin_solver_info get_solver_info();

		private:
			my_timer 			timer;
			int					level;
			int 				iterations;
			Grid_Resolution		is_fine;
			double				lin_sol_err;
			static FESystem<dim>	*fe;
			static QGauss<dim>		*quadrature_formula;
			static unsigned int	n_q_points;							// number of quadrature points per cell
			static unsigned int	dofs_per_cell;						// number of degrees of freedom per cell
//			static std::vector<solution_type<dim>*>	rhs;
//			static std::vector<ConstraintMatrix*>	constraints;
//			static RightHandSide<dim> rhs_function;
//			static solution_type<dim>* curr_rhs;
			RightHandSide<dim>	*rhs_function;
			Coefficient<dim>	*coefficient_function;
			BoundaryValues<dim> *BV_function;
			ConstraintMatrix	constraints;
			solution_type<dim>	rhs;
			matrix_type<dim> 	operator_matrix;
			matrix_type<dim> 	system_matrix;
			solution_type<dim>	solution_init_guess;
			solution_type<dim>	solution;							// must be accurate with this
			lin_solver_info		info;
			static void compute_constraints (	DoFHandler<dim>		&dof_handler,
												BoundaryValues<dim> &BV_function,
												ConstraintMatrix	&constraint);
			static void assemble_rhs (	DoFHandler<dim>		&dof_handler,
										RightHandSide<dim>	&rhs_function,
										solution_type<dim>	&rhs);
			void assemble_rhs(	DoFHandler<dim>		&dof_handler );
			void assemble_rhs_on_one_cell (	const typename DoFHandler<dim>::active_cell_iterator	&cell,
															ScratchData											&common_data,
															PerTaskRhsData											&data);
			void rhs_copy_local_to_global (const PerTaskRhsData &data);
			void assemble_matrix_on_one_cell (	const typename DoFHandler<dim>::active_cell_iterator	&cell,
															ScratchData											&common_data,
															PerTaskMatrixData											&data);
			void matrix_copy_local_to_global (const PerTaskMatrixData &data);
			void assemble_matrix (	DoFHandler<dim>		&dof_handler,
									Coefficient<dim>	&coefficient_function,
									matrix_type<dim>	&operator_matrix);
			void assemble_matrix (	DoFHandler<dim>		&dof_handler );
			void apply_boundary_values (ConstraintMatrix	&constraint,
										matrix_type<dim>	&matrix );
			void init_guess( solution_type<dim>	&solution, std::vector<double> parameter );
			void solve (	solution_type<dim> &solution,
							double lin_solver_error );
	};

//	template<int dim>	std::vector<solution_type<dim>*>	Problem<dim>::rhs;
//	template<int dim>	solution_type<dim>*					Problem<dim>::curr_rhs;
//	template<int dim>	std::vector<ConstraintMatrix*>		Problem<dim>::constraints;
	template<int dim>	FESystem<dim>*						Problem<dim>::fe					= DiscretizationData<dim>::fe;
	template<int dim>	QGauss<dim>*						Problem<dim>::quadrature_formula	= DiscretizationData<dim>::quadrature_formula;
	template<int dim>	unsigned int						Problem<dim>::n_q_points;
	template<int dim>	unsigned int						Problem<dim>::dofs_per_cell;
//	template<int dim>	RightHandSide<dim>					Problem<dim>::rhs_function;



/*---------------------------------------------------------------------------*/
/*                           Class implementation                            */
/*---------------------------------------------------------------------------*/


	template <int dim>
	Problem<dim>::Problem (	Grid_Resolution isfine,
							int lev,
							double lin_sol_error )
	{
		level = lev;
		is_fine = isfine;
		lin_sol_err = lin_sol_error;
	}


	template <int dim>
	void Problem<dim>::reinit (	Grid_Resolution isfine,
								int lev,
								double lin_sol_error )
	{
		level = lev;
		is_fine = isfine;
		lin_sol_err = lin_sol_error;
	}


	template <int dim>
	void Problem<dim>::initialize ( )
	{
		dofs_per_cell		= fe->dofs_per_cell;
		n_q_points			= quadrature_formula->size();
//		rhs_function 		= _rhs_function;
	}


	// TODO: obsolete
	template <int dim>
	void Problem<dim>::add_next_level_data ()
	{
//		int lev = DiscretizationData<dim>::num_of_levels - 1;
//
//		DoFHandler<dim>	*dof_handler = DiscretizationData<dim>::dof_handlers_ptr[lev];
//
//		// TODO: obsolete
//		ScratchData	fe_values;
//
//		/* compute constraints */
//		BoundaryValues<dim> BV_function;
//		ConstraintMatrix	*new_constraints = DiscretizationData<dim>::hanging_nodes_constraints_ptr[lev];
//		compute_constraints ( *dof_handler,	BV_function, *new_constraints);
//		constraints.push_back( new_constraints );
//
//
//		/* assemble rhs */
//		curr_rhs = new solution_type<dim>;	curr_rhs->reinit(lev);
//		assemble_rhs ( *dof_handler );
//		rhs.push_back( curr_rhs );
	}






	template <int dim>
	void Problem<dim>::run(	Coefficient<dim>	*coeff_function,
							BoundaryValues<dim> *bound_val_function,
							RightHandSide<dim>  *RHS_function,
							solution_type<dim>	&sol)
	{
		DoFHandler<dim>	*dof_handler = DiscretizationData<dim>::dof_handlers_ptr[level];

		coefficient_function = coeff_function;
		BV_function = bound_val_function;
		rhs_function = RHS_function;

		solution.reinit(sol.level);
		solution = sol;

		rhs.reinit(sol.level);

		timer.tic();
			assemble_matrix ( *dof_handler );
			assemble_rhs ( *dof_handler );
			compute_constraints ( *dof_handler,	*BV_function, constraints );
			apply_boundary_values (	constraints, operator_matrix );
//			apply_boundary_values (	*(constraints[level]), operator_matrix );
		info.assemble_CPU_time = timer.toc();

		if ( is_fine == regular )
		{
			sol.vector = 0.0;
			solution_init_guess = sol;

			timer.tic();
				solve ( sol, lin_sol_err );
			info.solve_CPU_time = timer.toc();

			info.zero_guess_iterations = iterations;
			info.zero_guess_total_CPU_time = info.solve_CPU_time;

			info.accelerated_iterations = iterations;
			info.accelerated_total_CPU_time = info.solve_CPU_time; // + info.kdtree_init_guess_CPU_time;
		}
		else if ( is_fine == coarse )
		{
			// solve with zero guess
			sol.vector = 0.0;

			timer.tic();
				solve ( sol, lin_sol_err );
			info.solve_CPU_time = timer.toc();

			info.zero_guess_iterations = iterations;
			info.zero_guess_total_CPU_time = info.solve_CPU_time;

			// solve with acceleration
			init_guess( sol, coefficient_function->get_coefficients() );

			timer.tic();
				solve ( sol, lin_sol_err );
			info.solve_CPU_time = timer.toc();

			info.accelerated_iterations = iterations;
			info.accelerated_total_CPU_time = info.solve_CPU_time; // + info.kdtree_init_guess_CPU_time;
		}
		else // fine
		{
			timer.tic();
				solve ( sol, lin_sol_err );
			info.solve_CPU_time = timer.toc();

			info.zero_guess_iterations = iterations;
			info.zero_guess_total_CPU_time = info.solve_CPU_time;

			info.accelerated_iterations = iterations;
			info.accelerated_total_CPU_time = info.solve_CPU_time;
		}


	}






/*------------------------------------RHS------------------------------------*/

	template <int dim>
	void Problem<dim>::assemble_rhs( DoFHandler<dim>		&dof_handler )
	{
		WorkStream::run(	dof_handler.begin_active(),
							dof_handler.end(),
							*this,
							&Problem<dim>::assemble_rhs_on_one_cell,
							&Problem<dim>::rhs_copy_local_to_global,
							ScratchData(),
							PerTaskRhsData(dofs_per_cell));
	}


	template <int dim>
	void Problem<dim>::assemble_rhs_on_one_cell (	const typename DoFHandler<dim>::active_cell_iterator	&cell,
													ScratchData												&scratch_data,
													PerTaskRhsData											&private_data)
	{
		// get fe info at a current cell
		scratch_data.fe_values.reinit(cell);

		const FEValuesExtractors::Vector velocities(0);

	    std::vector<Tensor<1,dim> >  phi_u      (dofs_per_cell);

	    std::vector<Tensor<1,dim> > old_velocity_values (n_q_points);
	    std::vector<Tensor<2,dim> > old_velocity_gradients(n_q_points);

	    // calculate values of the solution from previous Newton iteration
	    // in the quadrature points of the current cell
	    scratch_data.fe_values[velocities].get_function_values    ( solution.vector, old_velocity_values );
	    scratch_data.fe_values[velocities].get_function_gradients ( solution.vector, old_velocity_gradients );


		// initialize local rhs vector
		private_data.cell_rhs = 0.0;

		// values of rhs at quadrature points
		rhs_function->vector_value_list ( scratch_data.fe_values.get_quadrature_points(), scratch_data.function_values );

		// assemble local contributions
		for ( unsigned int q_point = 0; q_point < n_q_points; ++q_point )
		{
			for (unsigned int k = 0; k < dofs_per_cell; ++k)
				phi_u[k]      = scratch_data.fe_values[velocities].value(k, q_point);
			for ( unsigned int i = 0; i < dofs_per_cell; ++i )
			{
				// find nonzero component of the FE vector, i.e. which variable corresponds to the current degree of freedom
				const unsigned int component_i = fe->system_to_component_index(i).first;
				private_data.cell_rhs(i) += (	scratch_data.fe_values.shape_value(i,q_point) *
												scratch_data.function_values[q_point][component_i]
												+
												( old_velocity_gradients[q_point] * old_velocity_values[q_point] ) * phi_u[i] // ( old_velocity_gradients[q_point] * old_velocity_values[q_point] ) can be calculated outside i loop
											) * scratch_data.fe_values.JxW(q_point) ;
			}
		}

		// global indexes corresponding to local nodes
		cell->get_dof_indices ( private_data.local_dof_indices );
	}


	template <int dim>
	void Problem<dim>::rhs_copy_local_to_global ( const PerTaskRhsData &data )
	{
		// add local rhs  to global
		for ( unsigned int i = 0; i < dofs_per_cell; ++i )
			//curr_rhs->vector( data.local_dof_indices[i] ) += data.cell_rhs(i);
			rhs.vector[data.local_dof_indices[i]] += data.cell_rhs[i];
	}







/*---------------------------------MATRIX------------------------------------*/

	template <int dim>
	void Problem<dim>::assemble_matrix( DoFHandler<dim>		&dof_handler )
	{
		operator_matrix.reinit(level);

		WorkStream::run(	dof_handler.begin_active(),
							dof_handler.end(),
							*this,
							&Problem<dim>::assemble_matrix_on_one_cell,
							&Problem<dim>::matrix_copy_local_to_global,
							ScratchData(),
							PerTaskMatrixData(dofs_per_cell));
	}


	template <int dim>
	void Problem<dim>::assemble_matrix_on_one_cell (	const typename DoFHandler<dim>::active_cell_iterator	&cell,
														ScratchData												&common_data,
														PerTaskMatrixData										&data)
	{
		// get fe info at a current cell
		common_data.fe_values.reinit(cell);

		const FEValuesExtractors::Vector velocities(0);
		const FEValuesExtractors::Scalar pressure(dim);

	    std::vector<Tensor<1,dim> >  phi_u      (dofs_per_cell);
	    std::vector<Tensor<2,dim> >  grad_phi_u (dofs_per_cell);
	    std::vector<double>          div_phi_u  (dofs_per_cell);
	    std::vector<double>          phi_p      (dofs_per_cell);

	    std::vector<Tensor<1,dim>> old_velocity_values (n_q_points);
	    std::vector<Tensor<2,dim>> old_velocity_gradients(n_q_points);

	    // calculate values of the solution from previous Newton iteration
	    // at quadrature points of the current cell
	    common_data.fe_values[velocities].get_function_values    ( solution.vector, old_velocity_values );
	    common_data.fe_values[velocities].get_function_gradients ( solution.vector, old_velocity_gradients );

		// initialize local matrix
		data.cell_matrix = 0.0;

		// values of coefficient at quadrature points
		coefficient_function->vector_value_list( common_data.fe_values.get_quadrature_points(), common_data.function_values);

		// assemble local contributions
		for ( unsigned int q_point = 0; q_point < n_q_points; ++q_point )
		{
			for (unsigned int k = 0; k < dofs_per_cell; ++k)
			{
				phi_u[k]      = common_data.fe_values[velocities].value(k, q_point);
				grad_phi_u[k] = common_data.fe_values[velocities].gradient(k, q_point);
				div_phi_u[k]  = common_data.fe_values[velocities].divergence(k, q_point);
				phi_p[k]      = common_data.fe_values[pressure].value(k, q_point);
			}
			for ( unsigned int i = 0; i < dofs_per_cell; ++i )
				for ( unsigned int j = 0; j < dofs_per_cell; ++j )
					data.cell_matrix(i,j) += (	common_data.function_values[q_point][0] * scalar_product( grad_phi_u[j], grad_phi_u[i] )
												- div_phi_u[j] * phi_p[i]
												- div_phi_u[i] * phi_p[j]
												+ ( old_velocity_gradients[q_point] * phi_u[j]   ) * phi_u[i]
												+ ( grad_phi_u[j] * old_velocity_values[q_point] ) * phi_u[i]
												+ phi_p[j] * phi_p[i] )
												* common_data.fe_values.JxW(q_point);
		}

		// global indexes corresponding to local nodes
		cell->get_dof_indices (data.local_dof_indices);
	}


	template <int dim>
	void Problem<dim>::matrix_copy_local_to_global ( const PerTaskMatrixData &data )
	{
		for ( unsigned int i = 0; i < dofs_per_cell; ++i )
			for ( unsigned int j = 0; j < dofs_per_cell; ++j )
				operator_matrix.matrix.add (	data.local_dof_indices[i],
												data.local_dof_indices[j],
												data.cell_matrix(i,j) );
	}








/*-----------------------------BOUNDARY VALUES-------------------------------*/

	template <int dim>
	void Problem<dim>::compute_constraints (	DoFHandler<dim>		&dof_handler,
												BoundaryValues<dim> &BV_function,
												ConstraintMatrix	&constraint)
	{
		// boundary conditions for velocity components only
		FEValuesExtractors::Vector velocities(0);

		// compute constraints resulting from the presence of hanging nodes
		constraint.clear();
		DoFTools::make_hanging_node_constraints ( dof_handler, constraint );

		// compute Dirichlet boundary conditions
		VectorTools::interpolate_boundary_values (	dof_handler,
													0,
													BV_function,
													constraint,
													DiscretizationData<dim>::fe->component_mask(velocities));

		constraint.close(); // close the filling of entries for constraint matrix
	}


	template <int dim>
	void Problem<dim>::apply_boundary_values (	ConstraintMatrix	&constraint,
												matrix_type<dim>	&matrix )
	{
		system_matrix.reinit(level);
		system_matrix.matrix.copy_from(operator_matrix.matrix);
		constraint.condense( system_matrix.matrix);
	}






	template <int dim>
	void Problem<dim>::init_guess ( solution_type<dim> &solution, std::vector<double> parameter )
	{
		timer.tic();
//			allSolutions<dim>::find_closest( parameter, solution );
			solution.vector = 0.0;
		info.brute_init_guess_CPU_time = timer.toc();

		info.kdtree_init_guess_CPU_time = info.brute_init_guess_CPU_time;

		solution_init_guess = solution;
	}


	template <int dim>
	void Problem<dim>::solve (	solution_type<dim> &solution,
								double lin_sol_err )
	{
		system_matrix.matrix.block(1,1) = 0.0;

		solution = rhs;
		SparseDirectUMFPACK direct_solver;
		direct_solver.solve(system_matrix.matrix, solution.vector);

//		SolverControl		solver_control (10000, lin_sol_err );
//		SolverCG<>			solver (solver_control);
//		PreconditionSSOR<>	preconditioner;
//
//		preconditioner.initialize(system_matrix.matrix, 1.2);
//
////		solver.solve (system_matrix.matrix, solution.vector, rhs[level]->vector, PreconditionIdentity());
//		solver.solve (system_matrix.matrix, solution.vector, rhs[level]->vector, preconditioner);
//
//		iterations = solver_control.last_step();
	}











	template <int dim>
	solution_type<dim> Problem<dim>::get_init_guess()
	{
		return solution_init_guess;
	}


	template <int dim>
	lin_solver_info Problem<dim>::get_solver_info()
	{
		return info;
	}

}//deterministic_solver

#endif /* PROBLEM_H_ */
