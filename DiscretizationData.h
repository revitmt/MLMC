/*
 * MeshData.h
 *
 *  Created on: Dec 19, 2014
 *      Author: viktor
 */

#ifndef DISCRETIZATIONDATA_H_
#define DISCRETIZATIONDATA_H_

#include <iostream>
#include <fstream>

#include <deal.II/base/function.h>							// model for a general function --> ZeroFunction<dim>
#include <deal.II/base/quadrature_lib.h>					// Gauss-Legendre quadrature of arbitrary order --> QGauss<dim>

#include <deal.II/lac/sparsity_pattern.h>					// final  "static" sparsity pattern --> SparsityPattern
#include <deal.II/lac/compressed_sparsity_pattern.h>		// intermediate "dynamic" form of the SparsityPattern class --> CompressedSparsityPattern
#include <deal.II/lac/constraint_matrix.h>					// linear constraints on degrees of freedom (in our case - only boundary conditions)

#include <deal.II/grid/grid_generator.h>					// functions to generate standard grids
#include <deal.II/grid/tria.h>								// Triangulation class
#include <deal.II/grid/tria_boundary_lib.h>					// objects describing the boundary shapes
#include <deal.II/grid/intergrid_map.h>						// a map between two grids which are derived from the same coarse grid

#include <deal.II/dofs/dof_handler.h>						// manage distribution and numbering of the degrees of freedom and offer iterators
#include <deal.II/dofs/dof_tools.h>							// collection of functions operating on and manipulating the numbers of degrees of freedom --> make_sparsity_pattern

#include <deal.II/fe/fe_q.h>								// Implementation of a scalar Lagrange finite element --> FE_Q<dim>


using namespace dealii;


#define ASSERT fprintf(stderr,"File: %s \nLine number: %d \n\n", __FILE__ ,__LINE__);


namespace MonteCarlo
{

/*---------------------------------------------------------------------------*/
/*                           Class interface                                 */
/*---------------------------------------------------------------------------*/

	template< int dim >
	class DiscretizationData
	{
		public:
			DiscretizationData() {};
			virtual ~DiscretizationData() {};
			static void initialize();
			static void generate( int n_of_levels, int init_level = 2 );
			static void add_next_level( int init_level = 2 );
			static void finalize( int init_level = 1 );

		public:
			typedef InterGridMap<DoFHandler<dim>> intergrid_map_type;
			// I use vector of pointers because deal.II does not have copy constructor for DoFHandler class
			// which is required for the construction of vectors
			static int												num_of_levels;
			static std::vector<Triangulation<dim>*>					triangulations_ptr;
			static std::vector<DoFHandler<dim>*>					dof_handlers_ptr;
			static std::vector<std::vector<intergrid_map_type*>>	intergrid_maps_ptr;
			static std::vector<SparsityPattern*>					sparsity_patterns_ptr;
			static std::vector<ConstraintMatrix*>					hanging_nodes_constraints_ptr;
			static FE_Q<dim>*										fe;
			static QGauss<dim>*										quadrature_formula;					// Gauss-Legendre quadrature with 2 quadrature points (in each space direction)

		private:
			static void make_grid( int level, int init_level );
			static void setup_system ( int level );
	};

	// instantiation of template static member vectors
	template<int dim>	std::vector<Triangulation<dim>*>							DiscretizationData<dim> :: triangulations_ptr;
	template<int dim>	std::vector<DoFHandler<dim>*>								DiscretizationData<dim> :: dof_handlers_ptr;
	template<int dim>	std::vector<std::vector<InterGridMap<DoFHandler<dim>>*>>	DiscretizationData<dim> :: intergrid_maps_ptr;
	template<int dim>	std::vector<SparsityPattern*>								DiscretizationData<dim> :: sparsity_patterns_ptr;
	template<int dim>	std::vector<ConstraintMatrix*>								DiscretizationData<dim> :: hanging_nodes_constraints_ptr;
	template<int dim>	FE_Q<dim>*													DiscretizationData<dim> :: fe					= new FE_Q<dim>(1);
	template<int dim>	QGauss<dim>*												DiscretizationData<dim> :: quadrature_formula	= new QGauss<dim>(2);
	template<int dim>	int 														DiscretizationData<dim> :: num_of_levels = 0;



/*---------------------------------------------------------------------------*/
/*                           Class implementation                            */
/*---------------------------------------------------------------------------*/


	// generate grids and dof_handlers for all levels of discretization
	template< int dim >
	void DiscretizationData<dim>::generate( int n_of_levels, int init_level )
	{
		// check if the static members of the class have been generated previously
		// if so, free the memory
		// This part is tricky because of the Subscriptor class
		if ( fe != NULL )
		{
			for ( int level = 0; level < num_of_levels; level++ )
			{
				for ( int j = 0; j < num_of_levels; j++ )				// depends on dof_handlers, so has to be deleted first
					delete intergrid_maps_ptr[level][j];				//
				delete dof_handlers_ptr[level];							// depends on the triangulations & fe, so has to be deleted first
				delete triangulations_ptr[level];
				delete hanging_nodes_constraints_ptr[level];
				delete sparsity_patterns_ptr[level];
			}
			delete dof_handlers_ptr[num_of_levels];						// depends on the triangulations & fe, so has to be deleted first
			delete triangulations_ptr[num_of_levels];
			delete hanging_nodes_constraints_ptr[num_of_levels];
			delete fe;
			delete quadrature_formula;
		}

		num_of_levels = n_of_levels;

		// resize vectors given number of levels
		triangulations_ptr.resize(num_of_levels+1);
		dof_handlers_ptr.resize(num_of_levels+1);
		sparsity_patterns_ptr.resize(num_of_levels);
		hanging_nodes_constraints_ptr.resize(num_of_levels+1);
		intergrid_maps_ptr.resize(num_of_levels);
		for ( int level = 0; level < num_of_levels; level++ )
			intergrid_maps_ptr[level].resize(num_of_levels);

		// actually generate grid data
		fe = new FE_Q<dim>(1);
		quadrature_formula = new QGauss<dim>(2);
		for ( int level = 0; level < num_of_levels; level++ )
		{
			triangulations_ptr[level]				= new Triangulation<dim>;
			dof_handlers_ptr[level]					= new DoFHandler<dim> (*triangulations_ptr[level]);
			sparsity_patterns_ptr[level]			= new SparsityPattern;
			hanging_nodes_constraints_ptr[level]	= new ConstraintMatrix;

			make_grid(level, init_level);
			setup_system (level);
		}

		// generate intergrid maps
		for ( int i = 0; i < num_of_levels; i++ )
			for ( int j = 0; j < num_of_levels; j++ )
			{
				intergrid_maps_ptr[i][j] =  new InterGridMap<DoFHandler<dim>>;
				intergrid_maps_ptr[i][j] -> make_mapping ( *dof_handlers_ptr[i], *dof_handlers_ptr[j] );
			}

		// make one additional very fine level for exact function plot
		triangulations_ptr[num_of_levels]				= new Triangulation<dim>;
		dof_handlers_ptr[num_of_levels]					= new DoFHandler<dim> (*triangulations_ptr[num_of_levels]);
		hanging_nodes_constraints_ptr[num_of_levels]	= new ConstraintMatrix;
		make_grid(num_of_levels, init_level);
	}


	template< int dim >
	void DiscretizationData<dim>::initialize()
	{
//		fe					= new FE_Q<dim>(1);
//		quadrature_formula	= new QGauss<dim>(2)
	}


	template< int dim >
	void DiscretizationData<dim>::add_next_level( int init_level )
	{
		Triangulation<dim>*	new_triangulation				= new Triangulation<dim>;
		DoFHandler<dim>*	new_dof_handler					= new DoFHandler<dim>(*new_triangulation);
		SparsityPattern*	new_sparsity_pattern			= new SparsityPattern;
		ConstraintMatrix*	new_hanging_nodes_constraint	= new ConstraintMatrix;

		triangulations_ptr.push_back(new_triangulation);
		dof_handlers_ptr.push_back(new_dof_handler);
		sparsity_patterns_ptr.push_back(new_sparsity_pattern);
		hanging_nodes_constraints_ptr.push_back(new_hanging_nodes_constraint);

		make_grid( num_of_levels, init_level);
		setup_system ( num_of_levels );

		num_of_levels++;
	}


	template< int dim >
	void DiscretizationData<dim>::finalize( int init_level )
	{
		intergrid_maps_ptr.resize(num_of_levels);
		for ( int level = 0; level < num_of_levels; level++ )
			intergrid_maps_ptr[level].resize(num_of_levels);

		// generate intergrid maps
		for ( int i = 0; i < num_of_levels; i++ )
			for ( int j = 0; j < num_of_levels; j++ )
			{
				intergrid_maps_ptr[i][j] =  new InterGridMap<DoFHandler<dim>>;
				intergrid_maps_ptr[i][j] -> make_mapping ( *dof_handlers_ptr[i], *dof_handlers_ptr[j] );
			}

		// make one additional very fine level for exact function plot
//		Triangulation<dim>*	new_triangulation				= new Triangulation<dim>;
//		DoFHandler<dim>*	new_dof_handler					= new DoFHandler<dim>(*new_triangulation);
//		ConstraintMatrix*	new_hanging_nodes_constraint	= new ConstraintMatrix;
//		triangulations_ptr.push_back(new_triangulation);
//		dof_handlers_ptr.push_back(new_dof_handler);
//		hanging_nodes_constraints_ptr.push_back(new_hanging_nodes_constraint);
//		make_grid(num_of_levels, init_level);
	}


	template< int dim >
	void DiscretizationData<dim>::make_grid (int level, int init_level )
	{
		GridGenerator::hyper_cube (*triangulations_ptr[level], 0, 1);
//		GridGenerator::hyper_L (triangulation, 0, 1);
//		GridGenerator::hyper_cube_with_cylindrical_hole (triangulation, 0.1, 0.5, 1, 1, false);

//		triangulations_ptr[level]->begin_active()->face(2)->set_boundary_indicator(1);
//		triangulations_ptr[level]->begin_active()->face(3)->set_boundary_indicator(1);

		// make grid
		if ( level == 0 )
			triangulations_ptr[level]->refine_global(init_level);
		else
			triangulations_ptr[level]->refine_global(level+init_level);

		// distribute degrees of freedom
		dof_handlers_ptr[level]->distribute_dofs (*fe);

//		// compute constraints resulting from the presence of hanging nodes
//		hanging_nodes_constraints_ptr[level]->clear();
//		DoFTools::make_hanging_node_constraints (*dof_handlers_ptr[level], *hanging_nodes_constraints_ptr[level]);
	}


	template< int dim >
	void DiscretizationData<dim>::setup_system ( int level )
	{
		// generate sparsity pattern of the matrix
		CompressedSparsityPattern c_sparsity(dof_handlers_ptr[level]->n_dofs());

		DoFTools::make_sparsity_pattern (*dof_handlers_ptr[level], c_sparsity);
		sparsity_patterns_ptr[level]->copy_from(c_sparsity);
	}


} /* namespace MonteCarlo */

#endif /* DISCRETIZATIONDATA_H_ */
