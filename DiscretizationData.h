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

/* TODO: obsolete */
#include <deal.II/lac/sparsity_pattern.h>					// final  "static" sparsity pattern --> SparsityPattern

#include <deal.II/lac/block_sparsity_pattern.h>				// final  "static" block sparsity pattern --> BlockSparsityPattern
#include <deal.II/lac/compressed_sparsity_pattern.h>		// intermediate "dynamic" form of the SparsityPattern class --> CompressedSparsityPattern
#include <deal.II/lac/constraint_matrix.h>					// linear constraints on degrees of freedom (in our case - only boundary conditions)

#include <deal.II/grid/grid_generator.h>					// functions to generate standard grids
#include <deal.II/grid/tria.h>								// Triangulation class
#include <deal.II/grid/tria_boundary_lib.h>					// objects describing the boundary shapes
#include <deal.II/grid/intergrid_map.h>						// a map between two grids which are derived from the same coarse grid
#include <deal.II/grid/grid_in.h>							// input for grid data

#include <deal.II/dofs/dof_handler.h>						// manage distribution and numbering of the degrees of freedom and offer iterators
#include <deal.II/dofs/dof_tools.h>							// collection of functions operating on and manipulating the numbers of degrees of freedom --> make_sparsity_pattern
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>								// Implementation of a scalar Lagrange finite element --> FE_Q<dim>
#include <deal.II/fe/fe_system.h>							// Implementation of a vector-valued finite element -->  FESystem<dim>

#include <deal.II/numerics/vector_tools.h>					// VectorTools::interpolate_boundary_values


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
			static std::vector<BlockSparsityPattern*>				sparsity_patterns_ptr;
			static std::vector<ConstraintMatrix*>					hanging_nodes_constraints_ptr;
			static FESystem<dim>*									fe;
			static QGauss<dim>*										quadrature_formula;					// Gauss-Legendre quadrature with 2 quadrature points (in each space direction)
			static std::vector<std::vector<types::global_dof_index>>	dofs_per_block;

		private:
			static void make_grid( int level, int init_level );
			static void setup_system ( int level );
	};

	// instantiation of template static member vectors
	template<int dim>	std::vector<Triangulation<dim>*>							DiscretizationData<dim> :: triangulations_ptr;
	template<int dim>	std::vector<DoFHandler<dim>*>								DiscretizationData<dim> :: dof_handlers_ptr;
	template<int dim>	std::vector<std::vector<InterGridMap<DoFHandler<dim>>*>>	DiscretizationData<dim> :: intergrid_maps_ptr;
	template<int dim>	std::vector<BlockSparsityPattern*>							DiscretizationData<dim> :: sparsity_patterns_ptr;
	template<int dim>	std::vector<ConstraintMatrix*>								DiscretizationData<dim> :: hanging_nodes_constraints_ptr;
	template<int dim>	FESystem<dim>*												DiscretizationData<dim> :: fe					= new FESystem<dim>( FE_Q<dim>(2), dim, FE_Q<dim>(1), 1);
	template<int dim>	QGauss<dim>*												DiscretizationData<dim> :: quadrature_formula	= new QGauss<dim>(3);
	template<int dim>	std::vector<std::vector<types::global_dof_index>>			DiscretizationData<dim> :: dofs_per_block;
	template<int dim>	int 														DiscretizationData<dim> :: num_of_levels = 0;



/*---------------------------------------------------------------------------*/
/*                           Class implementation                            */
/*---------------------------------------------------------------------------*/


	// generate grids and dof_handlers for all levels of discretization
	template< int dim >
	void DiscretizationData<dim>::generate( int n_of_levels, int init_level )
	{
		for ( int lev = 0; lev < n_of_levels; lev++ )
			add_next_level(init_level);

		finalize(init_level);
	}


	template< int dim >
	void DiscretizationData<dim>::add_next_level( int init_level )
	{
		Triangulation<dim>*		new_triangulation				= new Triangulation<dim>;
		DoFHandler<dim>*		new_dof_handler					= new DoFHandler<dim>(*new_triangulation);
		BlockSparsityPattern*	new_sparsity_pattern			= new BlockSparsityPattern;
		ConstraintMatrix*		new_hanging_nodes_constraint	= new ConstraintMatrix;
		std::vector<types::global_dof_index> level_dofs_per_block(2);

		triangulations_ptr.push_back(new_triangulation);
		dof_handlers_ptr.push_back(new_dof_handler);
		sparsity_patterns_ptr.push_back(new_sparsity_pattern);
		hanging_nodes_constraints_ptr.push_back(new_hanging_nodes_constraint);
		dofs_per_block.push_back(level_dofs_per_block);

		make_grid( num_of_levels, init_level);
		setup_system ( num_of_levels );

		num_of_levels++;
	}


	template< int dim >
	void DiscretizationData<dim>::make_grid (int level, int init_level )
	{
		double aspect_ratio = 2.5;
		double diam = 1.0;
		double y_len = 40.0;

		std::vector<unsigned int> subdivisions(dim, 1);
		subdivisions[0] = aspect_ratio;
		subdivisions[1] = 1;

		const Point<dim> bottom_left = Point<dim>(0.0, 0.0);
		const Point<dim> top_right   = Point<dim>(aspect_ratio, 1.0);

//		GridGenerator::subdivided_hyper_rectangle ( *triangulations_ptr[level], subdivisions, bottom_left, top_right );
//		GridGenerator::hyper_cube_with_cylindrical_hole( *triangulations_ptr[level], 0.25, aspect_ratio );
//		GridGenerator::hyper_cube (*triangulations_ptr[level], 0, 1);
//		GridGenerator::hyper_L (triangulation, 0, 1);
//		GridGenerator::hyper_cube_with_cylindrical_hole (triangulation, 0.1, 0.5, 1, 1, false);
//		for ( typename Triangulation<dim>::active_cell_iterator cell = triangulations_ptr[level]->begin_active(); cell != triangulations_ptr[level]->end(); ++cell )
//			for ( unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f )
//			{
//				if ( cell->face(f)->center()[1] ==  1 )
//					cell->face(f)->set_all_boundary_indicators(1);
//			}



		// read grid from file
		GridIn<dim> grid_in;
		grid_in.attach_triangulation (*triangulations_ptr[level]);
		{
//			std::string filename = "nsbench2.inp";
//			std::string filename = "zigzag.inp";
//			std::string filename = "my_mesh.msh";
			std::string filename = "non_sym_flow.msh";
			std::ifstream file (filename.c_str());
			Assert (file, ExcFileNotOpen (filename.c_str()));
//			grid_in.read_ucd (file);
			grid_in.read_msh (file);
		}
		for ( typename Triangulation<dim>::active_cell_iterator cell = triangulations_ptr[level]->begin_active(); cell != triangulations_ptr[level]->end(); ++cell )
			for ( unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f )
			{
				if ( cell->face(f)->center()[0] == -15*diam )
					cell->face(f)->set_all_boundary_indicators(1);
				if ( cell->face(f)->center()[1] ==  y_len*diam || cell->face(f)->center()[1] == -y_len*diam )
					cell->face(f)->set_all_boundary_indicators(1);
				if ( cell->face(f)->center()[0] ==  50*diam )
					cell->face(f)->set_all_boundary_indicators(3);
			}


		// make grid
		if ( level == 0 )
			triangulations_ptr[level]->refine_global(init_level);
		else
			triangulations_ptr[level]->refine_global(level+init_level);

		// distribute degrees of freedom
		dof_handlers_ptr[level]->distribute_dofs (*fe);
		DoFRenumbering::Cuthill_McKee ( *dof_handlers_ptr[level] );

		// first dim components - velocities, the last component - pressure
		std::vector<unsigned int> block_component (dim+1,0);
		block_component[dim] = 1;
		DoFRenumbering::component_wise ( *dof_handlers_ptr[level], block_component );

	    DoFTools::count_dofs_per_block ( *DiscretizationData<dim>::dof_handlers_ptr[level], dofs_per_block[level], block_component);

		// compute constraints
		hanging_nodes_constraints_ptr[level]->clear();
			// constraints resulting from the presence of hanging nodes
			DoFTools::make_hanging_node_constraints( *dof_handlers_ptr[level], *hanging_nodes_constraints_ptr[level] );
			// constraints from boundary conditions for velocity components only ( required to make correct sparcity pattern )
			/* no need for these constraints since we apply boundary values after matrix creation */
		hanging_nodes_constraints_ptr[level]->close();
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
	void DiscretizationData<dim>::setup_system ( int level )
	{
	    const unsigned int n_u = dofs_per_block[level][0],
	                       n_p = dofs_per_block[level][1];

		// generate sparsity pattern of the matrix
		BlockCompressedSimpleSparsityPattern csp(2,2);

		csp.block(0,0).reinit(n_u, n_u);
		csp.block(1,0).reinit(n_p, n_u);
		csp.block(0,1).reinit(n_u, n_p);
		csp.block(1,1).reinit(n_p, n_p);

		csp.collect_sizes();

		DoFTools::make_sparsity_pattern ( *dof_handlers_ptr[level], csp, *hanging_nodes_constraints_ptr[level], true );
		sparsity_patterns_ptr[level]->copy_from(csp);
	}


} /* namespace MonteCarlo */

#endif /* DISCRETIZATIONDATA_H_ */
