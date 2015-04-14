#ifndef BOUSSINESQ_FLOW_PROBLEM 
#define BOUSSINESQ_FLOW_PROBLEM

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/trilinos_block_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <limits>
#include <locale>
#include <string>

#include <deal.II/distributed/solution_transfer.h>

#include <deal.II/base/index_set.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include "equation_data.h"
#include "linear_solver.h"
#include "assembly.h"

#include "parsed_grid_generator.h"
#include "parsed_finite_element.h"
#include "utilities.h"
#include "parsed_function.h"
//#include "boussinesq_flow_problem.h"

using namespace dealii;

template <int dim>
class BoussinesqFlowProblem
{
	public:
		struct Parameters;
		BoussinesqFlowProblem (Parameters &parameters);
      ~BoussinesqFlowProblem ();

		void run ();

	private:
		void make_grid_fe();
		void setup_dofs ();
		void assemble_stokes_preconditioner ();
		void build_stokes_preconditioner ();
		void assemble_stokes_system ();
		void solve ();
		void output_results ();
		void refine_mesh (const unsigned int max_grid_level);

	public:

		struct Parameters
		{
			Parameters (const std::string &parameter_filename);

			static void declare_parameters (ParameterHandler &prm);
			void parse_parameters (ParameterHandler &prm);

			double       end_time;

			unsigned int initial_global_refinement;
			unsigned int initial_adaptive_refinement;

			bool         generate_graphical_output;
			unsigned int graphical_output_interval;

			unsigned int adaptive_refinement_interval;

			double       stabilization_alpha;
			double       stabilization_c_R;
			double       stabilization_beta;

			unsigned int stokes_velocity_degree;
			bool         use_locally_conservative_discretization;

		};

	private:
		Parameters                               &parameters;

		ConditionalOStream                        pcout;

      SmartPointer<parallel::distributed::Triangulation<dim> > triangulation;

		double                                    global_Omega_diameter;

		const MappingQ<dim>                       mapping;

      SmartPointer<FiniteElement<dim,dim> >     stokes_fe;

      SmartPointer<DoFHandler<dim> >            stokes_dof_handler;

		ConstraintMatrix                          stokes_constraints;

		TrilinosWrappers::BlockSparseMatrix       stokes_matrix;
		TrilinosWrappers::BlockSparseMatrix       stokes_preconditioner_matrix;

      TrilinosWrappers::MPI::BlockVector        stokes_solution;
		TrilinosWrappers::MPI::BlockVector        old_stokes_solution;
		TrilinosWrappers::MPI::BlockVector        stokes_rhs;

		double                                    time_step;
		double                                    old_time_step;
		unsigned int                              timestep_number;

		std_cxx11::shared_ptr<TrilinosWrappers::PreconditionAMG>    Amg_preconditioner;
		std_cxx11::shared_ptr<TrilinosWrappers::PreconditionJacobi> Mp_preconditioner;
		std_cxx11::shared_ptr<TrilinosWrappers::PreconditionJacobi> T_preconditioner;

		bool                                      rebuild_stokes_matrix;
		bool                                      rebuild_stokes_preconditioner;

		TimerOutput                               computing_timer;

		void setup_stokes_matrix (const std::vector<IndexSet> &stokes_partitioning,
									const std::vector<IndexSet> &stokes_relevant_partitioning);
		void setup_stokes_preconditioner (const std::vector<IndexSet> &stokes_partitioning,
											const std::vector<IndexSet> &stokes_relevant_partitioning);


		void
		local_assemble_stokes_preconditioner (const typename DoFHandler<dim>::active_cell_iterator &cell,
												Assembly::Scratch::StokesPreconditioner<dim> &scratch,
												Assembly::CopyData::StokesPreconditioner<dim> &data);

		void
		copy_local_to_global_stokes_preconditioner (const Assembly::CopyData::StokesPreconditioner<dim> &data);


		void
		local_assemble_stokes_system (const typename DoFHandler<dim>::active_cell_iterator &cell,
										Assembly::Scratch::StokesSystem<dim>  &scratch,
										Assembly::CopyData::StokesSystem<dim> &data);

		void
		copy_local_to_global_stokes_system (const Assembly::CopyData::StokesSystem<dim> &data);

		class Postprocessor;

      ParsedGridGenerator<dim,dim> pgg;

      ParsedFiniteElement<dim,dim> fe_builder;

      ParsedFunction<dim, dim+1> boundary_conditions;

      ParsedFunction<dim, dim+1> right_hand_side;
};

#endif
