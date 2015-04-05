#ifndef BOUSSINESQ_FLOW_PROBLEM 
#define BOUSSINESQ_FLOW_PROBLEM

using namespace dealii;

template <int dim>
class BoussinesqFlowProblem
{
	public:
		struct Parameters;
		BoussinesqFlowProblem (Parameters &parameters);
		void run ();

	private:
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

			// unsigned int temperature_degree;
		};

	private:
		Parameters                               &parameters;

		ConditionalOStream                        pcout;

		parallel::distributed::Triangulation<dim> triangulation;
		double                                    global_Omega_diameter;

		const MappingQ<dim>                       mapping;

		const FESystem<dim>                       stokes_fe;
		DoFHandler<dim>                           stokes_dof_handler;
		ConstraintMatrix                          stokes_constraints;

		TrilinosWrappers::BlockSparseMatrix       stokes_matrix;
		TrilinosWrappers::BlockSparseMatrix       stokes_preconditioner_matrix;

		TrilinosWrappers::MPI::BlockVector        stokes_solution;
		TrilinosWrappers::MPI::BlockVector        old_stokes_solution;
		TrilinosWrappers::MPI::BlockVector        stokes_rhs;


		FE_Q<dim>                                 temperature_fe;
		DoFHandler<dim>                           temperature_dof_handler;
		ConstraintMatrix                          temperature_constraints;

		TrilinosWrappers::SparseMatrix            temperature_mass_matrix;
		TrilinosWrappers::SparseMatrix            temperature_stiffness_matrix;
		TrilinosWrappers::SparseMatrix            temperature_matrix;


		double                                    time_step;
		double                                    old_time_step;
		unsigned int                              timestep_number;

		std_cxx11::shared_ptr<TrilinosWrappers::PreconditionAMG>    Amg_preconditioner;
		std_cxx11::shared_ptr<TrilinosWrappers::PreconditionJacobi> Mp_preconditioner;
		std_cxx11::shared_ptr<TrilinosWrappers::PreconditionJacobi> T_preconditioner;

		bool                                      rebuild_stokes_matrix;
		bool                                      rebuild_stokes_preconditioner;
		bool                                      rebuild_temperature_matrices;
		bool                                      rebuild_temperature_preconditioner;

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
};

#endif