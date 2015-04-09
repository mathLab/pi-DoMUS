#ifndef BOUSSINESQ_FLOW_PROBLEM 
#define BOUSSINESQ_FLOW_PROBLEM

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

     // MPI_Comm                                  mpi_communicator;

      SmartPointer<parallel::distributed::Triangulation<dim> > triangulation;
		// parallel::distributed::Triangulation<dim> triangulation;

		double                                    global_Omega_diameter;

		const MappingQ<dim>                       mapping;

      SmartPointer<FiniteElement<dim,dim> >     stokes_fe;

		//const FESystem<dim>                       stokes_fe;
      SmartPointer<DoFHandler<dim> >            stokes_dof_handler;
		//DoFHandler<dim>                           stokes_dof_handler;
		ConstraintMatrix                          stokes_constraints;

		TrilinosWrappers::BlockSparseMatrix       stokes_matrix;
		TrilinosWrappers::BlockSparseMatrix       stokes_preconditioner_matrix;

		/*TrilinosWrappers::MPI::Vector        stokes_solution;
		TrilinosWrappers::MPI::Vector        old_stokes_solution;
		TrilinosWrappers::MPI::Vector        stokes_rhs;*/

      TrilinosWrappers::MPI::BlockVector        stokes_solution;
		TrilinosWrappers::MPI::BlockVector        old_stokes_solution;
		TrilinosWrappers::MPI::BlockVector        stokes_rhs;

      //TrilinosWrappers::MPI::Vector             locally_relevant_solution;

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
};

#endif
