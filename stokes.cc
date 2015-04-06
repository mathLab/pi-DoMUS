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
#include "boussinesq_flow_problem.h"

using namespace dealii;

  template <int dim>
  BoussinesqFlowProblem<dim>::Parameters::Parameters (const std::string &parameter_filename)
    :
    end_time (1e8),
    initial_global_refinement (2),
    initial_adaptive_refinement (2),
    adaptive_refinement_interval (10),
    stabilization_alpha (2),
    stabilization_c_R (0.11),
    stabilization_beta (0.078),
    stokes_velocity_degree (2),
    use_locally_conservative_discretization (true)//,
    // temperature_degree (2)
  {
    ParameterHandler prm;
    BoussinesqFlowProblem<dim>::Parameters::declare_parameters (prm);

    std::ifstream parameter_file (parameter_filename.c_str());

    if (!parameter_file)
      {
        parameter_file.close ();

        std::ostringstream message;
        message << "Input parameter file <"
                << parameter_filename << "> not found. Creating a"
                << std::endl
                << "template file of the same name."
                << std::endl;

        std::ofstream parameter_out (parameter_filename.c_str());
        prm.print_parameters (parameter_out,
                              ParameterHandler::Text);

        AssertThrow (false, ExcMessage (message.str().c_str()));
      }

    const bool success = prm.read_input (parameter_file);
    AssertThrow (success, ExcMessage ("Invalid input parameter file."));

    parse_parameters (prm);
  }


  template <int dim>
  void
  BoussinesqFlowProblem<dim>::Parameters::
  declare_parameters (ParameterHandler &prm)
  {
    prm.declare_entry ("End time", "1e8",
                       Patterns::Double (0),
                       "The end time of the simulation in years.");
    prm.declare_entry ("Initial global refinement", "2",
                       Patterns::Integer (0),
                       "The number of global refinement steps performed on "
                       "the initial coarse mesh, before the problem is first "
                       "solved there.");
    prm.declare_entry ("Initial adaptive refinement", "2",
                       Patterns::Integer (0),
                       "The number of adaptive refinement steps performed after "
                       "initial global refinement.");
    prm.declare_entry ("Time steps between mesh refinement", "10",
                       Patterns::Integer (1),
                       "The number of time steps after which the mesh is to be "
                       "adapted based on computed error indicators.");
    prm.declare_entry ("Generate graphical output", "false",
                       Patterns::Bool (),
                       "Whether graphical output is to be generated or not. "
                       "You may not want to get graphical output if the number "
                       "of processors is large.");
    prm.declare_entry ("Time steps between graphical output", "50",
                       Patterns::Integer (1),
                       "The number of time steps between each generation of "
                       "graphical output files.");

    prm.enter_subsection ("Stabilization parameters");
    {
      prm.declare_entry ("alpha", "2",
                         Patterns::Double (1, 2),
                         "The exponent in the entropy viscosity stabilization.");
      prm.declare_entry ("c_R", "0.11",
                         Patterns::Double (0),
                         "The c_R factor in the entropy viscosity "
                         "stabilization.");
      prm.declare_entry ("beta", "0.078",
                         Patterns::Double (0),
                         "The beta factor in the artificial viscosity "
                         "stabilization. An appropriate value for 2d is 0.052 "
                         "and 0.078 for 3d.");
    }
    prm.leave_subsection ();

    prm.enter_subsection ("Discretization");
    {
      prm.declare_entry ("Stokes velocity polynomial degree", "2",
                         Patterns::Integer (1),
                         "The polynomial degree to use for the velocity variables "
                         "in the Stokes system.");
      prm.declare_entry ("Temperature polynomial degree", "2",
                         Patterns::Integer (1),
                         "The polynomial degree to use for the temperature variable.");
      prm.declare_entry ("Use locally conservative discretization", "true",
                         Patterns::Bool (),
                         "Whether to use a Stokes discretization that is locally "
                         "conservative at the expense of a larger number of degrees "
                         "of freedom, or to go with a cheaper discretization "
                         "that does not locally conserve mass (although it is "
                         "globally conservative.");
    }
    prm.leave_subsection ();
  }


  template <int dim>
  void
  BoussinesqFlowProblem<dim>::Parameters::
  parse_parameters (ParameterHandler &prm)
  {
    end_time                    = prm.get_double ("End time");
    initial_global_refinement   = prm.get_integer ("Initial global refinement");
    initial_adaptive_refinement = prm.get_integer ("Initial adaptive refinement");

    adaptive_refinement_interval= prm.get_integer ("Time steps between mesh refinement");

    generate_graphical_output   = prm.get_bool ("Generate graphical output");
    graphical_output_interval   = prm.get_integer ("Time steps between graphical output");

    prm.enter_subsection ("Stabilization parameters");
    {
      stabilization_alpha = prm.get_double ("alpha");
      stabilization_c_R   = prm.get_double ("c_R");
      stabilization_beta  = prm.get_double ("beta");
    }
    prm.leave_subsection ();

    prm.enter_subsection ("Discretization");
    {
      stokes_velocity_degree = prm.get_integer ("Stokes velocity polynomial degree");
      // temperature_degree     = prm.get_integer ("Temperature polynomial degree");
      use_locally_conservative_discretization
        = prm.get_bool ("Use locally conservative discretization");
    }
    prm.leave_subsection ();
  }


  template <int dim>
  BoussinesqFlowProblem<dim>::BoussinesqFlowProblem (Parameters &parameters_)
    :
    parameters (parameters_),
    pcout (std::cout,
           (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
            == 0)),

    triangulation (MPI_COMM_WORLD,
                   typename Triangulation<dim>::MeshSmoothing
                   (Triangulation<dim>::smoothing_on_refinement |
                    Triangulation<dim>::smoothing_on_coarsening)),

    mapping (4),

    stokes_fe (FE_Q<dim>(parameters.stokes_velocity_degree),
               dim,
               (parameters.use_locally_conservative_discretization
                ?
                static_cast<const FiniteElement<dim> &>
                (FE_DGP<dim>(parameters.stokes_velocity_degree-1))
                :
                static_cast<const FiniteElement<dim> &>
                (FE_Q<dim>(parameters.stokes_velocity_degree-1))),
               1),

    stokes_dof_handler (triangulation),

    // temperature_fe (parameters.temperature_degree),
    temperature_fe (2),
    temperature_dof_handler (triangulation),

    time_step (0),
    old_time_step (0),
    timestep_number (0),
    rebuild_stokes_matrix (true),
    rebuild_stokes_preconditioner (true),
    rebuild_temperature_matrices (true),
    rebuild_temperature_preconditioner (true),

    computing_timer (MPI_COMM_WORLD,
                     pcout,
                     TimerOutput::summary,
                     TimerOutput::wall_times)
  {}


  template <int dim>
  void BoussinesqFlowProblem<dim>::
  setup_stokes_matrix (const std::vector<IndexSet> &stokes_partitioning,
                       const std::vector<IndexSet> &stokes_relevant_partitioning)
  {
    stokes_matrix.clear ();

    TrilinosWrappers::BlockSparsityPattern sp(stokes_partitioning, stokes_partitioning,
                                              stokes_relevant_partitioning,
                                              MPI_COMM_WORLD);

    Table<2,DoFTools::Coupling> coupling (dim+1, dim+1);
    for (unsigned int c=0; c<dim+1; ++c)
      for (unsigned int d=0; d<dim+1; ++d)
        if (! ((c==dim) && (d==dim)))
          coupling[c][d] = DoFTools::always;
        else
          coupling[c][d] = DoFTools::none;

    DoFTools::make_sparsity_pattern (stokes_dof_handler,
                                     coupling, sp,
                                     stokes_constraints, false,
                                     Utilities::MPI::
                                     this_mpi_process(MPI_COMM_WORLD));
    sp.compress();

    stokes_matrix.reinit (sp);
  }


  template <int dim>
  void BoussinesqFlowProblem<dim>::
  setup_stokes_preconditioner (const std::vector<IndexSet> &stokes_partitioning,
                               const std::vector<IndexSet> &stokes_relevant_partitioning)
  {
    Amg_preconditioner.reset ();
    Mp_preconditioner.reset ();

    stokes_preconditioner_matrix.clear ();

    TrilinosWrappers::BlockSparsityPattern sp(stokes_partitioning, stokes_partitioning,
                                              stokes_relevant_partitioning,
                                              MPI_COMM_WORLD);

    Table<2,DoFTools::Coupling> coupling (dim+1, dim+1);
    for (unsigned int c=0; c<dim+1; ++c)
      for (unsigned int d=0; d<dim+1; ++d)
        if (c == d)
          coupling[c][d] = DoFTools::always;
        else
          coupling[c][d] = DoFTools::none;

    DoFTools::make_sparsity_pattern (stokes_dof_handler,
                                     coupling, sp,
                                     stokes_constraints, false,
                                     Utilities::MPI::
                                     this_mpi_process(MPI_COMM_WORLD));
    sp.compress();

    stokes_preconditioner_matrix.reinit (sp);
  }


  template <int dim>
  void BoussinesqFlowProblem<dim>::setup_dofs ()
  {
    computing_timer.enter_section("Setup dof systems");

    std::vector<unsigned int> stokes_sub_blocks (dim+1,0);
    stokes_sub_blocks[dim] = 1;
    stokes_dof_handler.distribute_dofs (stokes_fe);
    DoFRenumbering::component_wise (stokes_dof_handler, stokes_sub_blocks);

    temperature_dof_handler.distribute_dofs (temperature_fe);

    std::vector<types::global_dof_index> stokes_dofs_per_block (2);
    DoFTools::count_dofs_per_block (stokes_dof_handler, stokes_dofs_per_block,
                                    stokes_sub_blocks);

    const unsigned int n_u = stokes_dofs_per_block[0],
                       n_p = stokes_dofs_per_block[1],
                       n_T = temperature_dof_handler.n_dofs();

    std::locale s = pcout.get_stream().getloc();
    pcout.get_stream().imbue(std::locale(""));
    pcout << "Number of active cells: "
          << triangulation.n_global_active_cells()
          << " (on "
          << triangulation.n_levels()
          << " levels)"
          << std::endl
          << "Number of degrees of freedom: "
          << n_u + n_p + n_T
          << " (" << n_u << '+' << n_p << '+'<< n_T <<')'
          << std::endl
          << std::endl;
    pcout.get_stream().imbue(s);


    std::vector<IndexSet> stokes_partitioning, stokes_relevant_partitioning;
    IndexSet temperature_partitioning (n_T), temperature_relevant_partitioning (n_T);
    IndexSet stokes_relevant_set;
    {
      IndexSet stokes_index_set = stokes_dof_handler.locally_owned_dofs();
      stokes_partitioning.push_back(stokes_index_set.get_view(0,n_u));
      stokes_partitioning.push_back(stokes_index_set.get_view(n_u,n_u+n_p));

      DoFTools::extract_locally_relevant_dofs (stokes_dof_handler,
                                               stokes_relevant_set);
      stokes_relevant_partitioning.push_back(stokes_relevant_set.get_view(0,n_u));
      stokes_relevant_partitioning.push_back(stokes_relevant_set.get_view(n_u,n_u+n_p));

      temperature_partitioning = temperature_dof_handler.locally_owned_dofs();
      DoFTools::extract_locally_relevant_dofs (temperature_dof_handler,
                                               temperature_relevant_partitioning);
    }

    {
      stokes_constraints.clear ();
      stokes_constraints.reinit (stokes_relevant_set);

      DoFTools::make_hanging_node_constraints (stokes_dof_handler,
                                               stokes_constraints);

      FEValuesExtractors::Vector velocity_components(0);
      VectorTools::interpolate_boundary_values (stokes_dof_handler,
                                                0,
                                                ZeroFunction<dim>(dim+1),
                                                stokes_constraints,
                                                stokes_fe.component_mask(velocity_components));

      std::set<types::boundary_id> no_normal_flux_boundaries;
      no_normal_flux_boundaries.insert (1);
      VectorTools::compute_no_normal_flux_constraints (stokes_dof_handler, 0,
                                                       no_normal_flux_boundaries,
                                                       stokes_constraints,
                                                       mapping);
      stokes_constraints.close ();
    }
    // {
    //   temperature_constraints.clear ();
    //   temperature_constraints.reinit (temperature_relevant_partitioning);
    // 
    //   DoFTools::make_hanging_node_constraints (temperature_dof_handler,
    //                                            temperature_constraints);
    //   temperature_constraints.close ();
    // }

    setup_stokes_matrix (stokes_partitioning, stokes_relevant_partitioning);
    setup_stokes_preconditioner (stokes_partitioning,
                                 stokes_relevant_partitioning);

    stokes_rhs.reinit (stokes_partitioning, stokes_relevant_partitioning,
                       MPI_COMM_WORLD, true);
    stokes_solution.reinit (stokes_relevant_partitioning, MPI_COMM_WORLD);
    old_stokes_solution.reinit (stokes_solution);

    rebuild_stokes_matrix              = true;
    rebuild_stokes_preconditioner      = true;
    rebuild_temperature_matrices       = true;
    rebuild_temperature_preconditioner = true;

    computing_timer.exit_section();
  }

  template <int dim>
  void
  BoussinesqFlowProblem<dim>::
  local_assemble_stokes_preconditioner (const typename DoFHandler<dim>::active_cell_iterator &cell,
                                        Assembly::Scratch::StokesPreconditioner<dim> &scratch,
                                        Assembly::CopyData::StokesPreconditioner<dim> &data)
  {
    const unsigned int   dofs_per_cell   = stokes_fe.dofs_per_cell;
    const unsigned int   n_q_points      = scratch.stokes_fe_values.n_quadrature_points;

    const FEValuesExtractors::Vector velocities (0);
    const FEValuesExtractors::Scalar pressure (dim);

    scratch.stokes_fe_values.reinit (cell);
    cell->get_dof_indices (data.local_dof_indices);

    data.local_matrix = 0;

    for (unsigned int q=0; q<n_q_points; ++q)
      {
        for (unsigned int k=0; k<dofs_per_cell; ++k)
          {
            scratch.grad_phi_u[k] = scratch.stokes_fe_values[velocities].gradient(k,q);
            scratch.phi_p[k]      = scratch.stokes_fe_values[pressure].value (k, q);
          }

        for (unsigned int i=0; i<dofs_per_cell; ++i)
          for (unsigned int j=0; j<dofs_per_cell; ++j)
            data.local_matrix(i,j) += (EquationData::eta *
                                       scalar_product (scratch.grad_phi_u[i],
                                                       scratch.grad_phi_u[j])
                                       +
                                       (1./EquationData::eta) *
                                       EquationData::pressure_scaling *
                                       EquationData::pressure_scaling *
                                       (scratch.phi_p[i] * scratch.phi_p[j]))
                                      * scratch.stokes_fe_values.JxW(q);
      }
  }



  template <int dim>
  void
  BoussinesqFlowProblem<dim>::
  copy_local_to_global_stokes_preconditioner (const Assembly::CopyData::StokesPreconditioner<dim> &data)
  {
    stokes_constraints.distribute_local_to_global (data.local_matrix,
                                                   data.local_dof_indices,
                                                   stokes_preconditioner_matrix);
  }


  template <int dim>
  void
  BoussinesqFlowProblem<dim>::assemble_stokes_preconditioner ()
  {
    stokes_preconditioner_matrix = 0;

    const QGauss<dim> quadrature_formula(parameters.stokes_velocity_degree+1);

    typedef
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
    CellFilter;

    WorkStream::
    run (CellFilter (IteratorFilters::LocallyOwnedCell(),
                     stokes_dof_handler.begin_active()),
         CellFilter (IteratorFilters::LocallyOwnedCell(),
                     stokes_dof_handler.end()),
         std_cxx11::bind (&BoussinesqFlowProblem<dim>::
                          local_assemble_stokes_preconditioner,
                          this,
                          std_cxx11::_1,
                          std_cxx11::_2,
                          std_cxx11::_3),
         std_cxx11::bind (&BoussinesqFlowProblem<dim>::
                          copy_local_to_global_stokes_preconditioner,
                          this,
                          std_cxx11::_1),
         Assembly::Scratch::
         StokesPreconditioner<dim> (stokes_fe, quadrature_formula,
                                    mapping,
                                    update_JxW_values |
                                    update_values |
                                    update_gradients),
         Assembly::CopyData::
         StokesPreconditioner<dim> (stokes_fe));

    stokes_preconditioner_matrix.compress(VectorOperation::add);
  }



  template <int dim>
  void
  BoussinesqFlowProblem<dim>::build_stokes_preconditioner ()
  {
    if (rebuild_stokes_preconditioner == false)
      return;

    computing_timer.enter_section ("   Build Stokes preconditioner");
    pcout << "   Rebuilding Stokes preconditioner..." << std::flush;

    assemble_stokes_preconditioner ();

    std::vector<std::vector<bool> > constant_modes;
    FEValuesExtractors::Vector velocity_components(0);
    DoFTools::extract_constant_modes (stokes_dof_handler,
                                      stokes_fe.component_mask(velocity_components),
                                      constant_modes);

    Mp_preconditioner.reset  (new TrilinosWrappers::PreconditionJacobi());
    Amg_preconditioner.reset (new TrilinosWrappers::PreconditionAMG());

    TrilinosWrappers::PreconditionAMG::AdditionalData Amg_data;
    Amg_data.constant_modes = constant_modes;
    Amg_data.elliptic = true;
    Amg_data.higher_order_elements = true;
    Amg_data.smoother_sweeps = 2;
    Amg_data.aggregation_threshold = 0.02;

    Mp_preconditioner->initialize (stokes_preconditioner_matrix.block(1,1));
    Amg_preconditioner->initialize (stokes_preconditioner_matrix.block(0,0),
                                    Amg_data);

    rebuild_stokes_preconditioner = false;

    pcout << std::endl;
    computing_timer.exit_section();
  }



  template <int dim>
  void
  BoussinesqFlowProblem<dim>::
  local_assemble_stokes_system (const typename DoFHandler<dim>::active_cell_iterator &cell,
                                Assembly::Scratch::StokesSystem<dim> &scratch,
                                Assembly::CopyData::StokesSystem<dim> &data)
  {
    const unsigned int dofs_per_cell = scratch.stokes_fe_values.get_fe().dofs_per_cell;
    const unsigned int n_q_points    = scratch.stokes_fe_values.n_quadrature_points;

    const FEValuesExtractors::Vector velocities (0);
    const FEValuesExtractors::Scalar pressure (dim);

    scratch.stokes_fe_values.reinit (cell);

    typename DoFHandler<dim>::active_cell_iterator
    temperature_cell (&triangulation,
                      cell->level(),
                      cell->index(),
                      &temperature_dof_handler);
    scratch.temperature_fe_values.reinit (temperature_cell);

    if (rebuild_stokes_matrix)
      data.local_matrix = 0;
    data.local_rhs = 0;


    for (unsigned int q=0; q<n_q_points; ++q)
      {
        const double old_temperature = scratch.old_temperature_values[q];

        for (unsigned int k=0; k<dofs_per_cell; ++k)
          {
            scratch.phi_u[k] = scratch.stokes_fe_values[velocities].value (k,q);
            if (rebuild_stokes_matrix)
              {
                scratch.grads_phi_u[k] = scratch.stokes_fe_values[velocities].symmetric_gradient(k,q);
                scratch.div_phi_u[k]   = scratch.stokes_fe_values[velocities].divergence (k, q);
                scratch.phi_p[k]       = scratch.stokes_fe_values[pressure].value (k, q);
              }
          }

        if (rebuild_stokes_matrix == true)
          for (unsigned int i=0; i<dofs_per_cell; ++i)
            for (unsigned int j=0; j<dofs_per_cell; ++j)
              data.local_matrix(i,j) += (EquationData::eta * 2 *
                                         (scratch.grads_phi_u[i] * scratch.grads_phi_u[j])
                                         - (EquationData::pressure_scaling *
                                            scratch.div_phi_u[i] * scratch.phi_p[j])
                                         - (EquationData::pressure_scaling *
                                            scratch.phi_p[i] * scratch.div_phi_u[j]))
                                        * scratch.stokes_fe_values.JxW(q);

        const Tensor<1,dim>
        gravity = EquationData::gravity_vector (scratch.stokes_fe_values
                                                .quadrature_point(q));

        for (unsigned int i=0; i<dofs_per_cell; ++i)
          data.local_rhs(i) += (EquationData::density *
                                gravity  *
                                scratch.phi_u[i]) *
                               scratch.stokes_fe_values.JxW(q);
      }

    cell->get_dof_indices (data.local_dof_indices);
  }



  template <int dim>
  void
  BoussinesqFlowProblem<dim>::
  copy_local_to_global_stokes_system (const Assembly::CopyData::StokesSystem<dim> &data)
  {
    if (rebuild_stokes_matrix == true)
      stokes_constraints.distribute_local_to_global (data.local_matrix,
                                                     data.local_rhs,
                                                     data.local_dof_indices,
                                                     stokes_matrix,
                                                     stokes_rhs);
    else
      stokes_constraints.distribute_local_to_global (data.local_rhs,
                                                     data.local_dof_indices,
                                                     stokes_rhs);
  }



  template <int dim>
  void BoussinesqFlowProblem<dim>::assemble_stokes_system ()
  {
    computing_timer.enter_section ("   Assemble Stokes system");

    if (rebuild_stokes_matrix == true)
      stokes_matrix=0;

    stokes_rhs=0;

    const QGauss<dim> quadrature_formula(parameters.stokes_velocity_degree+1);

    typedef
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
    CellFilter;

    WorkStream::
    run (CellFilter (IteratorFilters::LocallyOwnedCell(),
                     stokes_dof_handler.begin_active()),
         CellFilter (IteratorFilters::LocallyOwnedCell(),
                     stokes_dof_handler.end()),
         std_cxx11::bind (&BoussinesqFlowProblem<dim>::
                          local_assemble_stokes_system,
                          this,
                          std_cxx11::_1,
                          std_cxx11::_2,
                          std_cxx11::_3),
         std_cxx11::bind (&BoussinesqFlowProblem<dim>::
                          copy_local_to_global_stokes_system,
                          this,
                          std_cxx11::_1),
         Assembly::Scratch::
         StokesSystem<dim> (stokes_fe, mapping, quadrature_formula,
                            (update_values    |
                             update_quadrature_points  |
                             update_JxW_values |
                             (rebuild_stokes_matrix == true
                              ?
                              update_gradients
                              :
                              UpdateFlags(0))),
                            temperature_fe,
                            update_values),
         Assembly::CopyData::
         StokesSystem<dim> (stokes_fe));

    if (rebuild_stokes_matrix == true)
      stokes_matrix.compress(VectorOperation::add);
    stokes_rhs.compress(VectorOperation::add);

    rebuild_stokes_matrix = false;

    pcout << std::endl;
    computing_timer.exit_section();
  }

  template <int dim>
  void BoussinesqFlowProblem<dim>::solve ()
  {
    computing_timer.enter_section ("   Solve Stokes system");

    {
      pcout << "   Solving Stokes system... " << std::flush;

      TrilinosWrappers::MPI::BlockVector
      distributed_stokes_solution (stokes_rhs);
      distributed_stokes_solution = stokes_solution;

      distributed_stokes_solution.block(1) /= EquationData::pressure_scaling;

      const unsigned int
      start = (distributed_stokes_solution.block(0).size() +
               distributed_stokes_solution.block(1).local_range().first),
              end   = (distributed_stokes_solution.block(0).size() +
                       distributed_stokes_solution.block(1).local_range().second);
      for (unsigned int i=start; i<end; ++i)
        if (stokes_constraints.is_constrained (i))
          distributed_stokes_solution(i) = 0;


      PrimitiveVectorMemory<TrilinosWrappers::MPI::BlockVector> mem;

      unsigned int n_iterations = 0;
      const double solver_tolerance = 1e-8 * stokes_rhs.l2_norm();
      SolverControl solver_control (30, solver_tolerance);

      try
        {
          const LinearSolvers::BlockSchurPreconditioner<TrilinosWrappers::PreconditionAMG,
                TrilinosWrappers::PreconditionJacobi>
                preconditioner (stokes_matrix, stokes_preconditioner_matrix,
                                *Mp_preconditioner, *Amg_preconditioner,
                                false);

          SolverFGMRES<TrilinosWrappers::MPI::BlockVector>
          solver(solver_control, mem,
                 SolverFGMRES<TrilinosWrappers::MPI::BlockVector>::
                 AdditionalData(30, true));
          solver.solve(stokes_matrix, distributed_stokes_solution, stokes_rhs,
                       preconditioner);

          n_iterations = solver_control.last_step();
        }

      catch (SolverControl::NoConvergence)
        {
          const LinearSolvers::BlockSchurPreconditioner<TrilinosWrappers::PreconditionAMG,
                TrilinosWrappers::PreconditionJacobi>
                preconditioner (stokes_matrix, stokes_preconditioner_matrix,
                                *Mp_preconditioner, *Amg_preconditioner,
                                true);

          SolverControl solver_control_refined (stokes_matrix.m(), solver_tolerance);
          SolverFGMRES<TrilinosWrappers::MPI::BlockVector>
          solver(solver_control_refined, mem,
                 SolverFGMRES<TrilinosWrappers::MPI::BlockVector>::
                 AdditionalData(50, true));
          solver.solve(stokes_matrix, distributed_stokes_solution, stokes_rhs,
                       preconditioner);

          n_iterations = (solver_control.last_step() +
                          solver_control_refined.last_step());
        }


      stokes_constraints.distribute (distributed_stokes_solution);

      distributed_stokes_solution.block(1) *= EquationData::pressure_scaling;

      stokes_solution = distributed_stokes_solution;
      pcout << n_iterations  << " iterations."
            << std::endl;
    }
    computing_timer.exit_section();


    computing_timer.enter_section ("   Assemble temperature rhs");
    {
      // This part has to be removed:
      //  If removed an infinite loop starts
      old_time_step = time_step;
    
      const double scaling = (dim==3 ? 0.25 : 1.0);
      const double cfl_number = 1e-20;
      time_step = (scaling/(2.1*dim*std::sqrt(1.*dim)) /
                     (2 *
                      cfl_number));
                      
    }
    computing_timer.exit_section ();

  }



  template <int dim>
  class BoussinesqFlowProblem<dim>::Postprocessor : public DataPostprocessor<dim>
  {
  public:
    Postprocessor (const unsigned int partition,
                   const double       minimal_pressure);

    virtual
    void
    compute_derived_quantities_vector (const std::vector<Vector<double> >              &uh,
                                       const std::vector<std::vector<Tensor<1,dim> > > &duh,
                                       const std::vector<std::vector<Tensor<2,dim> > > &dduh,
                                       const std::vector<Point<dim> >                  &normals,
                                       const std::vector<Point<dim> >                  &evaluation_points,
                                       std::vector<Vector<double> >                    &computed_quantities) const;

    virtual std::vector<std::string> get_names () const;

    virtual
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    get_data_component_interpretation () const;

    virtual UpdateFlags get_needed_update_flags () const;

  private:
    const unsigned int partition;
    const double       minimal_pressure;
  };


  template <int dim>
  BoussinesqFlowProblem<dim>::Postprocessor::
  Postprocessor (const unsigned int partition,
                 const double       minimal_pressure)
    :
    partition (partition),
    minimal_pressure (minimal_pressure)
  {}


  template <int dim>
  std::vector<std::string>
  BoussinesqFlowProblem<dim>::Postprocessor::get_names() const
  {
    std::vector<std::string> solution_names (dim, "velocity");
    solution_names.push_back ("p");
    solution_names.push_back ("T");
    solution_names.push_back ("friction_heating");
    solution_names.push_back ("partition");

    return solution_names;
  }


  template <int dim>
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  BoussinesqFlowProblem<dim>::Postprocessor::
  get_data_component_interpretation () const
  {
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    interpretation (dim,
                    DataComponentInterpretation::component_is_part_of_vector);

    interpretation.push_back (DataComponentInterpretation::component_is_scalar);
    interpretation.push_back (DataComponentInterpretation::component_is_scalar);
    interpretation.push_back (DataComponentInterpretation::component_is_scalar);
    interpretation.push_back (DataComponentInterpretation::component_is_scalar);

    return interpretation;
  }


  template <int dim>
  UpdateFlags
  BoussinesqFlowProblem<dim>::Postprocessor::get_needed_update_flags() const
  {
    return update_values | update_gradients | update_q_points;
  }


  template <int dim>
  void
  BoussinesqFlowProblem<dim>::Postprocessor::
  compute_derived_quantities_vector (const std::vector<Vector<double> >              &uh,
                                     const std::vector<std::vector<Tensor<1,dim> > > &duh,
                                     const std::vector<std::vector<Tensor<2,dim> > > &/*dduh*/,
                                     const std::vector<Point<dim> >                  &/*normals*/,
                                     const std::vector<Point<dim> >                  &/*evaluation_points*/,
                                     std::vector<Vector<double> >                    &computed_quantities) const
  {
    const unsigned int n_quadrature_points = uh.size();
    Assert (duh.size() == n_quadrature_points,                  ExcInternalError());
    Assert (computed_quantities.size() == n_quadrature_points,  ExcInternalError());
    Assert (uh[0].size() == dim+2,                              ExcInternalError());

    for (unsigned int q=0; q<n_quadrature_points; ++q)
      {
        for (unsigned int d=0; d<dim; ++d)
          computed_quantities[q](d)
            = (uh[q](d) *  EquationData::year_in_seconds * 100);

        const double pressure = (uh[q](dim)-minimal_pressure);
        computed_quantities[q](dim) = pressure;

        const double temperature = uh[q](dim+1);
        computed_quantities[q](dim+1) = temperature;

        Tensor<2,dim> grad_u;
        for (unsigned int d=0; d<dim; ++d)
          grad_u[d] = duh[q][d];
        const SymmetricTensor<2,dim> strain_rate = symmetrize (grad_u);
        computed_quantities[q](dim+2) = 2 * EquationData::eta *
                                        strain_rate * strain_rate;

        computed_quantities[q](dim+3) = partition;
      }
  }


  template <int dim>
  void BoussinesqFlowProblem<dim>::output_results ()
  {
    computing_timer.enter_section ("Postprocessing");

    const FESystem<dim> joint_fe (stokes_fe, 1,
                                  temperature_fe, 1);

    DoFHandler<dim> joint_dof_handler (triangulation);
    joint_dof_handler.distribute_dofs (joint_fe);
    Assert (joint_dof_handler.n_dofs() ==
            stokes_dof_handler.n_dofs() + temperature_dof_handler.n_dofs(),
            ExcInternalError());

    TrilinosWrappers::MPI::Vector joint_solution;
    joint_solution.reinit (joint_dof_handler.locally_owned_dofs(), MPI_COMM_WORLD);

    joint_solution.compress(VectorOperation::insert);

    IndexSet locally_relevant_joint_dofs(joint_dof_handler.n_dofs());
    DoFTools::extract_locally_relevant_dofs (joint_dof_handler, locally_relevant_joint_dofs);
    TrilinosWrappers::MPI::Vector locally_relevant_joint_solution;
    locally_relevant_joint_solution.reinit (locally_relevant_joint_dofs, MPI_COMM_WORLD);
    locally_relevant_joint_solution = joint_solution;

    Postprocessor postprocessor (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD),
                                 stokes_solution.block(1).minimal_value());

    DataOut<dim> data_out;
    data_out.attach_dof_handler (joint_dof_handler);
    data_out.add_data_vector (locally_relevant_joint_solution, postprocessor);
    data_out.build_patches ();

    static int out_index=0;
    const std::string filename = ("solution-" +
                                  Utilities::int_to_string (out_index, 5) +
                                  "." +
                                  Utilities::int_to_string
                                  (triangulation.locally_owned_subdomain(), 4) +
                                  ".vtu");
    std::ofstream output (filename.c_str());
    data_out.write_vtu (output);


    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        std::vector<std::string> filenames;
        for (unsigned int i=0; i<Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD); ++i)
          filenames.push_back (std::string("solution-") +
                               Utilities::int_to_string (out_index, 5) +
                               "." +
                               Utilities::int_to_string(i, 4) +
                               ".vtu");
        const std::string
        pvtu_master_filename = ("solution-" +
                                Utilities::int_to_string (out_index, 5) +
                                ".pvtu");
        std::ofstream pvtu_master (pvtu_master_filename.c_str());
        data_out.write_pvtu_record (pvtu_master, filenames);

        const std::string
        visit_master_filename = ("solution-" +
                                 Utilities::int_to_string (out_index, 5) +
                                 ".visit");
        std::ofstream visit_master (visit_master_filename.c_str());
        data_out.write_visit_record (visit_master, filenames);
      }

    computing_timer.exit_section ();
    out_index++;
  }


  template <int dim>
  void BoussinesqFlowProblem<dim>::refine_mesh (const unsigned int max_grid_level)
  {
    // Rewrite this function using a refinement on stoke's equation!
  }

  template <int dim>
  void BoussinesqFlowProblem<dim>::make_grid()
  {
    GridGenerator::hyper_shell (triangulation,
                                Point<dim>(),
                                EquationData::R0,
                                EquationData::R1,
                                (dim==3) ? 96 : 12,
                                true);
    static HyperShellBoundary<dim> boundary;
    triangulation.set_boundary (0, boundary);
    triangulation.set_boundary (1, boundary);

    global_Omega_diameter = GridTools::diameter (triangulation);

    triangulation.refine_global (parameters.initial_global_refinement);
  }

  template <int dim>
  void BoussinesqFlowProblem<dim>::run ()
  {
    make_grid();

    setup_dofs();

    unsigned int pre_refinement_step = 0;

start_time_iteration:

    timestep_number           = 0;
    time_step = old_time_step = 0;

    double time = 0;

    do
      {
        pcout << "Timestep " << timestep_number
              << ":  t=" << time/EquationData::year_in_seconds
              << " years"
              << std::endl;

        assemble_stokes_system ();
        build_stokes_preconditioner ();

        solve ();

        // pcout << std::endl;
        // 
        // if ((timestep_number == 0) &&
        //     (pre_refinement_step < parameters.initial_adaptive_refinement))
        //   {
        //     refine_mesh (parameters.initial_global_refinement +
        //                  parameters.initial_adaptive_refinement);
        //     ++pre_refinement_step;
        //     goto start_time_iteration;
        //   }
        // else if ((timestep_number > 0)
        //          &&
        //          (timestep_number % parameters.adaptive_refinement_interval == 0))
        //   refine_mesh (parameters.initial_global_refinement +
        //                parameters.initial_adaptive_refinement);

        if (//(parameters.generate_graphical_output == true)
            //&&
            (timestep_number % parameters.graphical_output_interval == 0))
          output_results ();

        // if (time > parameters.end_time * EquationData::year_in_seconds)
        //   break;
        if (time > 5 * EquationData::year_in_seconds)
          break;

        TrilinosWrappers::MPI::BlockVector old_old_stokes_solution;
        old_old_stokes_solution      = old_stokes_solution;
        old_stokes_solution          = stokes_solution;
        if (old_time_step > 0)
          {
            {
              TrilinosWrappers::MPI::BlockVector distr_solution (stokes_rhs);
              distr_solution = stokes_solution;
              TrilinosWrappers::MPI::BlockVector distr_old_solution (stokes_rhs);
              distr_old_solution = old_old_stokes_solution;
              distr_solution .sadd (1.+time_step/old_time_step, -time_step/old_time_step,
                                    distr_old_solution);
              stokes_solution = distr_solution;
            }
          }

        if ((timestep_number > 0) && (timestep_number % 100 == 0))
          computing_timer.print_summary ();

        time += time_step;
        ++timestep_number;
      }
    while (true);

    if ((parameters.generate_graphical_output == true)
        &&
        !((timestep_number-1) % parameters.graphical_output_interval == 0))
      output_results ();
  }


int main (int argc, char *argv[])
{
  using namespace dealii;

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                      numbers::invalid_unsigned_int);

  try
    {
      deallog.depth_console (0);

      std::string parameter_filename;
      if (argc>=2)
        parameter_filename = argv[1];
      else
        parameter_filename = "stokes.prm";

      const int dim = 2;
      BoussinesqFlowProblem<dim>::Parameters  parameters(parameter_filename);
      BoussinesqFlowProblem<dim> flow_problem (parameters);
      flow_problem.run ();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
