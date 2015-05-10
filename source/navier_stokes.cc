#include "navier_stokes.h"
#include <deal.II/lac/trilinos_precondition.h>

using namespace dealii;

/* ------------------------ PARAMETERS ------------------------ */

template <int dim>
void
NavierStokes<dim>::
declare_parameters (ParameterHandler &prm)
{
  add_parameter(  prm,
                  &end_time,
                  "End time",
                  "1e8",
                  Patterns::Double (0));

  add_parameter(  prm,
                  &initial_global_refinement,
                  "Initial global refinement",
                  "1",
                  Patterns::Integer (0));

  add_parameter(  prm,
                  &initial_adaptive_refinement,
                  "Initial adaptive refinement",
                  "2",
                  Patterns::Integer (0));

  add_parameter(  prm,
                  &adaptive_refinement_interval,
                  "Time steps between mesh refinement",
                  "10",
                  Patterns::Integer (1));

  add_parameter(  prm,
                  &generate_graphical_output,
                  "Generate graphical output",
                  "false",
                  Patterns::Bool ());

  add_parameter(  prm,
                  &graphical_output_interval,
                  "Time steps between graphical output",
                  "50",
                  Patterns::Integer (1));

  add_parameter(  prm,
                  &stokes_velocity_degree,
                  "Stokes velocity polynomial degree",
                  "2",
                  Patterns::Integer (1));

  add_parameter(  prm,
                  &use_locally_conservative_discretization,
                  "Use locally conservative discretization",
                  "true",
                  Patterns::Bool ());
}

/* ------------------------ CONSTRUCTORS ------------------------ */

template <int dim>
NavierStokes<dim>::NavierStokes (const RefinementMode refinement_mode)
  :
  pcout (std::cout,
         (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
          == 0)),

  mapping (4),

  time_step (0),
  old_time_step (0),
  timestep_number (0),
  rebuild_navier_stokes_matrix (true),
  rebuild_navier_stokes_preconditioner (true),

  computing_timer (MPI_COMM_WORLD,
                   pcout,
                   TimerOutput::summary,
                   TimerOutput::wall_times),

  refinement_mode (refinement_mode),

  eh("ErrorHandler<1>", "u, u, p","L2, Linfty, H1; AddUp; L2"),

  pgg("Cube"),

  fe_builder(           "FE_Q",
                        "FESystem[FE_Q(2)^dim-FE_Q(1)]"),

  boundary_conditions(  "Dirichlet boundary conditions",
                        "k*pi*cos(k*pi*x)*cos(k*pi*y); k*pi*sin(k*pi*x)*sin(k*pi*y); 0",
                        "k=1"),

  right_hand_side(      "Right-hand side force",
                        "2*k^3*pi^3*cos(k*pi*x)*cos(k*pi*y); 2*k^3*pi^3*sin(k*pi*x)*sin(k*pi*y); 0",
                        "k=1" )

{}


/*
  template <int dim>
  NavierStokes<dim>::~NavierStokes ()
  {
    // navier_stokes_dof_handler->clear ();
    smart_delete(navier_stokes_dof_handler);
    smart_delete(navier_stokes_fe);
    smart_delete(triangulation);
  }
  */

/* ------------------------ DEGREE OF FREEDOM ------------------------ */

template <int dim>
void NavierStokes<dim>::setup_dofs ()
{
  computing_timer.enter_section("Setup dof systems");

  std::vector<unsigned int> navier_stokes_sub_blocks (dim+1,0);
  navier_stokes_sub_blocks[dim] = 1;
  navier_stokes_dof_handler->distribute_dofs (*navier_stokes_fe);
  DoFRenumbering::component_wise (*navier_stokes_dof_handler, navier_stokes_sub_blocks);

  std::vector<types::global_dof_index> navier_stokes_dofs_per_block (2);
  DoFTools::count_dofs_per_block (*navier_stokes_dof_handler, navier_stokes_dofs_per_block,
                                  navier_stokes_sub_blocks);

  const unsigned int n_u = navier_stokes_dofs_per_block[0],
                     n_p = navier_stokes_dofs_per_block[1];

  std::locale s = pcout.get_stream().getloc();
  pcout.get_stream().imbue(std::locale(""));
  pcout << "Number of active cells: "
        << triangulation->n_global_active_cells()
        << " (on "
        << triangulation->n_levels()
        << " levels)"
        << std::endl
        << "Number of degrees of freedom: "
        << n_u + n_p
        << " (" << n_u << '+' << n_p
        << ')'
        << std::endl
        << std::endl;
  pcout.get_stream().imbue(s);


  std::vector<IndexSet> navier_stokes_partitioning, navier_stokes_relevant_partitioning;
  IndexSet navier_stokes_relevant_set;
  {
    IndexSet navier_stokes_index_set = navier_stokes_dof_handler->locally_owned_dofs();
    navier_stokes_partitioning.push_back(navier_stokes_index_set.get_view(0,n_u));
    navier_stokes_partitioning.push_back(navier_stokes_index_set.get_view(n_u,n_u+n_p));

    DoFTools::extract_locally_relevant_dofs (*navier_stokes_dof_handler,
                                             navier_stokes_relevant_set);
    navier_stokes_relevant_partitioning.push_back(navier_stokes_relevant_set.get_view(0,n_u));
    navier_stokes_relevant_partitioning.push_back(navier_stokes_relevant_set.get_view(n_u,n_u+n_p));

  }

  {
    navier_stokes_constraints.clear ();
    navier_stokes_constraints.reinit (navier_stokes_relevant_set);

    DoFTools::make_hanging_node_constraints (*navier_stokes_dof_handler,
                                             navier_stokes_constraints);

    FEValuesExtractors::Vector velocity_components(0);
    //boundary_conditions.set_time(time_step*time_step_number);
    VectorTools::interpolate_boundary_values (*navier_stokes_dof_handler,
                                              0,
                                              boundary_conditions,
                                              navier_stokes_constraints,
                                              navier_stokes_fe->component_mask(velocity_components));

    navier_stokes_constraints.close ();
  }

  setup_navier_stokes_matrix (navier_stokes_partitioning, navier_stokes_relevant_partitioning);
  setup_navier_stokes_preconditioner (navier_stokes_partitioning,
                                      navier_stokes_relevant_partitioning);

  navier_stokes_rhs.reinit (navier_stokes_partitioning, navier_stokes_relevant_partitioning,
                            MPI_COMM_WORLD, true);
  navier_stokes_solution.reinit (navier_stokes_relevant_partitioning, MPI_COMM_WORLD);
  old_navier_stokes_solution.reinit (navier_stokes_solution);

  rebuild_navier_stokes_matrix              = true;
  rebuild_navier_stokes_preconditioner      = true;

  computing_timer.exit_section();
}

/* ------------------------ SETUP MATRIX ------------------------ */

template <int dim>
void NavierStokes<dim>::
setup_navier_stokes_matrix (const std::vector<IndexSet> &navier_stokes_partitioning,
                            const std::vector<IndexSet> &navier_stokes_relevant_partitioning)
{
  navier_stokes_matrix.clear ();

  TrilinosWrappers::BlockSparsityPattern sp(navier_stokes_partitioning, navier_stokes_partitioning,
                                            navier_stokes_relevant_partitioning,
                                            MPI_COMM_WORLD);

  Table<2,DoFTools::Coupling> coupling (dim+1, dim+1);
  for (unsigned int c=0; c<dim+1; ++c)
    for (unsigned int d=0; d<dim+1; ++d)
      if (! ((c==dim) && (d==dim)))
        coupling[c][d] = DoFTools::always;
      else
        coupling[c][d] = DoFTools::none;

  DoFTools::make_sparsity_pattern (*navier_stokes_dof_handler,
                                   coupling, sp,
                                   navier_stokes_constraints, false,
                                   Utilities::MPI::
                                   this_mpi_process(MPI_COMM_WORLD));
  sp.compress();

  navier_stokes_matrix.reinit (sp);
}

/* ------------------------ PRECONDITIONER ------------------------ */

template <int dim>
void NavierStokes<dim>::
setup_navier_stokes_preconditioner (const std::vector<IndexSet> &navier_stokes_partitioning,
                                    const std::vector<IndexSet> &navier_stokes_relevant_partitioning)
{
  Amg_preconditioner.reset ();
  Mp_preconditioner.reset ();

  navier_stokes_preconditioner_matrix.clear ();

  TrilinosWrappers::BlockSparsityPattern sp(navier_stokes_partitioning, navier_stokes_partitioning,
                                            navier_stokes_relevant_partitioning,
                                            MPI_COMM_WORLD);

  Table<2,DoFTools::Coupling> coupling (dim+1, dim+1);
  for (unsigned int c=0; c<dim+1; ++c)
    for (unsigned int d=0; d<dim+1; ++d)
      if (c == d)
        coupling[c][d] = DoFTools::always;
      else
        coupling[c][d] = DoFTools::none;

  DoFTools::make_sparsity_pattern (*navier_stokes_dof_handler,
                                   coupling, sp,
                                   navier_stokes_constraints, false,
                                   Utilities::MPI::
                                   this_mpi_process(MPI_COMM_WORLD));
  sp.compress();

  navier_stokes_preconditioner_matrix.reinit (sp);
}

template <int dim>
void
NavierStokes<dim>::
local_assemble_navier_stokes_preconditioner (const typename DoFHandler<dim>::active_cell_iterator &cell,
                                             Assembly::Scratch::NavierStokesPreconditioner<dim> &scratch,
                                             Assembly::CopyData::NavierStokesPreconditioner<dim> &data)
{
  const unsigned int   dofs_per_cell   = navier_stokes_fe->dofs_per_cell;
  const unsigned int   n_q_points      = scratch.navier_stokes_fe_values.n_quadrature_points;

  const FEValuesExtractors::Vector velocities (0);
  const FEValuesExtractors::Scalar pressure (dim);

  scratch.navier_stokes_fe_values.reinit (cell);
  cell->get_dof_indices (data.local_dof_indices);

  data.local_matrix = 0;

  for (unsigned int q=0; q<n_q_points; ++q)
    {
      for (unsigned int k=0; k<dofs_per_cell; ++k)
        {
          scratch.grad_phi_u[k] = scratch.navier_stokes_fe_values[velocities].gradient(k,q);
          scratch.phi_p[k]      = scratch.navier_stokes_fe_values[pressure].value (k, q);
        }

      for (unsigned int i=0; i<dofs_per_cell; ++i)
        for (unsigned int j=0; j<dofs_per_cell; ++j)
          data.local_matrix(i,j) += (EquationData::eta *
                                     scalar_product (scratch.grad_phi_u[i],
                                                     scratch.grad_phi_u[j])
                                     +
                                     (1./EquationData::eta) *
                                     (scratch.phi_p[i] * scratch.phi_p[j]))
                                    * scratch.navier_stokes_fe_values.JxW(q);
    }
}

template <int dim>
void
NavierStokes<dim>::
copy_local_to_global_navier_stokes_preconditioner (const Assembly::CopyData::NavierStokesPreconditioner<dim> &data)
{
  navier_stokes_constraints.distribute_local_to_global (  data.local_matrix,
                                                          data.local_dof_indices,
                                                          navier_stokes_preconditioner_matrix);
}

template <int dim>
void
NavierStokes<dim>::assemble_navier_stokes_preconditioner ()
{
  navier_stokes_preconditioner_matrix = 0;

  const QGauss<dim> quadrature_formula(stokes_velocity_degree+1);

  typedef
  FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
  CellFilter;

  WorkStream::
  run (CellFilter (IteratorFilters::LocallyOwnedCell(),
                   navier_stokes_dof_handler->begin_active()),
       CellFilter (IteratorFilters::LocallyOwnedCell(),
                   navier_stokes_dof_handler->end()),
       std_cxx11::bind (&NavierStokes<dim>::
                        local_assemble_navier_stokes_preconditioner,
                        this,
                        std_cxx11::_1,
                        std_cxx11::_2,
                        std_cxx11::_3),
       std_cxx11::bind (&NavierStokes<dim>::
                        copy_local_to_global_navier_stokes_preconditioner,
                        this,
                        std_cxx11::_1),
       Assembly::Scratch::
       NavierStokesPreconditioner<dim> (*navier_stokes_fe, quadrature_formula,
                                        mapping,
                                        update_JxW_values |
                                        update_values |
                                        update_gradients),
       Assembly::CopyData::
       NavierStokesPreconditioner<dim> (*navier_stokes_fe));

  navier_stokes_preconditioner_matrix.compress(VectorOperation::add);
}


template <int dim>
void
NavierStokes<dim>::build_navier_stokes_preconditioner ()
{
  if (rebuild_navier_stokes_preconditioner == false)
    return;

  computing_timer.enter_section ("   Build Stokes preconditioner");
  pcout << "   Rebuilding Stokes preconditioner..." << std::flush;

  assemble_navier_stokes_preconditioner ();

  std::vector<std::vector<bool> > constant_modes;
  FEValuesExtractors::Vector velocity_components(0);
  DoFTools::extract_constant_modes (*navier_stokes_dof_handler,
                                    navier_stokes_fe->component_mask(velocity_components),
                                    constant_modes);

  Mp_preconditioner.reset  (new TrilinosWrappers::PreconditionJacobi());
  Amg_preconditioner.reset (new TrilinosWrappers::PreconditionAMG());

  TrilinosWrappers::PreconditionAMG::AdditionalData Amg_data;
  Amg_data.constant_modes = constant_modes;
  Amg_data.elliptic = true;
  Amg_data.higher_order_elements = true;
  Amg_data.smoother_sweeps = 2;
  Amg_data.aggregation_threshold = 0.02;

  Mp_preconditioner->initialize (navier_stokes_preconditioner_matrix.block(1,1));
  Amg_preconditioner->initialize (navier_stokes_preconditioner_matrix.block(0,0),
                                  Amg_data);

  rebuild_navier_stokes_preconditioner = false;

  pcout << std::endl;
  computing_timer.exit_section();
}

/* ------------------------ ASSEMBLE THE SYSTEM ------------------------ */

template <int dim>
void
NavierStokes<dim>::
local_assemble_navier_stokes_system (const typename DoFHandler<dim>::active_cell_iterator &cell,
                                     Assembly::Scratch::NavierStokesSystem<dim> &scratch,
                                     Assembly::CopyData::NavierStokesSystem<dim> &data)
{
  const unsigned int dofs_per_cell = scratch.navier_stokes_fe_values.get_fe().dofs_per_cell;
  const unsigned int n_q_points    = scratch.navier_stokes_fe_values.n_quadrature_points;

  const FEValuesExtractors::Vector velocities (0);
  const FEValuesExtractors::Scalar pressure (dim);

  scratch.navier_stokes_fe_values.reinit (cell);

  std::vector<Vector<double> > rhs_values (n_q_points,
                                           Vector<double>(dim+1));

  right_hand_side.vector_value_list (scratch.navier_stokes_fe_values
                                     .get_quadrature_points(),
                                     rhs_values);

  if (rebuild_navier_stokes_matrix)
    data.local_matrix = 0;
  data.local_rhs = 0;


  for (unsigned int q=0; q<n_q_points; ++q)
    {
      for (unsigned int k=0; k<dofs_per_cell; ++k)
        {
          scratch.phi_u[k] = scratch.navier_stokes_fe_values[velocities].value (k,q);
          if (rebuild_navier_stokes_matrix)
            {
              scratch.grads_phi_u[k] = scratch.navier_stokes_fe_values[velocities].symmetric_gradient(k,q);
              scratch.div_phi_u[k]   = scratch.navier_stokes_fe_values[velocities].divergence (k, q);
              scratch.phi_p[k]       = scratch.navier_stokes_fe_values[pressure].value (k, q);
            }
        }

      if (rebuild_navier_stokes_matrix == true)
        for (unsigned int i=0; i<dofs_per_cell; ++i)
          {
            for (unsigned int j=0; j<dofs_per_cell; ++j)
              data.local_matrix(i,j) += (EquationData::eta * 2 *
                                         (scratch.grads_phi_u[i] * scratch.grads_phi_u[j])
                                         - (scratch.div_phi_u[i] * scratch.phi_p[j])
                                         - (scratch.phi_p[i] * scratch.div_phi_u[j]))
                                        * scratch.navier_stokes_fe_values.JxW(q);

            unsigned int comp_i = navier_stokes_fe->system_to_component_index(i).first;
            if (comp_i<dim)
              data.local_rhs(i) += (rhs_values[q](comp_i) *
                                    scratch.phi_u[i][comp_i] *
                                    scratch.navier_stokes_fe_values.JxW(q));
          }
    }

  cell->get_dof_indices (data.local_dof_indices);
}



template <int dim>
void
NavierStokes<dim>::
copy_local_to_global_navier_stokes_system (const Assembly::CopyData::NavierStokesSystem<dim> &data)
{
  if (rebuild_navier_stokes_matrix == true)
    navier_stokes_constraints.distribute_local_to_global (data.local_matrix,
                                                          data.local_rhs,
                                                          data.local_dof_indices,
                                                          navier_stokes_matrix,
                                                          navier_stokes_rhs);
  else
    navier_stokes_constraints.distribute_local_to_global (data.local_rhs,
                                                          data.local_dof_indices,
                                                          navier_stokes_rhs);
}



template <int dim>
void NavierStokes<dim>::assemble_navier_stokes_system ()
{
  computing_timer.enter_section ("   Assemble Stokes system");

  if (rebuild_navier_stokes_matrix == true)
    navier_stokes_matrix=0;

  navier_stokes_rhs=0;

  const QGauss<dim> quadrature_formula(stokes_velocity_degree+1);

  typedef
  FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
  CellFilter;

  WorkStream::
  run (CellFilter (IteratorFilters::LocallyOwnedCell(),
                   navier_stokes_dof_handler->begin_active()),
       CellFilter (IteratorFilters::LocallyOwnedCell(),
                   navier_stokes_dof_handler->end()),
       std_cxx11::bind (&NavierStokes<dim>::
                        local_assemble_navier_stokes_system,
                        this,
                        std_cxx11::_1,
                        std_cxx11::_2,
                        std_cxx11::_3),
       std_cxx11::bind (&NavierStokes<dim>::
                        copy_local_to_global_navier_stokes_system,
                        this,
                        std_cxx11::_1),
       Assembly::Scratch::
       NavierStokesSystem<dim> (*navier_stokes_fe, mapping, quadrature_formula,
                                (update_values    |
                                 update_quadrature_points  |
                                 update_JxW_values |
                                 (rebuild_navier_stokes_matrix == true
                                  ?
                                  update_gradients
                                  :
                                  UpdateFlags(0)))),
       Assembly::CopyData::
       NavierStokesSystem<dim> (*navier_stokes_fe));

  if (rebuild_navier_stokes_matrix == true)
    navier_stokes_matrix.compress(VectorOperation::add);
  navier_stokes_rhs.compress(VectorOperation::add);

  rebuild_navier_stokes_matrix = false;

  pcout << std::endl;
  computing_timer.exit_section();
}

/* ------------------------ SOLVE ------------------------ */

template <int dim>
void NavierStokes<dim>::solve ()
{

  pcout << "   Solving Stokes system... " << std::flush;

  TrilinosWrappers::MPI::BlockVector
  distributed_navier_stokes_solution (navier_stokes_rhs);
  distributed_navier_stokes_solution = navier_stokes_solution;

  const unsigned int
  start = (distributed_navier_stokes_solution.block(0).size() +
           distributed_navier_stokes_solution.block(1).local_range().first),
          end   = (distributed_navier_stokes_solution.block(0).size() +
                   distributed_navier_stokes_solution.block(1).local_range().second);
  for (unsigned int i=start; i<end; ++i)
    if (navier_stokes_constraints.is_constrained (i))
      distributed_navier_stokes_solution(i) = 0;


  PrimitiveVectorMemory<TrilinosWrappers::MPI::BlockVector> mem;

  unsigned int n_iterations = 0;
  const double solver_tolerance = 1e-8 * navier_stokes_rhs.l2_norm();
  //SolverControl solver_control (30, solver_tolerance);
  SolverControl solver_control (1000, solver_tolerance); // Original value: 100

  // BEGIN : linear operator part

  SolverFGMRES<TrilinosWrappers::MPI::BlockVector>
  solver(solver_control, mem,
         SolverFGMRES<TrilinosWrappers::MPI::BlockVector>::
         AdditionalData(30, true));

  // SYSTEM MATRIX:
  auto S00 = linear_operator< TrilinosWrappers::MPI::Vector >( navier_stokes_matrix.block(0,0) );
  auto S01 = linear_operator< TrilinosWrappers::MPI::Vector >( navier_stokes_matrix.block(0,1) );
  auto S10 =  transpose_operator(S01);
  // auto S10 = linear_operator< TrilinosWrappers::MPI::Vector >( navier_stokes_matrix.block(1,0) );
  auto S11 = linear_operator< TrilinosWrappers::MPI::Vector >( navier_stokes_matrix.block(1,1) );

  // PRECONDITIONER:
  Amg_preconditioner->initialize(navier_stokes_preconditioner_matrix.block(0,0));
  Mp_preconditioner->initialize(navier_stokes_preconditioner_matrix.block(1,1));

  auto A_inv = linear_operator< TrilinosWrappers::MPI::Vector,
       TrilinosWrappers::MPI::Vector,
       decltype(navier_stokes_matrix.block(0,0)),
       decltype(*Amg_preconditioner)>
       (navier_stokes_matrix.block(0,0),  *Amg_preconditioner );

  auto Schur_inv  = linear_operator< TrilinosWrappers::MPI::Vector,
       TrilinosWrappers::MPI::Vector,
       decltype(navier_stokes_matrix.block(1,1)),
       decltype(*Mp_preconditioner)>
       (navier_stokes_matrix.block(1,1),  *Mp_preconditioner );


  // Assemble preconditioner:

  auto P00 = A_inv;
  auto P01 = 0 * S01;
  auto P10 = Schur_inv * S10 * A_inv;
  auto P11 = -1 * Schur_inv;

  // S = P_inv * S;

  // ASSEMBLE THE PROBLEM:
  auto S     = block_operator<2, 2, TrilinosWrappers::MPI::BlockVector >({{ {{ S00, S01 }} , 
                                                                            {{ S10, S11 }} }});
  auto P_inv = block_operator<2, 2, TrilinosWrappers::MPI::BlockVector >({{ {{ P00, P01 }} , 
                                                                            {{ P10, P11 }} }});

  auto S_buffer = S;
  auto prec  = identity_operator<TrilinosWrappers::MPI::BlockVector>(S_buffer.reinit_range_vector);


  /*
  // An attempt to solve the problem separating preconditioner.
  auto buffer_rhs = navier_stokes_rhs;
  P_inv.vmult(buffer_rhs, navier_stokes_rhs);
  S = P_inv*S;
  auto S_inv = inverse_operator(S, solver, prec);
  S_inv.vmult(solution, buffer_rhs);
  */

  auto S_inv = inverse_operator(S, solver, P_inv);

  // Initialise solution in order to have the same dimension of navier_stokes_fe
  solution =  navier_stokes_rhs;
  S_inv.vmult(solution, navier_stokes_rhs);

  
  // pcout << "navier_stokes_matrix :                 " << type(navier_stokes_matrix) << std::endl;
  // pcout << "navier_stokes_preconditioner_matrix :  " << type(navier_stokes_preconditioner_matrix) << std::endl;
  pcout << std::endl;
  pcout << " Solution size :                       " << solution.size() << std::endl;
  pcout << " RHS size :                            " << navier_stokes_rhs.size() << std::endl;
  pcout << " S 00 size:                            " << navier_stokes_matrix.block(0,0).m() << " x " << navier_stokes_matrix.block(0,0).n() << std::endl;
  pcout << " S 01 size:                            " << navier_stokes_matrix.block(0,1).m() << " x " << navier_stokes_matrix.block(0,1).n() << std::endl;
  pcout << " S 10 size:                            " << navier_stokes_matrix.block(1,0).m() << " x " << navier_stokes_matrix.block(1,0).n() << std::endl;
  pcout << " S 11 size:                            " << navier_stokes_matrix.block(1,1).m() << " x " << navier_stokes_matrix.block(1,1).n() << std::endl;
  // pcout << " S00 :                                 " << type(S00) << std::endl;

  n_iterations = solver_control.last_step();

  pcout << " iterations:                           " <<  n_iterations 
        << std::endl;

  // std::cout << "navier_stokes_preconditioner_matrix -> " << type(navier_stokes_preconditioner_matrix) << std::endl;
  // auto AMG = linear_operator( *Amg_preconditioner );
  // auto Mp  = linear_operator< TrilinosWrappers::MPI::Vector >( *Mp_preconditioner );

  // auto boo = linear_operator(navier_stokes_matrix);

  // END
  /*
  try
    {
      const LinearSolvers::BlockSchurPreconditioner<TrilinosWrappers::PreconditionAMG,
            TrilinosWrappers::PreconditionJacobi>
            preconditioner (navier_stokes_matrix, navier_stokes_preconditioner_matrix,
                            *Mp_preconditioner, *Amg_preconditioner,
                            false);

      SolverFGMRES<TrilinosWrappers::MPI::BlockVector>
      solver(solver_control, mem,
             SolverFGMRES<TrilinosWrappers::MPI::BlockVector>::
             AdditionalData(30, true));

      solver.solve(navier_stokes_matrix, distributed_navier_stokes_solution, navier_stokes_rhs,
                   preconditioner);

      n_iterations = solver_control.last_step();
    }

  catch (SolverControl::NoConvergence)
    {
      const LinearSolvers::BlockSchurPreconditioner<TrilinosWrappers::PreconditionAMG,
            TrilinosWrappers::PreconditionJacobi>
            preconditioner (navier_stokes_matrix, navier_stokes_preconditioner_matrix,
                            *Mp_preconditioner, *Amg_preconditioner,
                            true);

      SolverControl solver_control_refined (navier_stokes_matrix.m(), solver_tolerance);

      SolverFGMRES<TrilinosWrappers::MPI::BlockVector>
      solver(solver_control_refined, mem,
             SolverFGMRES<TrilinosWrappers::MPI::BlockVector>::
             AdditionalData(50, true));


      solver.solve(navier_stokes_matrix, distributed_navier_stokes_solution, navier_stokes_rhs,
                   preconditioner);

      n_iterations = (solver_control.last_step() +
                      solver_control_refined.last_step());
    }


  navier_stokes_constraints.distribute (distributed_navier_stokes_solution);
  */
  // navier_stokes_solution = distributed_navier_stokes_solution;
  // pcout << n_iterations  << " iterations."
  //       << std::endl;
}

/* ------------------------ OUTPUTS ------------------------ */

template <int dim>
void NavierStokes<dim>::output_results ()
{
  computing_timer.enter_section ("Postprocessing");

  Postprocessor postprocessor (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD),
                               navier_stokes_solution.block(1).minimal_value());

  DataOut<dim> data_out;
  data_out.attach_dof_handler (*navier_stokes_dof_handler);
  data_out.add_data_vector (navier_stokes_solution, postprocessor);
  // data_out.build_patches ();

  static int out_index=0;
  const std::string filename = ("solution-" +
                                Utilities::int_to_string (out_index, 5) +
                                "." +
                                Utilities::int_to_string
                                (triangulation->locally_owned_subdomain(), 4) +
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

/* ------------------------ MESH AND GRID ------------------------ */

template <int dim>
void NavierStokes<dim>::refine_mesh ()
{
  switch (refinement_mode)
    {
    case global_refinement:
    {
      triangulation->refine_global (1);
      // triangulation->refine_global (2);
      break;
    }

    case adaptive_refinement:
    {
      triangulation->refine_global (1);
      // triangulation->refine_global (2);
      break;
    }

    default:
    {
      Assert (false, ExcNotImplemented());
    }
    }
}


template <int dim>
void NavierStokes<dim>::make_grid_fe()
{

  triangulation = SP(pgg.distributed(MPI_COMM_WORLD));
  navier_stokes_dof_handler = SP(new DoFHandler<dim>(*triangulation));

  // navier_stokes_dof_handler = new DoFHandler<dim>(*triangulation);

  navier_stokes_fe=SP(fe_builder());

  // std::cout << "navier stokes fe : " << *navier_stokes_fe.get_degree() << std::endl;
  triangulation->refine_global (initial_global_refinement);

  // Compute the velocity degree using the prm file:
  // the first component is the velocity and therfore
  // it is taken.
  stokes_velocity_degree = navier_stokes_fe->degree;

  triangulation->refine_global (initial_global_refinement);
}

/* ------------------------ ERRORS ------------------------ */

template <int dim>
void NavierStokes<dim>::process_solution ()
{
  eh.error_from_exact(*navier_stokes_dof_handler, navier_stokes_solution, Solution<dim>(), refinement_mode);
}

/* ------------------------ RUN ------------------------ */

template <int dim>
void NavierStokes<dim>::run ()
{

  const unsigned int n_cycles = (refinement_mode==global_refinement)?5:9;
  for (unsigned int cycle=0; cycle<n_cycles; ++cycle)
    {
      if (cycle == 0)
        {
          make_grid_fe ();
        }
      else
        refine_mesh ();

      setup_dofs ();
      assemble_navier_stokes_system ();
      build_navier_stokes_preconditioner ();
      solve ();
      process_solution ();
      output_results ();
    }

  // std::ofstream f("errors.txt");
  eh.output_table(pcout, refinement_mode);
  // f.close();
}


// template class NavierStokes<1>;
template class NavierStokes<2>;
// template class NavierStokes<3>;
