#include "n_fields_problem.h"
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/numerics/solution_transfer.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/fe/mapping_q_eulerian.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/trilinos_block_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
//
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/block_linear_operator.h>
#include <deal.II/lac/linear_operator.h>

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

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/distributed/solution_transfer.h>
//#include <deal.II/base/index_set.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include <typeinfo>
#include <fstream>
#include <iostream>
#include <sstream>
#include <limits>
#include <numeric>
#include <locale>
#include <string>
#include <math.h>

#include "equation_data.h"

using namespace dealii;

/* ------------------------ PARAMETERS ------------------------ */

template <int dim, int spacedim, int n_components>
void
NFieldsProblem<dim, spacedim, n_components>::
declare_parameters (ParameterHandler &prm)
{
  add_parameter(  prm,
                  &initial_global_refinement,
                  "Initial global refinement",
                  "1",
                  Patterns::Integer (0));

  add_parameter(  prm,
                  &n_cycles,
                  "Number of cycles",
                  "3",
                  Patterns::Integer (0));

  add_parameter(  prm,
                  &max_time_iterations,
                  "Maximum number of time steps",
                  "10000",
                  Patterns::Integer (0));

  add_parameter(  prm,
                  &timer_file_name,
                  "Timer output file",
                  "timer.txt",
                  Patterns::FileName());


  add_parameter(  prm,
                  &adaptive_refinement,
                  "Adaptive refinement",
                  "true",
                  Patterns::Bool());



}

/* ------------------------ CONSTRUCTORS ------------------------ */

template <int dim, int spacedim, int n_components>
NFieldsProblem<dim, spacedim, n_components>::NFieldsProblem (const Interface<dim, spacedim, n_components> &energy,
    const MPI_Comm &communicator)
  :
  OdeArgument<VEC>(communicator),
  comm(communicator),
  energy(energy),
  pcout (std::cout,
         (Utilities::MPI::this_mpi_process(comm)
          == 0)),
  timer_outfile("timer.txt"),
  tcout (timer_outfile,
         (Utilities::MPI::this_mpi_process(comm)
          == 0)),
  computing_timer (comm,
                   tcout,
                   TimerOutput::summary,
                   TimerOutput::wall_times),

  eh("Error Tables", energy.get_component_names(),
     print(std::vector<std::string>(n_components, "L2,H1"), ";")),

  pgg("Domain"),

  exact_solution("Exact solution"),
  initial_solution("Initial solution"),
  initial_solution_dot("Initial solution_dot"),

  data_out("Output Parameters", "vtu"),
  dae(*this)

{}


/* ------------------------ DEGREE OF FREEDOM ------------------------ */

template <int dim, int spacedim, int n_components>
void NFieldsProblem<dim, spacedim, n_components>::setup_dofs (const bool &first_run)
{
  computing_timer.enter_section("Setup dof systems");
  std::vector<unsigned int> sub_blocks = energy.get_component_blocks();
  dof_handler->distribute_dofs (*fe);
  DoFRenumbering::component_wise (*dof_handler, sub_blocks);

  mapping = energy.get_mapping(*dof_handler, solution);

  std::vector<types::global_dof_index> dofs_per_block (energy.n_blocks());
  DoFTools::count_dofs_per_block (*dof_handler, dofs_per_block,
                                  sub_blocks);

  std::locale s = pcout.get_stream().getloc();
  pcout.get_stream().imbue(std::locale(""));
  pcout << "Number of active cells: "
        << triangulation->n_global_active_cells()
        << " (on "
        << triangulation->n_levels()
        << " levels)"
        << std::endl
        << "Number of degrees of freedom: "
        << dof_handler->n_dofs()
        << "(" << print(dofs_per_block,"+") << ")"
        << std::endl
        << std::endl;
  pcout.get_stream().imbue(s);


  partitioning.resize(0);
  relevant_partitioning.resize(0);

  IndexSet relevant_set;
  {
    IndexSet index_set = dof_handler->locally_owned_dofs();
    for (unsigned int i=0; i<energy.n_blocks(); ++i)
      partitioning.push_back(index_set.get_view( std::accumulate(dofs_per_block.begin(), dofs_per_block.begin()+i, 0),
                                                 std::accumulate(dofs_per_block.begin(), dofs_per_block.begin()+i+1, 0)));

    DoFTools::extract_locally_relevant_dofs (*dof_handler,
                                             relevant_set);

    for (unsigned int i=0; i<energy.n_blocks(); ++i)
      relevant_partitioning.push_back(relevant_set.get_view(std::accumulate(dofs_per_block.begin(), dofs_per_block.begin()+i, 0),
                                                            std::accumulate(dofs_per_block.begin(), dofs_per_block.begin()+i+1, 0)));
  }
  constraints.clear ();
  constraints.reinit (relevant_set);

  DoFTools::make_hanging_node_constraints (*dof_handler,
                                           constraints);

  energy.apply_dirichlet_bcs(*dof_handler, constraints);

  constraints.close ();

  reinit_jacobian_matrix (partitioning, relevant_partitioning);
  reinit_jacobian_preconditioner (partitioning, relevant_partitioning);


  distributed_solution.reinit (partitioning, relevant_partitioning, comm);
  distributed_solution_dot.reinit (partitioning, relevant_partitioning, comm);
  solution.reinit (partitioning, comm);
  solution_dot.reinit (partitioning, comm);

  if (first_run)
    {
      if (fe->has_support_points())
        {
          VectorTools::interpolate(*mapping, *dof_handler, initial_solution, solution);
          VectorTools::interpolate(*mapping, *dof_handler, initial_solution_dot, solution_dot);
        }
      else
        {
          const QGauss<dim> quadrature_formula(fe->degree+1);
          //VectorTools::project(*mapping, *dof_handler, constraints, quadrature_formula, initial_solution, solution);
          //VectorTools::project(*mapping, *dof_handler, constraints, quadrature_formula, initial_solution_dot, solution_dot);
        }

    }

  // Store a global partitioning to be used anywhere we need to know
  // what global dofs we own. This is the sum of partitioning[i].
  global_partioning = solution.locally_owned_elements();

  computing_timer.exit_section();
}

/* ------------------------ SETUP MATRIX ------------------------ */

template <int dim, int spacedim, int n_components>
void NFieldsProblem<dim, spacedim, n_components>::
reinit_jacobian_matrix (const std::vector<IndexSet> &partitioning,
                        const std::vector<IndexSet> &relevant_partitioning)
{
  jacobian_matrix.clear ();

  TrilinosWrappers::BlockSparsityPattern sp(partitioning, partitioning,
                                            relevant_partitioning,
                                            comm);

  Table<2,DoFTools::Coupling> coupling = energy.get_coupling();

  DoFTools::make_sparsity_pattern (*dof_handler,
                                   coupling, sp,
                                   constraints, false,
                                   Utilities::MPI::
                                   this_mpi_process(comm));
  sp.compress();

  jacobian_matrix.reinit (sp);
}

/* ------------------------ PRECONDITIONER ------------------------ */

template <int dim, int spacedim, int n_components>
void NFieldsProblem<dim, spacedim, n_components>::
reinit_jacobian_preconditioner(const std::vector<IndexSet> &partitioning,
                               const std::vector<IndexSet> &relevant_partitioning)
{
  if (energy.get_jacobian_preconditioner_flags() != update_default)
    {
      jacobian_preconditioner_matrix.clear ();

      TrilinosWrappers::BlockSparsityPattern sp(partitioning, partitioning,
                                                relevant_partitioning,
                                                comm);

      Table<2,DoFTools::Coupling> coupling = energy.get_preconditioner_coupling();

      DoFTools::make_sparsity_pattern (*dof_handler,
                                       coupling, sp,
                                       constraints, false,
                                       Utilities::MPI::
                                       this_mpi_process(comm));
      sp.compress();

      jacobian_preconditioner_matrix.reinit (sp);
    }
}


template <int dim, int spacedim, int n_components>
void NFieldsProblem<dim, spacedim, n_components>::assemble_jacobian_matrix (const double t,
    const VEC &solution,
    const VEC &solution_dot,
    const double alpha)
{
  computing_timer.enter_section ("   Assemble system jacobian");

  jacobian_matrix = 0;

  const QGauss<dim> quadrature_formula(fe->degree+1);
  const QGauss<dim-1> face_quadrature_formula(fe->degree+1);
  SAKData system_data;

  distributed_solution = solution;
  distributed_solution_dot = solution_dot;

  energy.initialize_data(distributed_solution,
                         distributed_solution_dot, t, alpha);


  auto local_copy = [ this ]
                    (const SystemCopyData &data)
  {
    this->constraints.distribute_local_to_global (data.local_matrix,
                                                  data.local_dof_indices,
                                                  this->jacobian_matrix);
  };

  auto local_assemble = [ this ]
                        (const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                         Scratch &scratch,
                         SystemCopyData &data)
  {
    this->energy.assemble_local_system(cell, scratch, data);
  };

  typedef
  FilteredIterator<typename DoFHandler<dim,spacedim>::active_cell_iterator>
  CellFilter;
  WorkStream::
  run (CellFilter (IteratorFilters::LocallyOwnedCell(),
                   dof_handler->begin_active()),
       CellFilter (IteratorFilters::LocallyOwnedCell(),
                   dof_handler->end()),
       local_assemble,
       local_copy,
       Scratch(*mapping,
               *fe,
               quadrature_formula,
               energy.get_jacobian_flags(),
               face_quadrature_formula,
               energy.get_face_flags()),
       Assembly::CopyData::
       NFieldsSystem<dim,spacedim> (*fe));

  jacobian_matrix.compress(VectorOperation::add);

//  pcout << std::endl;

  auto id = solution.locally_owned_elements();
  for (unsigned int i=0; i<id.n_elements(); ++i)
    {
      auto j = id.nth_index_in_set(i);
      if (constraints.is_constrained(j))
        jacobian_matrix.set(j, j, 1.0);
    }
  jacobian_matrix.compress(VectorOperation::insert);

  computing_timer.exit_section();
}




template <int dim, int spacedim, int n_components>
void NFieldsProblem<dim, spacedim, n_components>::assemble_jacobian_preconditioner (const double t,
    const VEC &solution,
    const VEC &solution_dot,
    const double alpha)
{
  if (energy.get_jacobian_preconditioner_flags() != update_default)
    {
      computing_timer.enter_section ("   Build preconditioner");
      jacobian_preconditioner_matrix = 0;

      const QGauss<dim> quadrature_formula(fe->degree+1);
      const QGauss<dim-1> face_quadrature_formula(fe->degree+1);

      typedef
      FilteredIterator<typename DoFHandler<dim,spacedim>::active_cell_iterator>
      CellFilter;

      SAKData preconditioner_data;

      distributed_solution = solution;
      distributed_solution_dot = solution_dot;

      energy.initialize_data(distributed_solution,
                             distributed_solution_dot, t, alpha);


      auto local_copy = [this]
                        (const PreconditionerCopyData &data)
      {
        this->constraints.distribute_local_to_global (data.local_matrix,
                                                      data.local_dof_indices,
                                                      this->jacobian_preconditioner_matrix);
      };

      auto local_assemble = [ this ]
                            (const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                             Scratch &scratch,
                             PreconditionerCopyData &data)
      {
        this->energy.assemble_local_preconditioner(cell, scratch, data);
      };



      WorkStream::
      run (CellFilter (IteratorFilters::LocallyOwnedCell(),
                       dof_handler->begin_active()),
           CellFilter (IteratorFilters::LocallyOwnedCell(),
                       dof_handler->end()),
           local_assemble,
           local_copy,
           Scratch (*mapping,
                    *fe, quadrature_formula,
                    energy.get_jacobian_preconditioner_flags(),
                    face_quadrature_formula,
                    UpdateFlags(0)),
           Assembly::CopyData::
           NFieldsPreconditioner<dim,spacedim> (*fe));

      jacobian_preconditioner_matrix.compress(VectorOperation::add);
      computing_timer.exit_section();
    }
}

/* ------------------------ MESH AND GRID ------------------------ */

template <int dim, int spacedim, int n_components>
void NFieldsProblem<dim, spacedim, n_components>::refine_mesh ()
{
  computing_timer.enter_section ("   Mesh refinement");
  if (adaptive_refinement)
    {
      Vector<float> estimated_error_per_cell (triangulation->n_active_cells());
      KellyErrorEstimator<dim>::estimate (*dof_handler,
                                          QGauss<dim-1>(fe->degree+1),
                                          typename FunctionMap<dim>::type(),
                                          distributed_solution,
                                          estimated_error_per_cell,
                                          ComponentMask(),
                                          0,
                                          0,
                                          triangulation->locally_owned_subdomain());

      parallel::distributed::GridRefinement::
      refine_and_coarsen_fixed_fraction (*triangulation,
                                         estimated_error_per_cell,
                                         0.3, 0.1);


    }

  parallel::distributed::SolutionTransfer<dim,TrilinosWrappers::MPI::BlockVector> sol_tr(*dof_handler);
  parallel::distributed::SolutionTransfer<dim,TrilinosWrappers::MPI::BlockVector> sol_dot_tr(*dof_handler);
  TrilinosWrappers::MPI::BlockVector sol (distributed_solution);
  TrilinosWrappers::MPI::BlockVector sol_dot (distributed_solution_dot);
  sol = solution;
  sol_dot = solution_dot;

  triangulation->prepare_coarsening_and_refinement();
  sol_tr.prepare_for_coarsening_and_refinement (sol);
  sol_dot_tr.prepare_for_coarsening_and_refinement(sol_dot);
  if (adaptive_refinement)
    triangulation->execute_coarsening_and_refinement ();
  else
    triangulation->refine_global (1);

  setup_dofs(false);

  TrilinosWrappers::MPI::BlockVector tmp (solution);
  TrilinosWrappers::MPI::BlockVector tmp_dot (solution_dot);

  sol_tr.interpolate (tmp);
  sol_dot_tr.interpolate (tmp_dot);
  solution = tmp;
  solution_dot = tmp_dot;

  computing_timer.exit_section();
}


template <int dim, int spacedim, int n_components>
void NFieldsProblem<dim, spacedim, n_components>::make_grid_fe()
{
  triangulation = SP(pgg.distributed(comm));
  dof_handler = SP(new DoFHandler<dim,spacedim>(*triangulation));
  fe=SP(energy());
  triangulation->refine_global (initial_global_refinement);
}

/* ------------------------ ERRORS ------------------------ */

template <int dim, int spacedim, int n_components>
void NFieldsProblem<dim, spacedim, n_components>::process_solution ()
{
	//constraints.distribute(solution);
  eh.error_from_exact(*dof_handler, solution, exact_solution);
  eh.output_table(pcout);
}

/* ------------------------ RUN ------------------------ */

template <int dim, int spacedim, int n_components>
void NFieldsProblem<dim, spacedim, n_components>::run ()
{
//  if(timer_file_name != "")
//    timer_outfile.open(timer_file_name.c_str());
//  else
//    timer_outfile.open("/dev/null");
//

  for (current_cycle=0; current_cycle<n_cycles; ++current_cycle)
    {
      if (current_cycle == 0)
        {
          make_grid_fe();
          setup_dofs(true);
        }
      else
        refine_mesh();

      dae.start_ode(solution, solution_dot, max_time_iterations);
      distributed_solution = solution;
      eh.error_from_exact(*mapping, *dof_handler, distributed_solution, exact_solution);
    }

  eh.output_table(pcout);

  // std::ofstream f("errors.txt");
  computing_timer.print_summary();
  timer_outfile.close();
  // f.close();
}

/*** ODE Argument Interface ***/

template <int dim, int spacedim, int n_components>
shared_ptr<VEC>
NFieldsProblem<dim, spacedim, n_components>::create_new_vector() const
{
  shared_ptr<VEC> ret = SP(new VEC(solution));
  return ret;
}


template <int dim, int spacedim, int n_components>
unsigned int
NFieldsProblem<dim, spacedim, n_components>::n_dofs() const
{
  return dof_handler->n_dofs();
}


template <int dim, int spacedim, int n_components>
void
NFieldsProblem<dim, spacedim, n_components>::output_step(const double /* t */,
                                                         const VEC &solution,
                                                         const VEC &solution_dot,
                                                         const unsigned int step_number,
                                                         const double /* h */ )
{
  computing_timer.enter_section ("Postprocessing");
  TrilinosWrappers::MPI::BlockVector        tmp(solution);

	tmp = solution;
	constraints.distribute(tmp);
  distributed_solution = solution;
  distributed_solution_dot = solution_dot;

  std::stringstream suffix;
  suffix << "." << current_cycle << "." << step_number;
  data_out.prepare_data_output( *dof_handler,
                                suffix.str());
  data_out.add_data_vector (distributed_solution, energy.get_component_names());
  std::vector<std::string> sol_dot_names =
    Utilities::split_string_list( energy.get_component_names());
  for (auto &name : sol_dot_names)
    {
      name += "_dot";
    }
  data_out.add_data_vector (distributed_solution_dot, print(sol_dot_names,","));

  data_out.write_data_and_clear("",*mapping);

  computing_timer.exit_section ();
}



template <int dim, int spacedim, int n_components>
bool
NFieldsProblem<dim, spacedim, n_components>::solver_should_restart(const double t,
    const VEC &solution,
    const VEC &solution_dot,
    const unsigned int step_number,
    const double h)
{
  return false;
}


template <int dim, int spacedim, int n_components>
int
NFieldsProblem<dim, spacedim, n_components>::residual(const double t,
                                                      const VEC &solution,
                                                      const VEC &solution_dot,
                                                      VEC &dst) const
{
  computing_timer.enter_section ("Residual");
  energy.set_time(t);
  const QGauss<dim> quadrature_formula(fe->degree+1);
  const QGauss<dim-1> face_quadrature_formula(fe->degree+1);

  SAKData residual_data;

  distributed_solution = solution;
  distributed_solution_dot = solution_dot;

  energy.initialize_data(distributed_solution,
                         distributed_solution_dot, t, 0.0);

  dst = 0;

  auto local_copy = [&dst, this] (const SystemCopyData &data)
  {
    this->constraints.distribute_local_to_global (data.double_residual, //data.local_rhs,
                                                  data.local_dof_indices,
                                                  dst);
  };

  auto local_assemble = [ this ]
                        (const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                         Scratch &scratch,
                         SystemCopyData &data)
  {
    cell->get_dof_indices (data.local_dof_indices);
    data.local_rhs = 0;
    this->energy.get_system_residual(cell, scratch, data, data.double_residual);
  };

  typedef
  FilteredIterator<typename DoFHandler<dim,spacedim>::active_cell_iterator>
  CellFilter;
  WorkStream::
  run (CellFilter (IteratorFilters::LocallyOwnedCell(),
                   dof_handler->begin_active()),
       CellFilter (IteratorFilters::LocallyOwnedCell(),
                   dof_handler->end()),
       local_assemble,
       local_copy,
       Scratch(*mapping,
               *fe,
               quadrature_formula,
               energy.get_jacobian_flags(),
               face_quadrature_formula,
               energy.get_face_flags()),
       SystemCopyData(*fe));

  // constraints.distribute(dst);

  dst.compress(VectorOperation::add);

  auto id = solution.locally_owned_elements();
  for (unsigned int i=0; i<id.n_elements(); ++i)
    {
      auto j = id.nth_index_in_set(i);
      if (constraints.is_constrained(j))
        dst[j] = solution(j)-constraints.get_inhomogeneity(j);
    }

  dst.compress(VectorOperation::insert);
  computing_timer.exit_section();
  return 0;
}


template <int dim, int spacedim, int n_components>
int
NFieldsProblem<dim, spacedim, n_components>::solve_jacobian_system(const double t,
    const VEC &y,
    const VEC &y_dot,
    const VEC &,
    const double alpha,
    const VEC &src,
    VEC &dst) const
{
  computing_timer.enter_section ("   Solve system");
	dst=solution;
  set_constrained_dofs_to_zero(dst);

  unsigned int n_iterations = 0;
  const double solver_tolerance = 1e-8;

  PrimitiveVectorMemory<TrilinosWrappers::MPI::BlockVector> mem;
  SolverControl solver_control (30, solver_tolerance);
  SolverControl solver_control_refined (jacobian_matrix.m(), solver_tolerance);

  SolverFGMRES<TrilinosWrappers::MPI::BlockVector>
  solver(solver_control, mem,
         SolverFGMRES<TrilinosWrappers::MPI::BlockVector>::
         AdditionalData(30, true));

  SolverFGMRES<TrilinosWrappers::MPI::BlockVector>
  solver_refined(solver_control_refined, mem,
                 SolverFGMRES<TrilinosWrappers::MPI::BlockVector>::
                 AdditionalData(50, true));

  auto S_inv         = inverse_operator(jacobian_op, solver, jacobian_preconditioner_op);
  auto S_inv_refined = inverse_operator(jacobian_op, solver_refined, jacobian_preconditioner_op);
  try
    {
      S_inv.vmult(dst, src);
      n_iterations = solver_control.last_step();
    }
  catch ( SolverControl::NoConvergence )
    {
      S_inv_refined.vmult(dst, src);
      n_iterations = (solver_control.last_step() +
                      solver_control_refined.last_step());
    }

  set_constrained_dofs_to_zero(dst);
//   constraints.distribute (dst);

//  pcout << std::endl;
//  pcout << " iterations:                           " <<  n_iterations
//        << std::endl;
//  pcout << std::endl;
  computing_timer.exit_section();
  return 0;
}


template <int dim, int spacedim, int n_components>
int
NFieldsProblem<dim, spacedim, n_components>::setup_jacobian(const double t,
                                                            const VEC &src_yy,
                                                            const VEC &src_yp,
                                                            const VEC &,
                                                            const double alpha)
{
  computing_timer.enter_section ("   Setup Jacobian");
  assemble_jacobian_matrix(t, src_yy, src_yp, alpha);
  assemble_jacobian_preconditioner(t, src_yy, src_yp, alpha);

  energy.compute_system_operators(*dof_handler,
                                  jacobian_matrix, jacobian_preconditioner_matrix,
                                  jacobian_op, jacobian_preconditioner_op);
  computing_timer.exit_section();

  return 0;
}



template <int dim, int spacedim, int n_components>
VEC &
NFieldsProblem<dim, spacedim, n_components>::differential_components() const
{
  static VEC diff_comps;
  diff_comps.reinit(solution);
  std::vector<unsigned int> block_diff = energy.get_differential_blocks();
  for (unsigned int i=0; i<block_diff.size(); ++i)
    diff_comps.block(i) = block_diff[i];

  set_constrained_dofs_to_zero(diff_comps);
  return diff_comps;
}



template <int dim, int spacedim, int n_components>
void
NFieldsProblem<dim, spacedim, n_components>::set_constrained_dofs_to_zero(VEC &v) const
{
  for (unsigned int i=0; i<global_partioning.n_elements(); ++i)
    {
      auto j = global_partioning.nth_index_in_set(i);
      if (constraints.is_constrained(j))
        v[j] = 0;
    }
}

// template class NFieldsProblem<1>;

//template class NFieldsProblem<1,1,1>;
//template class NFieldsProblem<1,1,2>;
//template class NFieldsProblem<1,1,3>;
//template class NFieldsProblem<1,1,4>;

// template class NFieldsProblem<1,2,1>;
// template class NFieldsProblem<1,2,2>;
// template class NFieldsProblem<1,2,3>;
// template class NFieldsProblem<1,2,4>;

template class NFieldsProblem<2,2,1>;
template class NFieldsProblem<2,2,2>;
template class NFieldsProblem<2,2,3>;
template class NFieldsProblem<2,2,4>;


// template class NFieldsProblem<2,3,1>;
// template class NFieldsProblem<2,3,2>;
// template class NFieldsProblem<2,3,3>;
// template class NFieldsProblem<2,3,4>;


template class NFieldsProblem<3,3,1>;
template class NFieldsProblem<3,3,2>;
template class NFieldsProblem<3,3,3>;
template class NFieldsProblem<3,3,4>;
// template class NFieldsProblem<3>;
