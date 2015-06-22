#include "n_fields_problem.h"
#include <deal.II/lac/trilinos_precondition.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
// #include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/work_stream.h>

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

#include <deal.II/distributed/solution_transfer.h>
// #include <deal.II/base/index_set.h>
#include <deal.II/distributed/tria.h>
// #include <deal.II/distributed/grid_refinement.h>

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
}

/* ------------------------ CONSTRUCTORS ------------------------ */

template <int dim, int spacedim, int n_components>
NFieldsProblem<dim, spacedim, n_components>::NFieldsProblem (const Interface<dim, spacedim, n_components> &energy)
  :
  energy(energy),
  pcout (std::cout,
         (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
          == 0)),
  tcout (timer_outfile,
         (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
          == 0)),

  mapping (4),

  rebuild_matrix (true),
  rebuild_preconditioner (true),

  computing_timer (MPI_COMM_WORLD,
                   tcout,
                   TimerOutput::summary,
                   TimerOutput::wall_times),

  eh("Error Tables", energy.get_component_names(),
     print(std::vector<std::string>(n_components, "L2,H1"), ";")),

  pgg("Domain"),


  boundary_conditions("Dirichlet boundary conditions"),

  right_hand_side("Right-hand side force"),
  exact_solution("Exact solution"),

  data_out("Output Parameters", "vtu")

{}


/* ------------------------ DEGREE OF FREEDOM ------------------------ */

template <int dim, int spacedim, int n_components>
void NFieldsProblem<dim, spacedim, n_components>::setup_dofs ()
{
  computing_timer.enter_section("Setup dof systems");

  std::vector<unsigned int> sub_blocks = energy.get_component_blocks();
  dof_handler->distribute_dofs (*fe);
  DoFRenumbering::component_wise (*dof_handler, sub_blocks);

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


  std::vector<IndexSet> partitioning, relevant_partitioning;
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

  {
    constraints.clear ();
    constraints.reinit (relevant_set);

    DoFTools::make_hanging_node_constraints (*dof_handler,
                                             constraints);

    FEValuesExtractors::Vector velocity_components(0);
    //boundary_conditions.set_time(time_step*time_step_number);
    VectorTools::interpolate_boundary_values (*dof_handler,
                                              0,
                                              boundary_conditions,
                                              constraints,
                                              fe->component_mask(velocity_components));
    constraints.close ();
  }

  setup_matrix (partitioning, relevant_partitioning);
  setup_preconditioner (partitioning, relevant_partitioning);

  rhs.reinit (partitioning, relevant_partitioning,
              MPI_COMM_WORLD, true);
  solution.reinit (relevant_partitioning, MPI_COMM_WORLD);
  old_solution.reinit (solution);
  old_solution.reinit (solution);

  rebuild_matrix              = true;
  rebuild_preconditioner      = true;

  computing_timer.exit_section();
}

/* ------------------------ SETUP MATRIX ------------------------ */

template <int dim, int spacedim, int n_components>
void NFieldsProblem<dim, spacedim, n_components>::
setup_matrix (const std::vector<IndexSet> &partitioning,
              const std::vector<IndexSet> &relevant_partitioning)
{
  matrix.clear ();

  TrilinosWrappers::BlockSparsityPattern sp(partitioning, partitioning,
                                            relevant_partitioning,
                                            MPI_COMM_WORLD);

  Table<2,DoFTools::Coupling> coupling = energy.get_coupling();

  DoFTools::make_sparsity_pattern (*dof_handler,
                                   coupling, sp,
                                   constraints, false,
                                   Utilities::MPI::
                                   this_mpi_process(MPI_COMM_WORLD));
  sp.compress();

  matrix.reinit (sp);
}

/* ------------------------ PRECONDITIONER ------------------------ */

template <int dim, int spacedim, int n_components>
void NFieldsProblem<dim, spacedim, n_components>::
setup_preconditioner (const std::vector<IndexSet> &partitioning,
                      const std::vector<IndexSet> &relevant_partitioning)
{
  preconditioner_matrix.clear ();

  TrilinosWrappers::BlockSparsityPattern sp(partitioning, partitioning,
                                            relevant_partitioning,
                                            MPI_COMM_WORLD);

  Table<2,DoFTools::Coupling> coupling = energy.get_preconditioner_coupling();

  DoFTools::make_sparsity_pattern (*dof_handler,
                                   coupling, sp,
                                   constraints, false,
                                   Utilities::MPI::
                                   this_mpi_process(MPI_COMM_WORLD));
  sp.compress();

  preconditioner_matrix.reinit (sp);
}

template <int dim, int spacedim, int n_components>
void
NFieldsProblem<dim, spacedim, n_components>::
local_assemble_preconditioner (const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                               Assembly::Scratch::NFields<dim,spacedim> &scratch,
                               Assembly::CopyData::NFieldsPreconditioner<dim,spacedim> &data)
{
  const unsigned int   dofs_per_cell   = fe->dofs_per_cell;
  const unsigned int   n_q_points      = scratch.fe_values.n_quadrature_points;

  const FEValuesExtractors::Vector velocities (0);
  const FEValuesExtractors::Scalar pressure (dim);

  scratch.fe_values.reinit (cell);
  cell->get_dof_indices (data.local_dof_indices);

  data.local_matrix = 0;

  energy.get_preconditioner_residual(cell, scratch, data, data.sacado_residual);

  for (unsigned int i=0; i<dofs_per_cell; ++i)
    for (unsigned int j=0; j<dofs_per_cell; ++j)
      data.local_matrix(i,j) = data.sacado_residual[i].dx(j);

}

template <int dim, int spacedim, int n_components>
void
NFieldsProblem<dim, spacedim, n_components>::
copy_local_to_global_preconditioner (const Assembly::CopyData::NFieldsPreconditioner<dim,spacedim> &data)
{
  constraints.distribute_local_to_global (  data.local_matrix,
                                            data.local_dof_indices,
                                            preconditioner_matrix);
}

template <int dim, int spacedim, int n_components>
void
NFieldsProblem<dim, spacedim, n_components>::assemble_preconditioner ()
{
  preconditioner_matrix = 0;

  const QGauss<dim> quadrature_formula(fe->degree+1);

  typedef
  FilteredIterator<typename DoFHandler<dim,spacedim>::active_cell_iterator>
  CellFilter;

  SAKData preconditioner_data;

  std::vector<const TrilinosWrappers::MPI::BlockVector *> sols;
  sols.push_back(&solution);
  energy.initialize_data(fe->dofs_per_cell,
                         quadrature_formula.size(),
                         sols,preconditioner_data);

  WorkStream::
  run (CellFilter (IteratorFilters::LocallyOwnedCell(),
                   dof_handler->begin_active()),
       CellFilter (IteratorFilters::LocallyOwnedCell(),
                   dof_handler->end()),
       std_cxx11::bind (&NFieldsProblem<dim, spacedim, n_components>::
                        local_assemble_preconditioner,
                        this,
                        std_cxx11::_1,
                        std_cxx11::_2,
                        std_cxx11::_3),
       std_cxx11::bind (&NFieldsProblem<dim, spacedim, n_components>::
                        copy_local_to_global_preconditioner,
                        this,
                        std_cxx11::_1),
       Assembly::Scratch::
       NFields<dim,spacedim> (preconditioner_data,
                              *fe, quadrature_formula,
                              mapping,
                              update_JxW_values |
                              update_values |
                              update_gradients),
       Assembly::CopyData::
       NFieldsPreconditioner<dim,spacedim> (*fe));

  preconditioner_matrix.compress(VectorOperation::add);
}


template <int dim, int spacedim, int n_components>
void
NFieldsProblem<dim, spacedim, n_components>::build_preconditioner ()
{
  if (rebuild_preconditioner == false)
    return;

  computing_timer.enter_section ("   Build preconditioner");
  pcout << "   Rebuilding preconditioner..." << std::flush;

  assemble_preconditioner ();

  rebuild_preconditioner = false;

  pcout << std::endl;
  computing_timer.exit_section();
}

/* ------------------------ ASSEMBLE THE SYSTEM ------------------------ */

template <int dim, int spacedim, int n_components>
void
NFieldsProblem<dim, spacedim, n_components>::
local_assemble_system (const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                       Assembly::Scratch::NFields<dim,spacedim> &scratch,
                       Assembly::CopyData::NFieldsSystem<dim,spacedim> &data)
{
  const unsigned int dofs_per_cell = scratch.fe_values.get_fe().dofs_per_cell;
  scratch.fe_values.reinit (cell);
  cell->get_dof_indices (data.local_dof_indices);

  data.local_rhs = 0;

  if (rebuild_matrix == true)
    {
      data.local_matrix = 0;
      energy.get_system_residual(cell, scratch, data, data.sacado_residual);

      for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
          for (unsigned int j=0; j<dofs_per_cell; ++j)
            data.local_matrix(i,j) += data.sacado_residual[i].dx(j);

          data.local_rhs(i) -= data.sacado_residual[i].val();
        }
    }
  else
    {
      energy.get_system_residual(cell, scratch, data, data.double_residual);

      for (unsigned int i=0; i<dofs_per_cell; ++i)
        data.local_rhs(i) -= data.double_residual[i];
    }


}



template <int dim, int spacedim, int n_components>
void
NFieldsProblem<dim, spacedim, n_components>::
copy_local_to_global_system (const Assembly::CopyData::NFieldsSystem<dim,spacedim> &data)
{
  if (rebuild_matrix == true)
    constraints.distribute_local_to_global (data.local_matrix,
                                            data.local_rhs,
                                            data.local_dof_indices,
                                            matrix,
                                            rhs);
  else
    constraints.distribute_local_to_global (data.local_rhs,
                                            data.local_dof_indices,
                                            rhs);
}



template <int dim, int spacedim, int n_components>
void NFieldsProblem<dim, spacedim, n_components>::assemble_system ()
{
  computing_timer.enter_section ("   Assemble system");

  if (rebuild_matrix == true)
    matrix=0;

  rhs=0;

  const QGauss<dim> quadrature_formula(fe->degree+1);
  SAKData system_data;
  std::vector<const TrilinosWrappers::MPI::BlockVector *> sols;
  sols.push_back(&solution);
  energy.initialize_data(fe->dofs_per_cell,
                         quadrature_formula.size(),
                         sols, system_data);

  typedef
  FilteredIterator<typename DoFHandler<dim,spacedim>::active_cell_iterator>
  CellFilter;
  WorkStream::
  run (CellFilter (IteratorFilters::LocallyOwnedCell(),
                   dof_handler->begin_active()),
       CellFilter (IteratorFilters::LocallyOwnedCell(),
                   dof_handler->end()),
       std_cxx11::bind (&NFieldsProblem<dim, spacedim, n_components>::
                        local_assemble_system,
                        this,
                        std_cxx11::_1,
                        std_cxx11::_2,
                        std_cxx11::_3),
       std_cxx11::bind (&NFieldsProblem<dim, spacedim, n_components>::
                        copy_local_to_global_system,
                        this,
                        std_cxx11::_1),
       Assembly::Scratch::
       NFields<dim,spacedim> (system_data,
                              *fe,
                              quadrature_formula,
                              mapping,
                              (update_values    |
                               update_quadrature_points  |
                               update_JxW_values |
                               (rebuild_matrix == true
                                ?
                                update_gradients
                                :
                                UpdateFlags(0)))),
       Assembly::CopyData::
       NFieldsSystem<dim,spacedim> (*fe));

  if (rebuild_matrix == true)
    matrix.compress(VectorOperation::add);
  rhs.compress(VectorOperation::add);

  rebuild_matrix = false;

  pcout << std::endl;
  computing_timer.exit_section();
}

/* ------------------------ SOLVE ------------------------ */

template <int dim, int spacedim, int n_components>
void NFieldsProblem<dim, spacedim, n_components>::solve ()
{

  pcout << "   Solving system... " << std::flush;

  TrilinosWrappers::MPI::BlockVector
  distributed_solution (rhs);
  distributed_solution = solution;

  const unsigned int
  start = (distributed_solution.block(0).size() +
           distributed_solution.block(1).local_range().first);
  const unsigned int
  end   = (distributed_solution.block(0).size() +
           distributed_solution.block(1).local_range().second);
  for (unsigned int i=start; i<end; ++i)
    if (constraints.is_constrained (i))
      distributed_solution(i) = 0;

  unsigned int n_iterations = 0;
  const double solver_tolerance = 1e-8;

  energy.compute_system_operators(*dof_handler,
                                  matrix, preconditioner_matrix,
                                  system_op, preconditioner_op);

  PrimitiveVectorMemory<TrilinosWrappers::MPI::BlockVector> mem;
  SolverControl solver_control (30, solver_tolerance);
  SolverControl solver_control_refined (matrix.m(), solver_tolerance);

  SolverFGMRES<TrilinosWrappers::MPI::BlockVector>
  solver(solver_control, mem,
         SolverFGMRES<TrilinosWrappers::MPI::BlockVector>::
         AdditionalData(30, true));

  SolverFGMRES<TrilinosWrappers::MPI::BlockVector>
  solver_refined(solver_control_refined, mem,
                 SolverFGMRES<TrilinosWrappers::MPI::BlockVector>::
                 AdditionalData(50, true));

  auto S_inv         = inverse_operator(system_op, solver, preconditioner_op);
  auto S_inv_refined = inverse_operator(system_op, solver_refined, preconditioner_op);
  try
    {
      S_inv.vmult(distributed_solution, rhs);
      n_iterations = solver_control.last_step();
    }
  catch ( SolverControl::NoConvergence )
    {
      S_inv_refined.vmult(distributed_solution, rhs);
      n_iterations = (solver_control.last_step() +
                      solver_control_refined.last_step());
    }


  constraints.distribute (distributed_solution);
  solution = distributed_solution;

  pcout << std::endl;
  pcout << " iterations:                           " <<  n_iterations
        << std::endl;
  pcout << std::endl;

}

/* ------------------------ OUTPUTS ------------------------ */

template <int dim, int spacedim, int n_components>
void NFieldsProblem<dim, spacedim, n_components>::output_results ()
{
  computing_timer.enter_section ("Postprocessing");

  std::stringstream suffix;
  unsigned int cycle = 0;
  suffix << "." << cycle;
  data_out.prepare_data_output( *dof_handler,
                                suffix.str());
  data_out.add_data_vector (solution, energy.get_component_names());
  data_out.write_data_and_clear();

  computing_timer.exit_section ();
}

/* ------------------------ MESH AND GRID ------------------------ */

template <int dim, int spacedim, int n_components>
void NFieldsProblem<dim, spacedim, n_components>::refine_mesh ()
{
  triangulation->refine_global (1);
}


template <int dim, int spacedim, int n_components>
void NFieldsProblem<dim, spacedim, n_components>::make_grid_fe()
{
  triangulation = SP(pgg.distributed(MPI_COMM_WORLD));
  dof_handler = SP(new DoFHandler<dim,spacedim>(*triangulation));
  fe=SP(energy());
  triangulation->refine_global (initial_global_refinement);
}

/* ------------------------ ERRORS ------------------------ */

template <int dim, int spacedim, int n_components>
void NFieldsProblem<dim, spacedim, n_components>::process_solution ()
{
  eh.error_from_exact(*dof_handler, solution, exact_solution);
  eh.output_table(pcout);
}

/* ------------------------ RUN ------------------------ */

template <int dim, int spacedim, int n_components>
void NFieldsProblem<dim, spacedim, n_components>::run ()
{

  for (unsigned int cycle=0; cycle<n_cycles; ++cycle)
    {
      if (cycle == 0)
        {
          make_grid_fe ();
        }
      else
        refine_mesh ();

      setup_dofs ();
      assemble_system ();
      build_preconditioner ();
      solve ();
      output_results ();
      process_solution ();
    }

  // std::ofstream f("errors.txt");
  computing_timer.print_summary();
  timer_outfile.close();
  // f.close();
}


// template class NFieldsProblem<1>;

template class NFieldsProblem<1,1,1>;
template class NFieldsProblem<1,1,2>;
template class NFieldsProblem<1,1,3>;
template class NFieldsProblem<1,1,4>;

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
