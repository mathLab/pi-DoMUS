#include "pidomus.h"
#include "pidomus_macros.h"

#include <deal.II/numerics/solution_transfer.h>

#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_refinement.h>

using namespace dealii;
using namespace deal2lkit;


// This file contains the implementation of the functions
// required to refine the mesh and transfer the solutions
// to the newly created mesh.

template <int dim, int spacedim, typename LAC>
void piDoMUS<dim, spacedim, LAC>::refine_mesh ()
{
  auto _timer = computing_timer.scoped_timer ("Mesh refinement");

  signals.begin_refine_mesh();

  if (adaptive_refinement)
    {
      Vector<float> estimated_error_per_cell (triangulation->n_active_cells());

      interface.estimate_error_per_cell(estimated_error_per_cell);

      pgr.mark_cells(estimated_error_per_cell, *triangulation);
    }

  refine_and_transfer_solutions(solution,
                                solution_dot,
                                locally_relevant_solution,
                                locally_relevant_solution_dot,
                                locally_relevant_explicit_solution,
                                adaptive_refinement);

  current_time = std::numeric_limits<double>::quiet_NaN();
  previous_time =   std::numeric_limits<double>::quiet_NaN();

  signals.end_refine_mesh();
}


template <int dim, int spacedim, typename LAC>
bool
piDoMUS<dim, spacedim, LAC>::solver_should_restart(const double t,
                                                   typename LAC::VectorType &solution,
                                                   typename LAC::VectorType &solution_dot)
{

  auto _timer = computing_timer.scoped_timer ("Solver should restart");
  signals.begin_solver_should_restart();
  if (use_space_adaptivity)
    {
      double max_kelly=0;
      auto _timer = computing_timer.scoped_timer ("Compute error estimator");
      update_functions_and_constraints(t);

      train_constraints[0]->distribute(solution);
      locally_relevant_solution = solution;
      constraints_dot.distribute(solution_dot);
      locally_relevant_solution_dot = solution_dot;

      Vector<float> estimated_error_per_cell (triangulation->n_active_cells());

      interface.estimate_error_per_cell(estimated_error_per_cell);

      max_kelly = estimated_error_per_cell.linfty_norm();
      max_kelly = Utilities::MPI::max(max_kelly, comm);

      if (max_kelly > kelly_threshold)

        {
          pcout << "  ################ restart ######### \n"
                << "max_kelly > threshold\n"
                << max_kelly  << " >  " << kelly_threshold
                << std::endl
                << "######################################\n";
          pgr.mark_cells(estimated_error_per_cell, *triangulation);

          refine_and_transfer_solutions(solution,
                                        solution_dot,
                                        locally_relevant_solution,
                                        locally_relevant_solution_dot,
                                        locally_relevant_explicit_solution,
                                        adaptive_refinement);

          update_functions_and_constraints(t);
          train_constraints[0]->distribute(solution);
          constraints_dot.distribute(solution_dot);

          signals.fix_solutions_after_refinement(solution,solution_dot);

          MPI_Barrier(comm);
          current_time = std::numeric_limits<double>::quiet_NaN();
          signals.end_solver_should_restart();
          return true;
        }
      else // if max_kelly > kelly_threshold
        {
          signals.end_solver_should_restart();
          return false;
        }

    }
  else // use space adaptivity
    {
      signals.end_solver_should_restart();
      return false;
    }
}

template <int dim, int spacedim, typename LAC>
void piDoMUS<dim, spacedim, LAC>::
refine_and_transfer_solutions(LATrilinos::VectorType &y,
                              LATrilinos::VectorType &y_dot,
                              LATrilinos::VectorType &locally_relevant_y,
                              LATrilinos::VectorType &locally_relevant_y_dot,
                              LATrilinos::VectorType &locally_relevant_y_expl,
                              bool adaptive_refinement)
{
  signals.begin_refine_and_transfer_solutions();
  locally_relevant_y = y;
  locally_relevant_y_dot = y_dot;

  parallel::distributed::SolutionTransfer<dim, LATrilinos::VectorType, DoFHandler<dim,spacedim> > sol_tr(*dof_handler);

  std::vector<const LATrilinos::VectorType *> old_sols (3);
  old_sols[0] = &locally_relevant_y;
  old_sols[1] = &locally_relevant_y_dot;
  old_sols[2] = &locally_relevant_y_expl;

  triangulation->prepare_coarsening_and_refinement();
  sol_tr.prepare_for_coarsening_and_refinement (old_sols);

  if (adaptive_refinement)
    triangulation->execute_coarsening_and_refinement ();
  else
    triangulation->refine_global (1);

  setup_dofs(false);


  LATrilinos::VectorType new_sol (y);
  LATrilinos::VectorType new_sol_dot (y);
  LATrilinos::VectorType new_sol_expl (y);

  std::vector<LATrilinos::VectorType *> new_sols (3);
  new_sols[0] = &new_sol;
  new_sols[1] = &new_sol_dot;
  new_sols[2] = &new_sol_expl;

  sol_tr.interpolate (new_sols);

  y = new_sol;
  y_dot = new_sol_dot;


  update_functions_and_constraints(current_time);
  train_constraints[0]->distribute(y);
  constraints_dot.distribute(y_dot);

  locally_relevant_y = y;
  locally_relevant_y_dot = y_dot;
  locally_relevant_y_expl = new_sol_expl;

  signals.end_refine_and_transfer_solutions();
}

template <int dim, int spacedim, typename LAC>
void piDoMUS<dim, spacedim, LAC>::
refine_and_transfer_solutions(LADealII::VectorType &y,
                              LADealII::VectorType &y_dot,
                              LADealII::VectorType &locally_relevant_y,
                              LADealII::VectorType &locally_relevant_y_dot,
                              LADealII::VectorType &y_expl,
                              bool adaptive_refinement)
{
  signals.begin_refine_and_transfer_solutions();
  SolutionTransfer<dim, LADealII::VectorType, DoFHandler<dim,spacedim> > sol_tr(*dof_handler);

  std::vector<LADealII::VectorType> old_sols (3);
  old_sols[0] = y;
  old_sols[1] = y_dot;
  old_sols[2] = y_expl;

  triangulation->prepare_coarsening_and_refinement();
  sol_tr.prepare_for_coarsening_and_refinement (old_sols);

  if (adaptive_refinement)
    triangulation->execute_coarsening_and_refinement ();
  else
    triangulation->refine_global (1);

  setup_dofs(false);

  std::vector<LADealII::VectorType> new_sols (3);

  new_sols[0].reinit(y);
  new_sols[1].reinit(y_dot);
  new_sols[2].reinit(y_expl);

  sol_tr.interpolate (old_sols, new_sols);

  y      = new_sols[0];
  y_dot  = new_sols[1];
  y_expl = new_sols[2];

  update_functions_and_constraints(previous_time);
  train_constraints[0]->distribute(y_expl);

  update_functions_and_constraints(current_time);
  train_constraints[0]->distribute(y);
  constraints_dot.distribute(y_dot);

  locally_relevant_y = y;
  locally_relevant_y_dot = y_dot;

  signals.end_refine_and_transfer_solutions();
}

#define INSTANTIATE(dim,spacedim,LAC) \
  template class piDoMUS<dim,spacedim,LAC>;


PIDOMUS_INSTANTIATE(INSTANTIATE)
