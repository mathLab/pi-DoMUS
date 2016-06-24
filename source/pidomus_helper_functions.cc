#include "pidomus.h"
#include "pidomus_macros.h"

using namespace dealii;
using namespace deal2lkit;

// This file contains the implementation of helper functions
// or functions required by specific interface with external
// libraries (e.g. Sundials).

template <int dim, int spacedim, typename LAC>
shared_ptr<typename LAC::VectorType>
piDoMUS<dim, spacedim, LAC>::create_new_vector() const
{
  shared_ptr<typename LAC::VectorType> ret = SP(new typename LAC::VectorType(solution));
  //   *ret *= 0;
  return ret;
}


template <int dim, int spacedim, typename LAC>
unsigned int
piDoMUS<dim, spacedim, LAC>::n_dofs() const
{
  return dof_handler->n_dofs();
}


template <int dim, int spacedim, typename LAC>
typename LAC::VectorType &
piDoMUS<dim, spacedim, LAC>::
get_solution()
{
  return solution;
}


template <int dim, int spacedim, typename LAC>
void
piDoMUS<dim,spacedim,LAC>::
syncronize(const double &t,
           const typename LAC::VectorType &solution,
           const typename LAC::VectorType &solution_dot)
{
  auto _timer = computing_timer.scoped_timer ("Syncronize");
  if (std::isnan(current_time))
    {
      // we are at the very first time step
      // solution = initial solution
      // explicit solution will be zero
      // previous explicit solution will be zero
      current_time = t;
      if (std::isnan(previous_time)) //very first step
        previous_time = t;
      update_functions_and_constraints(t);
      typename LAC::VectorType tmp(solution);
      typename LAC::VectorType tmp_dot(solution_dot);
      constraints.distribute(tmp);
      constraints_dot.distribute(tmp_dot);

      locally_relevant_solution = tmp;
      locally_relevant_solution_dot = tmp_dot;
      locally_relevant_explicit_solution = tmp;
    }
  else if (current_time < t) // next temporal step
    {
      second_to_last_dt = previous_dt;
      previous_dt       = current_dt;
      current_dt        = t - current_time;

      second_to_last_time = previous_time;
      previous_time = current_time;
      current_time = t;

      locally_relevant_previous_explicit_solution     = locally_relevant_explicit_solution;
      locally_relevant_explicit_solution     = solution;

      update_functions_and_constraints(t);
      typename LAC::VectorType tmp(solution);
      typename LAC::VectorType tmp_dot(solution_dot);
      constraints.distribute(tmp);
      constraints_dot.distribute(tmp_dot);

      locally_relevant_solution = tmp;
      locally_relevant_solution_dot = tmp_dot;
    }
  else if (current_time == t)
    {
      // we are calling this function in different part of the code
      // within the same time step (e.g. during the non-linear solver)
      // so no need to call udpdate_functions_and_constraints()

      typename LAC::VectorType tmp(solution);
      typename LAC::VectorType tmp_dot(solution_dot);
      constraints.distribute(tmp);
      constraints_dot.distribute(tmp_dot);
      locally_relevant_solution = tmp;
      locally_relevant_solution_dot = tmp_dot;
    }
  else
    {
      // we get here when t<current_time. This may happen when we are
      // trying to estimate the time step and we need to reduce the
      // first guess. IDA enters here often times
      current_time = t;
      update_functions_and_constraints(t);

      typename LAC::VectorType tmp(solution);
      typename LAC::VectorType tmp_dot(solution_dot);
      constraints.distribute(tmp);
      constraints_dot.distribute(tmp_dot);

      locally_relevant_solution = tmp;
      locally_relevant_solution_dot = tmp_dot;
    }

}


template <int dim, int spacedim, typename LAC>
void piDoMUS<dim, spacedim, LAC>::update_functions_and_constraints (const double &t)
{
  auto _timer = computing_timer.scoped_timer ("Update functions and constraints");
  if (!std::isnan(t))
    {
      dirichlet_bcs.set_time(t);
      dirichlet_bcs_dot.set_time(t);
      forcing_terms.set_time(t);
      neumann_bcs.set_time(t);
    }
  // clear previously stored constraints
  constraints.clear();
  constraints_dot.clear();

  // compute hanging nodes
  DoFTools::make_hanging_node_constraints (*dof_handler,
                                           constraints);

  DoFTools::make_hanging_node_constraints (*dof_handler,
                                           constraints_dot);

  // compute boundary values
  apply_dirichlet_bcs(*dof_handler, dirichlet_bcs, constraints);
  apply_dirichlet_bcs(*dof_handler, dirichlet_bcs_dot, constraints_dot);


  // apply zero average constraints
  zero_average.apply_zero_average_constraints(*dof_handler, constraints);

  // add user-supplied bcs
  signals.update_constraint_matrices(constraints,constraints_dot);

  // close the constraints
  constraints.close ();
  constraints_dot.close ();
}

template <int dim, int spacedim, typename LAC>
int
piDoMUS<dim,spacedim,LAC>::jacobian_vmult(const typename LAC::VectorType &src, typename LAC::VectorType &dst) const
{
  auto _timer = computing_timer.scoped_timer ("Jacobian vmult");

  if (we_are_parallel == false &&
      use_direct_solver == true)
    matrices[0]->vmult(dst, src);
  else
    jacobian_op.vmult(dst, src);

  return 0;
}



template <int dim, int spacedim, typename LAC>
typename LAC::VectorType &
piDoMUS<dim, spacedim, LAC>::differential_components() const
{
  static typename LAC::VectorType diff_comps;
  diff_comps.reinit(solution);
  std::vector<unsigned int> block_diff = interface.get_differential_blocks();
  for (unsigned int i = 0; i < block_diff.size(); ++i)
    diff_comps.block(i) = block_diff[i];
  signals.fix_differential_components(diff_comps);
  set_constrained_dofs_to_zero(diff_comps);
  return diff_comps;
}



template <int dim, int spacedim, typename LAC>
void
piDoMUS<dim, spacedim, LAC>::set_constrained_dofs_to_zero(typename LAC::VectorType &v) const
{
  if (global_partitioning.n_elements() > 0)
    {
      auto k = global_partitioning.nth_index_in_set(0);
      if (constraints.is_constrained(k))
        v[k] = 0;
      else
        v[k] = v[k];
      for (unsigned int i = 1; i < global_partitioning.n_elements(); ++i)
        {
          auto j = global_partitioning.nth_index_in_set(i);
          if (constraints.is_constrained(j))
            v[j] = 0;
        }
      v.compress(VectorOperation::insert);
    }
}



template <int dim, int spacedim, typename LAC>
void
piDoMUS<dim, spacedim, LAC>::output_step(const double  t,
                                         const typename LAC::VectorType &solution,
                                         const typename LAC::VectorType &solution_dot,
                                         const unsigned int step_number,
                                         const double // h
                                        )
{
  auto _timer = computing_timer.scoped_timer ("Postprocessing");

  syncronize(t,solution,solution_dot);

  interface.output_solution(current_cycle,
                            step_number);
}



#define INSTANTIATE(dim,spacedim,LAC) \
  template class piDoMUS<dim,spacedim,LAC>;


PIDOMUS_INSTANTIATE(INSTANTIATE)
