#include "pidomus.h"
#include "pidomus_macros.h"

using namespace dealii;
using namespace deal2lkit;

// This file contains the implementation of the
// declare_parameters() and parse_parameters_call_back().

template <int dim, int spacedim, typename LAC>
void
piDoMUS<dim, spacedim, LAC>::
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
                  "1",
                  Patterns::Integer (0));

  add_parameter(  prm,
                  &max_time_iterations,
                  "Maximum number of time steps",
                  "10000",
                  Patterns::Integer (0));

  add_parameter(  prm,
                  &jacobian_solver_tolerance,
                  "Jacobian solver tolerance",
                  "1e-8",
                  Patterns::Double (0.0));

  add_parameter(  prm,
                  &use_direct_solver,
                  "Use direct solver if available",
                  "true",
                  Patterns::Bool());

  add_parameter(  prm,
                  &adaptive_refinement,
                  "Adaptive refinement",
                  "true",
                  Patterns::Bool());

  add_parameter(  prm,
                  &verbose,
                  "Print some useful informations about processes",
                  "true",
                  Patterns::Bool());

  add_parameter(  prm,
                  &overwrite_iter,
                  "Overwrite Newton's iterations",
                  "true",
                  Patterns::Bool());

  add_parameter(  prm,
                  &time_stepper,
                  "Time stepper",
                  "euler",
                  Patterns::Selection("ida|euler|imex")); //imex

  add_parameter(  prm,
                  &use_space_adaptivity,
                  "Refine mesh during transient",
                  "false",
                  Patterns::Bool());

  add_parameter(  prm,
                  &kelly_threshold,
                  "Threshold for solver's restart",
                  "1e-2",
                  Patterns::Double(0.0));

  add_parameter(  prm,
                  &max_iterations,
                  "Max iterations",
                  "50",
                  Patterns::Integer (0),
                  "Maximum number of iterations for solving the Newtons's system.\n"
                  "If this variables is 0, then the size of the matrix is used.");

  add_parameter(  prm,
                  &max_iterations_finer,
                  "Max iterations finer prec.",
                  "0",
                  Patterns::Integer (0),
                  "Maximum number of iterations for solving the Newtons's system \n"
                  "using the finer preconditioner.\n"
                  "If this variables is 0, then the size of the matrix is used.");

  add_parameter(  prm,
                  &enable_finer_preconditioner,
                  "Enable finer preconditioner",
                  "false",
                  Patterns::Bool());

  add_parameter(  prm,
                  &max_tmp_vector,
                  "Max tmp vectors",
                  "30",
                  Patterns::Integer (1),
                  "Maximum number of temporary vectors used by FGMRES for the \n"
                  "solution of the linear system using the coarse preconditioner.");

  add_parameter(  prm,
                  &max_tmp_vector_finer,
                  "Max tmp vectors for finer system",
                  "50",
                  Patterns::Integer (1),
                  "Maximum number of temporary vectors used by FGMRES for the \n"
                  "solution of the linear system using the finer preconditioner.");

#ifdef DEAL_II_WITH_ARPACK
  add_parameter(  prm,
                  &n_eigenvalues,
                  "Number of eigenvalues to compute",
                  "10",
                  Patterns::Integer (1));

  add_parameter(  prm,
                  &n_arnoldi_vectors,
                  "Number of used Arnoldi vectors",
                  "0",
                  Patterns::Integer (0),
                  "If 0, the number of vectors used will be\n"
                  "2*number_of_eigenvalues+2");

  add_parameter(  prm,
                  &which_eigenvalues,
                  "Which eigenvalues",
                  "smallest_real_part",
                  Patterns::Selection("algebraically_largest"
                                      "|algebraically_smallest"
                                      "|largest_magnitude"
                                      "|smallest_magnitude"
                                      "|largest_real_part"
                                      "|smallest_real_part"
                                      "|largest_imaginary_part"
                                      "|smallest_imaginary_part"
                                      "|both_ends"));
#endif

  add_parameter (prm,
                 &resume_computation,
                 "Resume computation from snapshot",
                 "false",
                 Patterns::Bool());

  add_parameter(prm,
                &snap_prefix,
                "Snapshot prefix",
                "",
                Patterns::Anything());

  add_parameter (prm,
                 &save_snapshot,
                 "Save snapshots during simulation",
                 "true",
                 Patterns::Bool());

}


template <int dim, int spacedim, typename LAC>
void
piDoMUS<dim, spacedim, LAC>::parse_parameters_call_back()
{
  use_direct_solver &= (typeid(typename LAC::BlockMatrix) == typeid(dealii::BlockSparseMatrix<double>));
#ifdef DEAL_II_WITH_ARPACK
  if (n_arnoldi_vectors == 0)
    n_arnoldi_vectors = 2*n_eigenvalues+2;
#endif
}


#define INSTANTIATE(dim,spacedim,LAC) \
  template class piDoMUS<dim,spacedim,LAC>;


PIDOMUS_INSTANTIATE(INSTANTIATE)


