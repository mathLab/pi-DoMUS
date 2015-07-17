/**
 * Solve time-dependent non-linear n-fields problem
 * in parallel.
 *
 *
 *
 *
 *
 *
 *
 *
 *
 */

#ifndef _N_FIELDS_LINEAR_PROBLEM_
#define _N_FIELDS_LINEAR_PROBLEM_


#include <deal.II/base/timer.h>
// #include <deal.II/base/parameter_handler.h>

#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/linear_operator.h>

// #include <deal.II/lac/precondition.h>

#include "assembly.h"
#include "interface.h"
#include "parsed_grid_generator.h"
#include "parsed_finite_element.h"
#include "error_handler.h"
#include "parsed_function.h"
#include "parsed_data_out.h"
#include "parameter_acceptor.h"
#include "ode_argument.h"
#include "dae_time_integrator.h"

#include "sak_data.h"

#include "fe_values_cache.h"

#include "mpi.h"

using namespace dealii;

typedef TrilinosWrappers::MPI::BlockVector VEC;

template <int dim, int spacedim=dim, int n_components=1>
class NFieldsProblem : public ParameterAcceptor, public OdeArgument<VEC>
{

  typedef typename Assembly::CopyData::NFieldsSystem<dim,spacedim> SystemCopyData;
  typedef typename Assembly::CopyData::NFieldsPreconditioner<dim,spacedim> PreconditionerCopyData;
  typedef typename Assembly::Scratch::NFields<dim,spacedim> Scratch;

  // This is a class required to make tests
  template<int fdim, int fspacedim, int fn_components>
  friend void test(NFieldsProblem<fdim,fspacedim,fn_components> &);

public:


  NFieldsProblem (const Interface<dim,spacedim,n_components> &energy,
                  const MPI_Comm &comm=MPI_COMM_WORLD);

  virtual void declare_parameters(ParameterHandler &prm);

  void run ();


  /*********************************************************
   * Public interface from OdeArgument
   *********************************************************/
  virtual shared_ptr<VEC>
  create_new_vector() const;

  /** Returns the number of degrees of freedom. Pure virtual function. */
  virtual unsigned int n_dofs() const;

  /** This function is called at the end of each iteration step for
   * the ode solver. Once again, the conversion between pointers and
   * other forms of vectors need to be done inside the inheriting
   * class. */
  virtual void output_step(const double t,
                           const VEC &solution,
                           const VEC &solution_dot,
                           const unsigned int step_number,
                           const double h);

  /** This function will check the behaviour of the solution. If it
   * is converged or if it is becoming unstable the time integrator
   * will be stopped. If the convergence is not achived the
   * calculation will be continued. If necessary, it can also reset
   * the time stepper. */
  virtual bool solver_should_restart(const double t,
                                     const VEC &solution,
                                     const VEC &solution_dot,
                                     const unsigned int step_number,
                                     const double h);

  /** For dae problems, we need a
   residual function. */
  virtual int residual(const double t,
                       const VEC &src_yy,
                       const VEC &src_yp,
                       VEC &dst) const;

  /** Setup Jacobian system and preconditioner. */
  virtual int setup_jacobian(const double t,
                             const VEC &src_yy,
                             const VEC &src_yp,
                             const VEC &residual,
                             const double alpha);


  /** Inverse of the Jacobian vector product. */
  virtual int solve_jacobian_system(const double t,
                                    const VEC &y,
                                    const VEC &y_dot,
                                    const VEC &residual,
                                    const double alpha,
                                    const VEC &src,
                                    VEC &dst) const;



  /** And an identification of the
   differential components. This
   has to be 1 if the
   corresponding variable is a
   differential component, zero
   otherwise.  */
  virtual VEC &differential_components() const;

private:
  void make_grid_fe();
  void setup_dofs (const bool &first_run=true);

  void reinit_jacobian_matrix (const std::vector<IndexSet> &partitioning,
                               const std::vector<IndexSet> &relevant_partitioning);


  void assemble_jacobian_matrix (const double t,
                                 const VEC &y,
                                 const VEC &y_dot,
                                 const double alpha);


  void reinit_jacobian_preconditioner (const std::vector<IndexSet> &partitioning,
                                       const std::vector<IndexSet> &relevant_partitioning);


  void assemble_jacobian_preconditioner (const double t,
                                         const VEC &y,
                                         const VEC &y_dot,
                                         const double alpha);
  void refine_mesh ();
  void process_solution ();

  void set_constrained_dofs_to_zero(VEC &v) const;

  const MPI_Comm &comm;
  const Interface<dim,spacedim,n_components>    &energy;

  unsigned int n_cycles;
  unsigned int current_cycle;
  unsigned int initial_global_refinement;
  unsigned int max_time_iterations;
  double fixed_alpha;

  std::string timer_file_name;

  ConditionalOStream        pcout;
  std::ofstream         timer_outfile;
  ConditionalOStream        tcout;

  shared_ptr<Mapping<dim,spacedim> >             mapping;

  shared_ptr<parallel::distributed::Triangulation<dim,spacedim> > triangulation;
  shared_ptr<FiniteElement<dim,spacedim> >       fe;
  shared_ptr<DoFHandler<dim,spacedim> >          dof_handler;

  ConstraintMatrix                          constraints;

  TrilinosWrappers::BlockSparseMatrix       jacobian_matrix;
  TrilinosWrappers::BlockSparseMatrix       jacobian_preconditioner_matrix;

  LinearOperator<TrilinosWrappers::MPI::BlockVector> jacobian_preconditioner_op;
  LinearOperator<TrilinosWrappers::MPI::BlockVector> jacobian_op;

  TrilinosWrappers::MPI::BlockVector        solution;
  TrilinosWrappers::MPI::BlockVector        solution_dot;

  mutable TrilinosWrappers::MPI::BlockVector        distributed_solution;
  mutable TrilinosWrappers::MPI::BlockVector        distributed_solution_dot;


  mutable TimerOutput                               computing_timer;


  ErrorHandler<1>       eh;
  ParsedGridGenerator<dim,spacedim>   pgg;

  ParsedFunction<spacedim, n_components>        exact_solution;

  ParsedFunction<spacedim, n_components>        initial_solution;
  ParsedFunction<spacedim, n_components>        initial_solution_dot;

  ParsedDataOut<dim, spacedim>                  data_out;

  DAETimeIntegrator<VEC>  dae;

  IndexSet global_partioning;
  std::vector<IndexSet> partitioning;
  std::vector<IndexSet> relevant_partitioning;

  bool adaptive_refinement;
};

#endif
