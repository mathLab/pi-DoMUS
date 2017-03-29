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

#ifndef __pi_DoMUS_h_
#define __pi_DoMUS_h_


#include <deal.II/base/timer.h>
// #include <deal.II/base/parameter_handler.h>

#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/linear_operator.h>



#include <mpi.h>


// #include <deal.II/lac/precondition.h>


#include "base_interface.h"
#include "simulator_access.h"
#include "pidomus_signals.h"
#include "pidomus_lambdas.h"


#include <deal2lkit/parsed_grid_generator.h>
#include <deal2lkit/parsed_finite_element.h>
#include <deal2lkit/parsed_grid_refinement.h>
#include <deal2lkit/error_handler.h>
#include <deal2lkit/parsed_function.h>
#include <deal2lkit/parameter_acceptor.h>
#include <deal2lkit/ida_interface.h>
#include <deal2lkit/imex_stepper.h>
#include <deal2lkit/parsed_zero_average_constraints.h>
#include <deal2lkit/parsed_mapped_functions.h>
#include <deal2lkit/parsed_dirichlet_bcs.h>

#include <deal2lkit/any_data.h>
#include <deal2lkit/fe_values_cache.h>

#include "lac/lac_type.h"
#include "lac/lac_initializer.h"

using namespace dealii;
using namespace deal2lkit;
using namespace pidomus;

template <int dim, int spacedim = dim, typename LAC = LATrilinos>
class piDoMUS : public ParameterAcceptor
{


  // This is a class required to make tests
  template<int fdim, int fspacedim, typename fn_LAC>
  friend void test(piDoMUS<fdim, fspacedim, fn_LAC> &);

public:

#ifdef DEAL_II_WITH_MPI
  piDoMUS (const std::string &name,
           const BaseInterface<dim, spacedim, LAC> &energy,
           const MPI_Comm comm = MPI_COMM_WORLD);
#else
  piDoMUS (const std::string &name,
           const BaseInterface<dim, spacedim, LAC> &energy);
#endif

  virtual void declare_parameters(ParameterHandler &prm);
  virtual void parse_parameters_call_back();

  void run ();

  mutable shared_ptr<SolverControl> solver_control;
  mutable shared_ptr<SolverControl> solver_control_finer;

  /*********************************************************
   * Public interface from SundialsInterface
   *********************************************************/
  virtual shared_ptr<typename LAC::VectorType>
  create_new_vector() const;

  /** Returns the number of degrees of freedom. Pure virtual function. */
  virtual unsigned int n_dofs() const;

  /** This function is called at the end of each iteration step for
   * the ode solver. Once again, the conversion between pointers and
   * other forms of vectors need to be done inside the inheriting
   * class. */
  virtual void output_step(const double t,
                           const typename LAC::VectorType &solution,
                           const typename LAC::VectorType &solution_dot,
                           const unsigned int step_number);

  /** Check the behaviour of the solution. If it
   * is converged or if it is becoming unstable the time integrator
   * will be stopped. If the convergence is not achived the
   * calculation will be continued. If necessary, it can also reset
   * the time stepper. */
  virtual bool solver_should_restart(const double t,
                                     typename LAC::VectorType &solution,
                                     typename LAC::VectorType &solution_dot);

  /** For dae problems, we need a
   residual function. */
  virtual int residual(const double t,
                       const typename LAC::VectorType &src_yy,
                       const typename LAC::VectorType &src_yp,
                       typename LAC::VectorType &dst);

  /** Setup Jacobian system and preconditioner. */
  virtual int setup_jacobian(const double t,
                             const typename LAC::VectorType &src_yy,
                             const typename LAC::VectorType &src_yp,
                             const double alpha);


  /** Inverse of the Jacobian vector product. */
  virtual int solve_jacobian_system(const typename LAC::VectorType &src,
                                    typename LAC::VectorType &dst) const;

  /**
   * compute jacobian vmult. It is used by the KINSOL solver
   */
  virtual int jacobian_vmult(const typename LAC::VectorType &v,
                             typename LAC::VectorType &dst) const;

  /** And an identification of the
   differential components. This
   has to be 1 if the
   corresponding variable is a
   differential component, zero
   otherwise.  */
  virtual typename LAC::VectorType &differential_components() const;

  /**
   * Get back the solution.
   */
  typename LAC::VectorType &get_solution();

  /**
   * save the diagonal of the lumped mass matrix in @p diag, which is
   * used as scaling vector in KINSOL
   */
  virtual void get_lumped_mass_matrix(typename LAC::VectorType &diag) const;


#ifdef DEAL_II_WITH_ARPACK
  /**
    * solve eigenvalue problem
    */
  void solve_eigenproblem();

  /**
   * get eigenvalues
   *
   */
  const std::vector<std::complex<double> > &get_eigenvalues();

  /**
   * get eigenvectors
   *
   */
  const std::vector<typename LAC::VectorType> &get_eigenvectors();


private:
  void do_solve_eigenproblem(const LADealII::BlockMatrix &mat,
                             const LADealII::BlockMatrix &mass,
                             const LinearOperator<LADealII::VectorType> &jac,
                             const LinearOperator<LADealII::VectorType> &jac_prec,
                             const LinearOperator<LADealII::VectorType> &jac_prec_fin,
                             std::vector<LADealII::VectorType> &eigenvectors,
                             std::vector<std::complex<double> > &eigenvalues);


  void do_solve_eigenproblem(const LATrilinos::BlockMatrix &mat,
                             const LATrilinos::BlockMatrix &mass,
                             const LinearOperator<LATrilinos::VectorType> &jac,
                             const LinearOperator<LATrilinos::VectorType> &jac_prec,
                             const LinearOperator<LATrilinos::VectorType> &jac_prec_fin,
                             std::vector<LATrilinos::VectorType> &eigenvectors,
                             std::vector<std::complex<double> > &eigenvalues);

  /**
   * number of eigenvalues to compute
   */
  unsigned int n_eigenvalues;

  /**
   * number of Arnoldi vectors used by Arpack
   */
  unsigned int n_arnoldi_vectors;

  /**
   * available options:
   *
   * algebraically_largest
   * algebraically_smallest
   * largest_magnitude
   * smallest_magnitude
   * largest_real_part
   * smallest_real_part
   * largest_imaginary_part
   * smallest_imaginary_part
   * both_ends
   */
  std::string which_eigenvalues;

  /**
   * eigenvectors
   */
  std::vector<typename LAC::VectorType> eigv;

  /**
   * eigenvalues
   */
  std::vector<std::complex<double> > eigval;


#endif

private:


  /**
   * Set time to @p t for forcing terms and boundary conditions
   */
  void update_functions_and_constraints(const double &t);




  /**
   * Apply Dirichlet boundary conditions.
   * It takes as argument a DoF handler @p dof_handler, a
   * ParsedDirichletBCs and a constraint matrix @p constraints.
   *
   */
  void apply_dirichlet_bcs (const DoFHandler<dim,spacedim> &dof_handler,
                            const ParsedDirichletBCs<dim,spacedim> &bc,
                            ConstraintMatrix &constraints) const;

  /**
   * Apply Neumann boundary conditions.
   *
   */
  void apply_neumann_bcs (const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                          FEValuesCache<dim,spacedim> &scratch,
                          std::vector<double> &local_residual) const;


  /**
   * Applies CONSERVATIVE forcing terms, which can be defined by
   * expressions in the parameter file.
   *
   * If the problem involves NON-conservative loads, they must be
   * included in the residual formulation.
   *
   */
  void apply_forcing_terms (const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                            FEValuesCache<dim,spacedim> &scratch,
                            std::vector<double> &local_residual) const;



  void make_grid_fe();
  void setup_dofs (const bool &first_run = true);

  void assemble_matrices (const double t,
                          const typename LAC::VectorType &y,
                          const typename LAC::VectorType &y_dot,
                          const double alpha);
  void refine_mesh ();

  void refine_and_transfer_solutions (LADealII::VectorType &y,
                                      LADealII::VectorType &y_dot,
                                      LADealII::VectorType &locally_relevant_y,
                                      LADealII::VectorType &locally_relevant_y_dot,
                                      LADealII::VectorType &locally_relevant_y_expl,
                                      bool adaptive_refinement);


  void refine_and_transfer_solutions (LATrilinos::VectorType &y,
                                      LATrilinos::VectorType &y_dot,
                                      LATrilinos::VectorType &locally_relevant_y,
                                      LATrilinos::VectorType &locally_relevant_y_dot,
                                      LATrilinos::VectorType &locally_relevant_y_expl,
                                      bool adaptive_refinement);


  void set_constrained_dofs_to_zero(typename LAC::VectorType &v) const;

#ifdef DEAL_II_WITH_MPI
  const MPI_Comm comm;
#endif
  const BaseInterface<dim, spacedim, LAC>    &interface;

  unsigned int n_cycles;
  unsigned int current_cycle;
  unsigned int initial_global_refinement;

  ConditionalOStream        pcout;

  ParsedGridGenerator<dim, spacedim>   pgg;
  ParsedGridRefinement  pgr;

  shared_ptr<parallel::distributed::Triangulation<dim, spacedim> > triangulation;
  shared_ptr<FiniteElement<dim, spacedim> >       fe;
  shared_ptr<DoFHandler<dim, spacedim> >          dof_handler;

  std::vector<shared_ptr<ConstraintMatrix> >      train_constraints;
  ConstraintMatrix                                constraints_dot;

  LinearOperator<typename LAC::VectorType> jacobian_preconditioner_op;
  LinearOperator<typename LAC::VectorType> jacobian_preconditioner_op_finer;
  LinearOperator<typename LAC::VectorType> jacobian_op;


  //// current time

  /**
   * solution at current time step
   */
  typename LAC::VectorType        solution;

  /**
   * solution_dot at current time step
   */
  typename LAC::VectorType        solution_dot;

  /**
   * distributed solution at current time step
   */
  mutable typename LAC::VectorType        locally_relevant_solution;

  /**
   * distributed solution_dot at current time step
   */
  mutable typename LAC::VectorType        locally_relevant_solution_dot;

  /**
   * current time
   */
  double current_time;

  /**
   * current alpha
   */
  double current_alpha;

  /**
   * current dt, i.e. the dt used to go from t to to+dt
   */
  double current_dt;



  //// previous time step

  /**
   * distributed solution at previous time step
   */
  mutable typename LAC::VectorType        locally_relevant_explicit_solution;

  /**
   * previous time
   */
  double previous_time;

  /**
   * previous time step
   */
  double previous_dt;



  //// second-to-last time step


  /**
   * distributed solution at second-to-last time step
   */
  mutable typename LAC::VectorType        locally_relevant_previous_explicit_solution;


  /**
   * second-to-last time
   */
  double second_to_last_time;

  /**
   * second-to-last time step
   */
  double second_to_last_dt;

  /**
   * Teucos timer file
   */
  mutable TimeMonitor       computing_timer;

  const unsigned int n_matrices;
  std::vector<shared_ptr<typename LAC::BlockSparsityPattern> > matrix_sparsities;
  std::vector<shared_ptr<typename LAC::BlockMatrix> >  matrices;

  ErrorHandler<1>       eh;

  ParsedFunction<spacedim>        exact_solution;

  ParsedFunction<spacedim>        initial_solution;
  ParsedFunction<spacedim>        initial_solution_dot;


  ParsedMappedFunctions<spacedim>  forcing_terms; // on the volume
  ParsedMappedFunctions<spacedim>  neumann_bcs;
  ParsedDirichletBCs<dim,spacedim> dirichlet_bcs;
  ParsedDirichletBCs<dim,spacedim> dirichlet_bcs_dot;

  ParsedZeroAverageConstraints<dim,spacedim> zero_average;



  IDAInterface<typename LAC::VectorType>  ida;
  IMEXStepper<typename LAC::VectorType> imex;


  std::vector<types::global_dof_index> dofs_per_block;
  IndexSet global_partitioning;
  std::vector<IndexSet> partitioning;
  std::vector<IndexSet> relevant_partitioning;

  bool adaptive_refinement;
  const bool we_are_parallel;
  bool use_direct_solver;

  /**
   * Solver tolerance for the equation:
   * \f[ \mathcal{J}(F)[u] = R[u] \f]
   */
  double jacobian_solver_tolerance;

  /**
   * Print all avaible information about processes.
   */
  bool verbose;

  /**
   * Overwrite newton's iterations: every time step shows only the last value.
   */
  bool overwrite_iter;

  /**
   * time stepper to be used
   */
  std::string time_stepper;


  /**
   * refine mesh during transients
   */
  bool use_space_adaptivity;

  /**
   * threshold for refine mesh during transients
   */
  double kelly_threshold;

  /**
   * Maximum number of iterations for solving the Newtons's system.
   * If this variables is 0, then the size of the matrix is used.
   */
  unsigned int max_iterations;

  /**
   * Maximum number of temporary vectors used by FGMRES for the solution of the
   * linear system using the coarse preconditioner.
   */
  unsigned int max_tmp_vector;

  /**
   * Maximum number of iterations for solving the Newtons's system
   * using the finer preconditioner.  If this variables is 0, then the
   * size of the matrix is used.
   */
  unsigned int max_iterations_finer;

  /**
   * Maximum number of temporary vectors used by FGMRES for the solution of the
   * linear system using the finer preconditioner.
   */
  unsigned int max_tmp_vector_finer;

  /**
   * use a coarse preconditioner and then a finer preconditioner
   */
  bool enable_finer_preconditioner;


  /**
   * Syncronize to time t. This function update functions and
   * constraints to time t and udpate the explicit solutions and
   * previous explicit solutions vectors, as well as the related
   * variables (i.e., previous_time, previous_dt etc.)
   */
  void syncronize(const double &t,
                  const typename LAC::VectorType &solution,
                  const typename LAC::VectorType &solution_dot);


  /**
   * Return the norm of vector @p v. Internally it calls the
   * virtual function BaseInterface::vector_norm().
   */
  double vector_norm(const typename LAC::VectorType &v) const;

  /**
   * Struct containing the signals
   */
  Signals<dim,spacedim,LAC>    signals;

  /**
   * SimulatorAccess accesses to all internal variables and returns a
   * const reference to them through functions named get_variable()
   */
  friend class SimulatorAccess<dim,spacedim,LAC>;

public:

  friend class Lambdas<dim,spacedim,LAC>;

  Lambdas<dim,spacedim,LAC> lambdas;

};



#endif
