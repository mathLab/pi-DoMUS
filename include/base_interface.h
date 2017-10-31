#ifndef _pidomus_base_interface_h_
#define _pidomus_base_interface_h_

#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/block_linear_operator.h>
#include <deal.II/lac/packaged_operation.h>

#include <deal.II/differentiation/ad/sacado_product_types.h>

#include <deal.II/numerics/error_estimator.h>

#include <deal2lkit/parsed_finite_element.h>
#include <deal2lkit/parsed_data_out.h>


#include "copy_data.h"
#include "lac/lac_type.h"
#include "simulator_access.h"

//forward declaration
template <int dim, int spacedim, typename LAC> struct Signals;


using namespace pidomus;
/**
 * Base Interface
 *
 * *Goal*: provide a derivable interface to solve a particular PDE
 * System (time dependent, first-order, non linear).
 *
 * *Interface*: provide some default implementations of the standard
 * requirements for Systems of PDEs (finite element definition,
 * boundary and initial conditions, etc.) and provides an unified
 * interface to fill the local (cell-wise) contributions of all
 * matrices and residuals required for the definition of the problem.
 *
 * The underlying pi-DoMUS driver uses TBB and MPI to assemble
 * matrices and vectors, and calls the virtual methods
 * assemble_local_matrices() and assemble_local_residuals() of this
 * class. If the user wishes to maximise efficiency, then these
 * methods can be directly overloaded.
 *
 * Their default implementation exploit the Sacado package of the
 * Trilinos library to automatically compute the local matrices taking
 * the derivative of residuals and the hessian of the energies
 * supplied by the assemble_energies_and_residuals() method. This
 * method is overloaded for different types, and since these types
 * cannot be inherited by derived classes using template arguments,
 * the Curiously recurring template pattern strategy (CRTP) or F-bound
 * polymorphism
 * (https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern)
 * is used to allow the implementation in user derived classes of a
 * single templated function energies_and_residuals() that is
 * statically linked and called insied the method
 * PDESystemInterface::assemble_energies_and_residuals().
 *
 * The user can directly overload the methods of this class (which
 * cannot be templated), or derive their classes from PDESystemInterface
 * instead, using CRTP, as in the following example:
 *
 * \code
 * template<int dim, int spacedim, typename LAC>
 * MyInterface : public PDESystemInterface<dim,spacedim, MyInterface<dim,spacedim,LAC>, LAC> {
 * public:
 *  template<typename Number>
 *  void energies_and_residual(...);
 * }
 * \endcode
 *
 * The class PDESystemInterface is derived from BaseInterface, and implements
 * CRTP.
 *
 * BaseInterface derives from SimulatorAccess class, which stores a reference to
 * the simulator (i.e. pi-DoMUS) where the specific pde system is solved.
 * Each variable inside the simulator can be accessed with a function
 * "get_solution()", "get_locally_relevant_solution()", "get_time()".
 */
template <int dim,int spacedim=dim, typename LAC=LATrilinos>
class BaseInterface : public ParameterAcceptor, public SimulatorAccess<dim,spacedim,LAC>
{

public:

  /**
   * virtual destructor.
   */
  virtual ~BaseInterface() {}

  /** @name Function needed in order to construct the interface */
  /** @{ */

  /**
   * Constructor. It takes the name of the subsection within the
   * parameter file, the number of components, the number of matrices,
   * the finite element used to discretize the system, the name of the
   * components and a string were the block of differential and
   * algebraic components are specified.
   */
  BaseInterface(const std::string &name="",
                const unsigned int &n_comp=0,
                const unsigned int &n_matrices=0,
                const std::string &default_fe="FE_Q(1)",
                const std::string &default_component_names="u",
                const std::string &default_differential_components="");
  /**
   * Set the dimension of coupling consistenly with the number of
   * components and matrices set in the constructor.  This function
   * must be called inside the constructor of the derived class.
   */
  void init ();
  /** @} */

  /**
   * Override this function in your derived interface in order to
   * connect to the signals, which are defined in the struct Signals.
   *
   * Example of implementation:
   * @code
   * auto &signals = this->get_signals();
   * signals.fix_initial_conditions.connect(
   *         [this](VEC &y, VEC &y_dot)
   *          {
   *            y=1;
   *            y_dot=0;
   *          });
   * @endcode
   *
   * An example of implementation is given in the poisson_problem_signals.h file.
   */
  virtual void connect_to_signals() const;

  /**
   * Solution preprocessing. This function can be used to store
   * variables, which are needed during the assembling of energies and
   * residuals, that cannot be computed there (e.g., GLOBAL
   * variables). The variables must be stored inside the AnyData of
   * the passed FEValuescache. You may want to use a WorkStream inside
   * this function.
   */
  virtual void solution_preprocessing (FEValuesCache<dim,spacedim> &scratch) const;

  /**
   * This function is called inside the output_step of pi-DoMUS and
   * defines what is stored/printed. By default it stores the solution
   * and solution_dot in file with extension parsed in the parameter
   * file. If you need to perform post-processing on the solution you
   * must override this function.
   */
  virtual void output_solution (const unsigned int &cycle,
                                const unsigned int &step_number) const;

#ifdef DEAL_II_WITH_ARPACK
  /**
    * This function is called after an eigenvalue calculation
    */
  virtual void output_eigenvectors(const std::vector<typename LAC::VectorType> &eigenfunctions,
                                   const std::vector<std::complex<double> > &eigenvalues,
                                   const unsigned int &current_cycle) const;
#endif

  /** @name Functions dedicated to assemble system and preconditioners */
  /** @{ */

  /**
   * Assemble energies and residuals. To be used when computing only residual
   * quantities, i.e., the energy here is a Sacado double, while the residual
   * is a pure double.
   */
  virtual void assemble_energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &,
                                               FEValuesCache<dim,spacedim> &,
                                               std::vector<Sdouble> &energies,
                                               std::vector<std::vector<double> > &local_residuals,
                                               bool compute_only_system_terms) const;

  /**
   * Assemble energies and residuals. To be used when computing energetical
   * quantities, i.e., the energy here is a SacadoSacado double, while the residual
   * is a Sacado double.
   */
  virtual void assemble_energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &,
                                               FEValuesCache<dim,spacedim> &,
                                               std::vector<SSdouble> &energies,
                                               std::vector<std::vector<Sdouble> > &local_residuals,
                                               bool compute_only_system_terms) const;

  /**
   * Assemble local matrices. The default implementation calls
   * BaseInterface::assemble_energies_and_residuals and creates the
   * local matrices by performing automatic differentiation on the
   * results.
   */
  virtual void assemble_local_matrices (const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                                        FEValuesCache<dim,spacedim> &scratch,
                                        CopyData &data) const;

  /**
   * Assemble the local system residual associated to the given cell.
   * This function is called to evaluate the local system residual at each
   * Newton iteration.
   */
  virtual void assemble_local_system_residual (const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                                               FEValuesCache<dim,spacedim> &scratch,
                                               CopyData &data) const;
#ifdef DEAL_II_WITH_ARPACK
  /**
   * Assemble the local mass matrix. The mass matrix is used for solving
   * the generalized eigenvalue problem. By default the mass matrix is
   * assembled. If you want to solve a different eigenvalue problem
   * you need to override this function.
   */
  virtual void assemble_local_mass_matrix(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                                          FEValuesCache<dim,spacedim> &scratch,
                                          CopyMass &data) const;
#endif

  /**
   * Compute linear operators needed by the problem: - @p system_op
   * represents the system matrix associated to the Newton's
   * iterations; - @p prec_op represents the preconditioner; - @p
   * prec_op_finer represents a finer preconditioner that can be used
   * in the case the problem does not converge using @p prec_op .
   *
   * To clarify the difference between @p prec_op and @p prec_op_finer
   * consider the case where you have a matrix A and you want to
   * provide an 'inverse' using AMG.  A possible strategy consist in
   * using the linear operator associated to AMG and a further
   * strategy is to invert A using AMG.  In detail, @p prec_op
   * should be
   * \code{.cpp}
   *   auto A_inv = linear_operator(A, AMG)
   * \endcode
   * while @p prec_op_finer
   * \code{.cpp}
   *   auto A_inv = inverse_operator(A, solver, AMG)
   * \endcode
   *
   * In the .prm file it is possible to specify the maximum
   * number of iterations allowed for the solver in the case we are
   * using @p prec_op or @p prec_op_finer.
   *
   * To enable the finer preconditioner it is sufficient to set
   * "Enable finer preconditioner" equals to true.
   */
  virtual void compute_system_operators(const std::vector<shared_ptr<typename LATrilinos::BlockMatrix> > &,
                                        LinearOperator<LATrilinos::VectorType> &system_op,
                                        LinearOperator<LATrilinos::VectorType> &prec_op,
                                        LinearOperator<LATrilinos::VectorType> &prec_op_finer) const;

  /**
   * Compute linear operators needed by the problem. When using
   * deal.II vector and matrix types, this function is empty, since a
   * direct solver is used by default.
   */
  void compute_system_operators(const std::vector<shared_ptr<typename LADealII::BlockMatrix> > &,
                                LinearOperator<typename LADealII::VectorType> &,
                                LinearOperator<typename LADealII::VectorType> &,
                                LinearOperator<typename LADealII::VectorType> &) const;

  /**
   * Call the reinit of the dealii::FEValues with the given cell, and
   * cache the local solution, solution_dot and explicit_solution, and
   * properly sets the independent degrees of freedom to work with
   * Sacado.
   */
  template<typename Number>
  void reinit(const Number &alpha,
              const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
              FEValuesCache<dim,spacedim> &fe_cache) const;

  /**
   * Call the reinit of the dealii::FEFaceValues with the given cell,
   * and cache the local solution, solution_dot and explicit_solution,
   * and properly sets the independent degrees of freedom to work with
   * Sacado.
   */
  template<typename Number>
  void reinit(const Number &alpha,
              const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
              const unsigned int face_no,
              FEValuesCache<dim,spacedim> &fe_cache) const;


  /** @} */


  /** @name Functions dedicated to set properties of the interface */
  /** @{ */

  /**
   * Declare parameters.
   */
  virtual void declare_parameters (ParameterHandler &prm);

  /**
   * Return the vector @p of differential blocks.
   */
  const std::vector<unsigned int> get_differential_blocks() const;

  /**
   * Return the mapping used when no different mapping has been
   *  specified.
   * Return dealii::StaticMappingQ1<dim,spacedim>::mapping by default.
   */
  virtual const Mapping<dim,spacedim> &get_default_mapping() const;

  /**
   * Return the mapping to use with the output.
   */
  virtual const Mapping<dim,spacedim> &get_output_mapping() const;

  /**
   * Return the mapping to use with FEValues.
   */
  virtual const Mapping<dim,spacedim> &get_fe_mapping() const;

  /**
   * Return the mapping to use with boundary conditions.
   */
  virtual const Mapping<dim,spacedim> &get_bc_mapping() const;

  /**
   * Return the mapping to use with the class
   *  dealii::KellyErrorEstimator< dim, spacedim >.
   */
  virtual const Mapping<dim,spacedim> &get_kelly_mapping() const;

  /**
   * Return the mapping to use with errors and convergence
   * rates.
   */
  virtual const Mapping<dim,spacedim> &get_error_mapping() const;

  /**
   * Return the mapping to use with interpolation of functions..
   */
  virtual const Mapping<dim,spacedim> &get_interpolate_mapping() const;

  /**
   * Return the mapping to use with projection of functions...
   */
  virtual const Mapping<dim,spacedim> &get_project_mapping() const;

  /**
   * This function is called in order to know what are the update flags
   * on the face cell. By default it returns
   * (update_values         | update_quadrature_points  |
   *  update_normal_vectors | update_JxW_values);
   * If you want to use different update flags you need to overwrite
   * this function.
   */
  virtual UpdateFlags get_face_update_flags() const;

  /**
   * This function is called in order to know what are the update flags
   * on the cell. By default it returns
   * (update_values         | update_quadrature_points  |
   *  update_gradients      | update_JxW_values);
   * If you want to use different update flags you need to overwrite
   * this function.
   */
  virtual UpdateFlags get_cell_update_flags() const;
  /** @} */




  /**
   * This function is called to get the coupling of the @p i-th matrix
   */
  const Table<2,DoFTools::Coupling> &get_matrix_coupling(const unsigned int &i) const;

  /**
   * Return the component names.
   */
  std::string get_component_names() const;

  virtual void estimate_error_per_cell(Vector<float> &estimated_error) const;

  /**
   * Number of components
   */
  const unsigned int n_components;

  /**
   * Number of matrices to be assembled
   */
  const unsigned int n_matrices;


  /**
   * ParsedFiniteElement.
   */
  ParsedFiniteElement<dim,spacedim> pfe;
  /**
  * set internal variable stepper to the time stepper used by pidomus
  * this is needed in the fix_solution_dot_derivative() functions
  * @param s
  */
  void set_stepper(const std::string &s) const;

  /**
    * Return norm of vector @p v. This function is called by the IMEX stepper.
    * By default it returns the l2 norm.
    */
  virtual double vector_norm(const typename LAC::VectorType &v) const;

protected:

  /** @name Helper functions */
  /** @{ */

  /**
   * Do nothing but is required for Sacado types.
   */
  void fix_solution_dot_derivative(FEValuesCache<dim,spacedim> &, double) const;

  /**
   * Fix the "Sacado" and "non-Sacado" parts of y_dot.
   */
  void fix_solution_dot_derivative(FEValuesCache<dim,spacedim> &fe_cache, Sdouble alpha) const;

  /**
   * Fix the "Sacado" and "non-Sacado" parts of y_dot.
   */
  void fix_solution_dot_derivative(FEValuesCache<dim,spacedim> &fe_cache, SSdouble alpha) const;
  /** @} */


  void build_couplings();

  /**
   * Define the coupling of each matrix among its blocks.  By default it
   * sets a fully coupling among each block.  If you want to specify the
   * coupling you need to override this function and implement it
   * according to the following example
   * @code
   * void set_matrix_couplings(std::vector<std::string> &couplings) const
   * {
   *   // suppose we have 2 matrices (system and preconditioner)
   *   // we are solving incompressible Stokes equations
   *   couplings[0] = "1,1;1,0"; // first block is the velocity, the second is the pressure
   *   couplings[1] = "1,0;0,1";
   * }
   * @endcode
   */
  virtual void set_matrix_couplings(std::vector<std::string> &couplings) const;

  void convert_string_to_int(const std::string &str_coupling,
                             std::vector<std::vector<unsigned int> > &int_coupling) const;

  /**
   * Convert integer table into a coupling table.
   */
  Table<2, DoFTools::Coupling> to_coupling(const std::vector<std::vector<unsigned int> > &table) const;

  std::string str_diff_comp;

  std::vector<unsigned int> _diff_comp;

  unsigned int dofs_per_cell;
  unsigned int n_q_points;
  unsigned int n_face_q_points;

  std::vector<Table<2,DoFTools::Coupling> > matrix_couplings;

  mutable ParsedDataOut<dim, spacedim>            data_out;

  mutable std::string stepper;

};

template <int dim, int spacedim, typename LAC>
template<typename Number>
void
BaseInterface<dim,spacedim,LAC>::reinit(const Number &,
                                        const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                                        FEValuesCache<dim,spacedim> &fe_cache) const
{
  double dummy=0;
  Number alpha = this->get_alpha();
  fe_cache.reinit(cell);
  fe_cache.cache_local_solution_vector("explicit_solution",
                                       this->get_locally_relevant_explicit_solution(), dummy);
  fe_cache.cache_local_solution_vector("solution",
                                       this->get_locally_relevant_solution(), alpha);
  fe_cache.cache_local_solution_vector("solution_dot",
                                       this->get_locally_relevant_solution_dot(), alpha);
  this->fix_solution_dot_derivative(fe_cache, alpha);
}


template <int dim, int spacedim, typename LAC>
template<typename Number>
void
BaseInterface<dim,spacedim,LAC>::reinit(const Number &,
                                        const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                                        const unsigned int face_no,
                                        FEValuesCache<dim,spacedim> &fe_cache) const
{
  double dummy=0;
  Number alpha = this->get_alpha();
  fe_cache.reinit(cell, face_no);
  fe_cache.cache_local_solution_vector("explicit_solution",
                                       this->get_locally_relevant_explicit_solution(), dummy);
  fe_cache.cache_local_solution_vector("solution",
                                       this->get_locally_relevant_solution(), alpha);
  fe_cache.cache_local_solution_vector("solution_dot",
                                       this->get_locally_relevant_solution_dot(), alpha);
  this->fix_solution_dot_derivative(fe_cache, alpha);
}

#endif
