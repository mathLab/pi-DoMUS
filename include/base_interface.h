#ifndef _pidomus_base_interface_h_
#define _pidomus_base_interface_h_

#include <deal.II/lac/linear_operator.h>
#include <deal.II/base/sacado_product_type.h>

#include <deal.II/numerics/error_estimator.h>

#include <deal2lkit/parsed_finite_element.h>
#include <deal2lkit/parsed_function.h>
#include <deal2lkit/parsed_mapped_functions.h>
#include <deal2lkit/parsed_dirichlet_bcs.h>
#include <deal2lkit/parsed_data_out.h>


#include "copy_data.h"
#include "lac/lac_type.h"

using namespace pidomus;
/**
 * Base Interface
 *
 * *Goal*: provide a derivable interface to solve a particular PDE
 * System (time dependent, first-order, non linear).
 *
 * *Interface*: This class provides some default implementations of the
 * standard requirements for Systems of PDEs (finite element
 * definition, boundary and initial conditions, etc.) and provides an
 * unified interface to fill the local (cell-wise) contributions of
 * all matrices and residuals required for the definition of the problem.
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
 */
template <int dim,int spacedim=dim, typename LAC=LATrilinos>
class BaseInterface : public ParameterAcceptor
{

public:

  /**
   * virtual destructor.
   */
  virtual ~BaseInterface() {};

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

  void init ();


  virtual void declare_parameters (ParameterHandler &prm);

  /**
   * Return the vector @p of differential blocks
   */
  const std::vector<unsigned int> get_differential_blocks() const;

  /**
   * Postprocess the newly generated triangulation.
   */
  virtual void postprocess_newly_created_triangulation(Triangulation<dim, spacedim> &tria) const;

  /**
   * Initialize all data required for the system
   *
   * This function is used to initialize the internal variables
   * according to the given arguments, which are @p dof_handler, @p solution,
   * @p solution_dot, @p explicit_solution, @p t and @p alpha.
   */
  virtual void initialize_data(const DoFHandler<dim,spacedim> &dof_handler,
                               const typename LAC::VectorType &solution,
                               const typename LAC::VectorType &solution_dot,
                               const typename LAC::VectorType &explicit_solution,
                               const double t,
                               const double alpha) const;


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

  /**
   * Compute linear operators needed by the problem
   *
   * This function is used to assemble linear operators related
   * to the problem. It is only needed if we use iterative solvers.
   */
  virtual void compute_system_operators(const std::vector<shared_ptr<typename LATrilinos::BlockMatrix> >,
                                        LinearOperator<typename LATrilinos::VectorType> &,
                                        LinearOperator<typename LATrilinos::VectorType> &) const;

  /**
   * Compute linear operators needed by the problem. When using deal.II vector and matrix types, this
   * function is empty, since a direct solver is used by default.
   */
  void compute_system_operators(const std::vector<shared_ptr<typename LADealII::BlockMatrix> >,
                                LinearOperator<typename LADealII::VectorType> &,
                                LinearOperator<typename LADealII::VectorType> &) const;



  virtual const Mapping<dim,spacedim> &get_mapping() const;

  virtual UpdateFlags get_face_update_flags() const;

  virtual UpdateFlags get_cell_update_flags() const;

  void fix_solution_dot_derivative(FEValuesCache<dim,spacedim> &, double) const;

  void fix_solution_dot_derivative(FEValuesCache<dim,spacedim> &fe_cache, Sdouble alpha) const;


  void fix_solution_dot_derivative(FEValuesCache<dim,spacedim> &fe_cache, SSdouble alpha) const;

  template<typename Number>
  void reinit(const Number &alpha,
              const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
              FEValuesCache<dim,spacedim> &fe_cache) const;


  template<typename Number>
  void reinit(const Number &alpha,
              const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
              const unsigned int face_no,
              FEValuesCache<dim,spacedim> &fe_cache) const;



  const Table<2,DoFTools::Coupling> &get_matrix_coupling(const unsigned int &i) const;

  const unsigned int n_components;
  const unsigned int n_matrices;

  ParsedFiniteElement<dim,spacedim> pfe;

  /**
   * Return the component names.
   */
  std::string get_component_names() const;

  virtual void estimate_error_per_cell(const DoFHandler<dim,spacedim> &dof,
                                       const typename LAC::VectorType &solution,
                                       Vector<float> &estimated_error) const;

protected:



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

  /**
   * Solution vector evaluated at time t
   */
  mutable const typename LAC::VectorType *solution;

  /**
   * Solution vector evaluated at time t-dt
   */
  mutable const typename LAC::VectorType *explicit_solution;

  /**
   * Time derivative solution vector evaluated at time t
   */
  mutable const typename LAC::VectorType *solution_dot;

  mutable const DoFHandler<dim,spacedim> *dof_handler;

  /**
   * Current time step
   */
  mutable double t;
  mutable double alpha;
  unsigned int dofs_per_cell;
  unsigned int n_q_points;
  unsigned int n_face_q_points;

  std::vector<Table<2,DoFTools::Coupling> > matrix_couplings;
  mutable ParsedDataOut<dim, spacedim>            data_out;



};

template <int dim, int spacedim, typename LAC>
template<typename Number>
void
BaseInterface<dim,spacedim,LAC>::reinit(const Number &,
                                        const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                                        FEValuesCache<dim,spacedim> &fe_cache) const
{
  double dummy=0;
  Number alpha = this->alpha;
  fe_cache.reinit(cell);
  fe_cache.cache_local_solution_vector("explicit_solution", *this->explicit_solution, dummy);
  fe_cache.cache_local_solution_vector("solution", *this->solution, alpha);
  fe_cache.cache_local_solution_vector("solution_dot", *this->solution_dot, alpha);
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
  Number alpha = this->alpha;
  fe_cache.reinit(cell, face_no);
  fe_cache.cache_local_solution_vector("explicit_solution", *this->explicit_solution, dummy);
  fe_cache.cache_local_solution_vector("solution", *this->solution, alpha);
  fe_cache.cache_local_solution_vector("solution_dot", *this->solution_dot, alpha);
  this->fix_solution_dot_derivative(fe_cache, alpha);
}

#endif
