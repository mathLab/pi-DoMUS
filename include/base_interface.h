/**
 * Base Interface
 *
 * Goal: provide a derivable interface to solve a particular
 *       PDE problem (time depending, first-order, non linear).
 *
 * Usage: This class requires some arguments related to the setting
 *        of the problem: finite elements, boundary conditions,
 *        and initial conditions.
 *        Moreover, it helps to write the system matrix and
 *        the preconditioner matrix.
 * Variables:
 *  - General:
 *    - Finite Elements
 *    - Boundary conditions ( Dirichlet, Neumann, and Robin )
 *    - Initial conditions ( y(0) and d/dt y (0) )
 *  - Couplings:
 *    - coupling variable is a matrix of zeroes and ones: if the row
 *      and the coloumn are indipendent there will be 0, otherwise 1
 *  - @p default_differential_components : this variable is a list of ones and
 *    zeroes. 1 in the case the corresponding variable should be differentiable
 *    and 0 otherwise.
 */

#ifndef _pidomus_base_interface_h_
#define _pidomus_base_interface_h_

#include <deal.II/lac/linear_operator.h>
/* #include <deal.II/lac/linear_operator.h> */
/* #include <deal.II/lac/block_linear_operator.h> */
/* #include <deal.II/lac/packaged_operation.h> */
/* #include <deal.II/lac/solver_cg.h> */
#include <deal.II/base/sacado_product_type.h>


#include <deal2lkit/parsed_finite_element.h>
#include <deal2lkit/parsed_function.h>
#include <deal2lkit/parsed_mapped_functions.h>
#include <deal2lkit/parsed_dirichlet_bcs.h>

#include "copy_data.h"
#include "lac/lac_type.h"

using namespace pidomus;

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
   * according to the given arguments, which are
   * @p solution, @p solution_dot, @p t and @p alpha.
   */
  virtual void initialize_data(const typename LAC::VectorType &solution,
                               const typename LAC::VectorType &solution_dot,
                               const double t,
                               const double alpha) const;


  /**
   * Assemble energies and residuals
   */
  virtual void assemble_energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &,
                                               FEValuesCache<dim,spacedim> &,
                                               std::vector<Sdouble> &energies,
                                               std::vector<std::vector<double> > &local_residuals,
                                               bool compute_only_system_terms) const;

  /**
   * Assemble energies and residuals
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
   * to the problem.
   */
  virtual void compute_system_operators(const DoFHandler<dim,spacedim> &,
                                        const std::vector<shared_ptr<typename LATrilinos::BlockMatrix> >,
                                        LinearOperator<typename LATrilinos::VectorType> &,
                                        LinearOperator<typename LATrilinos::VectorType> &) const;

  void compute_system_operators(const DoFHandler<dim,spacedim> &,
                                const std::vector<shared_ptr<typename LADealII::BlockMatrix> >,
                                LinearOperator<typename LADealII::VectorType> &,
                                LinearOperator<typename LADealII::VectorType> &) const
  {};


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


protected:

  void build_couplings();

  /**
   * Define the coupling of each matrix among its blocks.  By default it
   * sets a fully coupling among each block.  If you want to specify the
   * coupling you need to override this function and implement it
   * according to the following example
   * @code
   * void set_matrix_couplings(std::vector<std::string &couplings) const
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
  mutable typename LAC::VectorType old_solution;

  /**
   * Time derivative solution vector evaluated at time t
   */
  mutable const typename LAC::VectorType *solution_dot;

  /**
   * Current time step
   */
  mutable double t;

  /**
   * Previous time step
   */
  mutable double old_t;

  mutable double alpha;
  unsigned int dofs_per_cell;
  unsigned int n_q_points;
  unsigned int n_face_q_points;

  std::vector<Table<2,DoFTools::Coupling> > matrix_couplings;


};

template <int dim, int spacedim, typename LAC>
template<typename Number>
void
BaseInterface<dim,spacedim,LAC>::reinit(const Number &alpha,
                                        const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                                        FEValuesCache<dim,spacedim> &fe_cache) const
{
  fe_cache.reinit(cell);
  fe_cache.cache_local_solution_vector("old_solution", this->old_solution, alpha);
  fe_cache.cache_local_solution_vector("solution", *this->solution, alpha);
  fe_cache.cache_local_solution_vector("solution_dot", *this->solution_dot, alpha);
  this->fix_solution_dot_derivative(fe_cache, alpha);
}


template <int dim, int spacedim, typename LAC>
template<typename Number>
void
BaseInterface<dim,spacedim,LAC>::reinit(const Number &alpha,
                                        const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                                        const unsigned int face_no,
                                        FEValuesCache<dim,spacedim> &fe_cache) const
{
  fe_cache.reinit(cell, face_no);
  fe_cache.cache_local_solution_vector("old_solution", this->old_solution, alpha);
  fe_cache.cache_local_solution_vector("solution", *this->solution, alpha);
  fe_cache.cache_local_solution_vector("solution_dot", *this->solution_dot, alpha);
  this->fix_solution_dot_derivative(fe_cache, alpha);
}

#endif
