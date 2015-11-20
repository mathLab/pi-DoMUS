/**
 * Base Interface
 *
 * Goal: provide a derivable interface to solve a particular
 *       PDEs problem (time depending, first-order, non linear).
 *
 * Usage: This class requires some arguments related to the setting
 *        of the problem: finite elements, boundary conditions,
 *        and initial conditions.
 *        Moreover, it helps to write the system matrix and
 *        the preconditioner matrix.
 * Varibles:
 *  - General:
 *    - Finite Elements
 *    - Boundary conditions ( Dirichlet, Neumann, and Robin )
 *    - Initial conditions ( y(0) and d/dt y (0) )
 *  - System Matrix:
 *    - coupling (coupling variable is a matrix of zeroes and ones: if the row
 *      and the coloumn are indipendent there will be 0, otherwise 1)
 *  - Preconditioner:
 *    - preconditioner coupling (same as above)
 *  - @p default_differential_components : this variable is a list of ones and
 *    zeroes. 1 in the case the corresponding variable should be differentiable
 *    and 0 otherwise.
 *  TODO: add flags
 */

#ifndef _pidomus_base_interface_h_
#define _pidomus_base_interface_h_

#include <deal.II/lac/linear_operator.h>

#include <deal2lkit/parsed_finite_element.h>
#include <deal2lkit/parsed_function.h>
#include <deal2lkit/parsed_mapped_functions.h>
#include <deal2lkit/parsed_dirichlet_bcs.h>

#include "data/assembly.h"
#include "lac/lac_type.h"


template <int dim,int spacedim=dim, int n_components=1, typename LAC=LATrilinos>
class BaseInterface : public ParsedFiniteElement<dim,spacedim>
{

public:

  virtual ~BaseInterface() {};

  BaseInterface(const std::string &name="",
                const std::string &default_fe="FE_Q(1)",
                const std::string &default_component_names="u",
                const std::string &default_differential_components="") :
    ParsedFiniteElement<dim,spacedim>(name, default_fe, default_component_names,
                                      n_components),
    forcing_terms("Forcing terms", default_component_names, ""),
    neumann_bcs("Neumann boundary conditions", default_component_names, ""),
    dirichlet_bcs("Dirichlet boundary conditions", default_component_names, "0=ALL"),
    str_diff_comp(default_differential_components),
    old_t(-1.0)
  {
    n_matrices = get_number_of_matrices();
    mapping = set_mapping();
  };

  virtual void declare_parameters (ParameterHandler &prm);
  virtual void parse_parameters_call_back ();
  const std::vector<unsigned int> get_differential_blocks() const;

  /**
   * set time to @p t for forcing terms and boundary conditions
   */
  virtual void set_time (const double &t) const;


  /**
   * This function is used to modify triangulation using boundary_id or manifold_id.
   * In the case a signal is required, this is the function to modify.
   */
  virtual void postprocess_newly_created_triangulation(Triangulation<dim, spacedim> &tria) const;

  /**
   * Applies Dirichlet boundary conditions
   *
   * This function is used to applies Dirichlet boundary conditions.
   * It takes as argument a DoF handler @p dof_handler and a constraint
   * matrix @p constraints.
   */
  virtual void apply_dirichlet_bcs (const DoFHandler<dim,spacedim> &dof_handler,
                                    ConstraintMatrix &constraints) const;

  /**
   * Applies Neumann boundary conditions
   *
   */
  void apply_neumann_bcs (const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                          FEValuesCache<dim,spacedim> &scratch,
                          std::vector<double> &local_residual) const;


  /**
   * Applies CONSERVATIVE forcing terms.
   * This function applies the conservative forcing terms, which can be
   * defined by expressions in the parameter file.
   *
   * If the problem involves NON-conservative loads, they must be included
   * in the residual formulation.
   *
   */
  void apply_forcing_terms (const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                            FEValuesCache<dim,spacedim> &scratch,
                            std::vector<double> &local_residual) const;

  /**
   * Initialize all data required for the system
   *
   * This function is used to initialize the varibale AnyData @p d
   * that contains all data of the problem (solutions, DoF, quadrature
   * points, solutions vector, etc ).
   * It takes as argument the number of DoF per cell @p dofs_per_cell,
   * the number of quadrature points @p n_q_points, the number of
   * quadrature points per face @p n_face_q_points, the reference to
   * solutions vectors @p sol and the reference to the AnyData @p d.
   *
   * TODO: add current_time and current_alpha
   */
  virtual void initialize_data(const typename LAC::VectorType &solution,
                               const typename LAC::VectorType &solution_dot,
                               const double t,
                               const double alpha) const;


  /**
   * Definition of energies
   */
  virtual void get_energies(const typename DoFHandler<dim,spacedim>::active_cell_iterator &,
                            FEValuesCache<dim,spacedim> &,
                            std::vector<Sdouble> &,
			    bool compute_only_system_matrix) const;

  /**
   * Definition of energies
   */
  virtual void get_energies(const typename DoFHandler<dim,spacedim>::active_cell_iterator &,
                            FEValuesCache<dim,spacedim> &,
                            std::vector<SSdouble> &,
			    bool compute_only_system_matrix) const;


  /**
   * Definition of residulas
   */
  virtual void get_residuals (const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
			      FEValuesCache<dim,spacedim> &scratch,
			      std::vector<std::vector<double> > &local_residuals,
			      bool compute_only_system_matrix) const;


  /**
   * Definition of residulas
   */
  virtual void get_residuals (const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
			      FEValuesCache<dim,spacedim> &scratch,
			      std::vector<std::vector<Sdouble> > &local_residuals
			      bool compute_only_system_matrix) const;

/**
 * This function can be overloaded to directly implement the local
 * matrices (i.e. as it is usally done in standard FE codes)
 */
  virtual void assemble_local_matrices (const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                                        FEValuesCache<dim,spacedim> &scratch,
                                        CopyData &data) const;



  /**
   * Compute linear operators needed by the problem
   *
   * This function is used to assemble linear operators related
   * to the problem.
   */
  virtual void compute_system_operators(const DoFHandler<dim,spacedim> &,
                                        const std::vector<shared_ptr<typename LAC::BlockMatrix> >,
                                        LinearOperator<typename LAC::VectorType> &,
                                        LinearOperator<typename LAC::VectorType> &) const;


 shared_ptr<Mapping<dim,spacedim> > get_mapping() const;
virtual shared_ptr<Mapping<dim,spacedim> > set_mapping() const;

  virtual UpdateFlags get_face_flags() const;

  virtual void set_matrices_update_flags();

  UpdateFlags get_matrix_flags(const unsigned int &i) const;

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



  const table<2,DoFTools::Coupling> &get_matrix_coupling(const unsigned int &i) const;
  virtual unsigned int get_number_of_matrices() const;

protected:



  mutable ParsedMappedFunctions<spacedim,n_components>  forcing_terms; // on the volume
  mutable ParsedMappedFunctions<spacedim,n_components>  neumann_bcs;
  mutable ParsedDirichletBCs<dim,spacedim,n_components> dirichlet_bcs;
  mutable ParsedDirichletBCs<dim,spacedim,n_components> dirichlet_bcs_dot;

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

unsigned int n_matrices;

  std::vector<UpdateFlags> matrices_update_flags;
  std::vector<Table<2,DoFTools::Coupling> > matrices_coupling;
shared_ptr<Mapping<dim,spacedim> >  mapping;



};

template <int dim, int spacedim, int n_components, typename LAC>
template<typename Number>
void
BaseInterface<dim,spacedim,n_components,LAC>::reinit(const Number &alpha,
                                                     const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                                                     FEValuesCache<dim,spacedim> &fe_cache) const
{
  fe_cache.reinit(cell);
  fe_cache.cache_local_solution_vector("old_solution", this->old_solution, alpha);
  fe_cache.cache_local_solution_vector("solution", *this->solution, alpha);
  fe_cache.cache_local_solution_vector("solution_dot", *this->solution_dot, alpha);
  this->fix_solution_dot_derivative(fe_cache, alpha);
}


template <int dim, int spacedim, int n_components, typename LAC>
template<typename Number>
void
BaseInterface<dim,spacedim,n_components,LAC>::reinit(const Number &alpha,
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
