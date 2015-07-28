/**
 * Interface
 *
 * This class has two child: conservative_interface.h and
 *  non_conservative_interface.h
 * Users should not derive directly from this class,
 * but from its specialization classes.
 * For istance, a stokes problem should consist in a
 * interface class derived from conservative_interface.h.
 * (see include/interfaces/ for some examples)
 *
 * Goal: provide a derivable interface to solve a particular
 *       PDEs problem (time depending, first-order, non linear).
 *
 * Usage: This class requires some arguments related to the setting
 *        of the problem: finite elements, boundary conditions,
 *        and initial conditions.
 *        Moreover, it helps to write the system matrix and
 *        the preconditioner matrix.
 *        (see conservative_interface.h and
 *        non_conservative_interface.h)
 *
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

#ifndef _interface_h_
#define _interface_h_

#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/trilinos_block_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/block_linear_operator.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal2lkit/dof_utilities.h>
#include <deal2lkit/parsed_finite_element.h>
#include <deal2lkit/sak_data.h>
#include <deal2lkit/parsed_function.h>
#include <deal2lkit/parsed_mapped_functions.h>
#include <deal2lkit/parsed_dirichlet_bcs.h>
#include "assembly.h"
#include <deal2lkit/utilities.h>
#include "lac_type.h"
#include <deal.II/base/sacado_product_type.h>

template <int dim,int spacedim=dim, int n_components=1, typename LAC=LATrilinos>
class Interface : public ParsedFiniteElement<dim,spacedim>
{
  typedef FEValuesCache<dim,spacedim> Scratch;
  typedef Assembly::CopyData::NFieldsPreconditioner<dim,dim> CopyPreconditioner;
  typedef Assembly::CopyData::NFieldsSystem<dim,dim> CopySystem;

public:

  virtual ~Interface() {};

  Interface(const std::string &name="",
            const std::string &default_fe="FE_Q(1)",
            const std::string &default_component_names="u",
            const std::string &default_coupling="",
            const std::string &default_preconditioner_coupling="",
            const std::string &default_differential_components="") :
    ParsedFiniteElement<dim,spacedim>(name, default_fe, default_component_names,
                                      n_components, default_coupling, default_preconditioner_coupling),
    forcing_terms("Forcing terms", default_component_names, "0=ALL"),
    neumann_bcs("Neumann boundary conditions", default_component_names, "0=ALL"),
    dirichlet_bcs("Dirichlet boundary conditions", default_component_names, "0=ALL"),
    str_diff_comp(default_differential_components)
  {};

  virtual void declare_parameters (ParameterHandler &prm);
  virtual void parse_parameters_call_back ();
  const std::vector<unsigned int> get_differential_blocks() const;

  /**
   * update time of all parsed mapped functions
   */
  virtual void set_time (const double &t) const
  {
    dirichlet_bcs.set_time(t);
    forcing_terms.set_time(t);
    neumann_bcs.set_time(t);
  }

  /**
   * Applies Dirichlet boundary conditions
   *
   * This function is used to applies Dirichlet boundary conditions.
   * It takes as argument a DoF handler @p dof_handler and a constraint
   * matrix @p constraints.
   */
  virtual void apply_dirichlet_bcs (const DoFHandler<dim,spacedim> &dof_handler,
                                    ConstraintMatrix &constraints) const
  {
    if (this->operator()()->has_support_points())
      dirichlet_bcs.interpolate_boundary_values(dof_handler,constraints);
    else
      {
        const QGauss<dim-1> quad(this->operator()()->degree+1);
        dirichlet_bcs.project_boundary_values(dof_handler,quad,constraints);
      }
    dirichlet_bcs.compute_nonzero_normal_flux_constraints(dof_handler,constraints);
  };

  /**
   * Applies Neumann boundary conditions
   *
   */
  template<typename Number>
  void apply_neumann_bcs (const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                          Scratch &scratch,
                          CopySystem &data,
                          Number &energy) const
  {

    Number dummy = 0.0;

    for (unsigned int face=0; face < GeometryInfo<dim>::faces_per_cell; ++face)
      {
        unsigned int face_id = cell->face(face)->boundary_id();
        if (cell->face(face)->at_boundary() && neumann_bcs.acts_on_id(face_id))
          {
            this->reinit(dummy, cell, face, scratch);

            auto &vars = scratch.get_values("solution", dummy);
            auto &q_points = scratch.get_quadrature_points();
            auto &JxW = scratch.get_JxW_values();

            for (unsigned int q=0; q<q_points.size(); ++q)
              {
                Vector<double> T(n_components);
                neumann_bcs.get_mapped_function(face_id)->vector_value(q_points[q], T);

                for (unsigned int i=0; i < n_components; ++i)
                  if (neumann_bcs.get_mapped_mask(face_id)[i])
                    {
                      const std::vector<Number> &var_face = vars[q];

                      energy -= (T[i]*var_face[i])*JxW[q];
                    }
              }
            break;
          }
      }
  }


  /**
   * Applies Forcing terms
   *
   */
  template<typename Number>
  void apply_forcing_terms (const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                            Scratch &scratch,
                            CopySystem &data,
                            Number &energy) const
  {
    unsigned cell_id = cell->material_id();
    if (forcing_terms.acts_on_id(cell_id))
      {
        Number dummy = 0.0;
        this->reinit(dummy, cell, scratch);

        auto &vars = scratch.get_values("solution", dummy);
        auto &q_points = scratch.get_quadrature_points();
        auto &JxW = scratch.get_JxW_values();

        for (unsigned int q=0; q<q_points.size(); ++q)
          {
            const std::vector<Number> &var = vars[q]; // u,u,u
            for (unsigned int i=0; i < n_components; ++i)
              if (forcing_terms.get_mapped_mask(cell_id)[i])
                {
                  double B = forcing_terms.get_mapped_function(cell_id)->value(q_points[q],i);
                  energy -= B*var[i]*JxW[q];
                }
          }
      }
  }

  /**
   * Initialize all data required for the system
   *
   * This function is used to initialize the varibale SAKData @p d
   * that contains all data of the problem (solutions, DoF, quadrature
   * points, solutions vector, etc ).
   * It takes as argument the number of DoF per cell @p dofs_per_cell,
   * the number of quadrature points @p n_q_points, the number of
   * quadrature points per face @p n_face_q_points, the reference to
   * solutions vectors @p sol and the reference to the SAKData @p d.
   *
   * TODO: add current_time and current_alpha
   */
  virtual void initialize_data(const typename LAC::VectorType &solution,
                               const typename LAC::VectorType &solution_dot,
                               const double t,
                               const double alpha) const;


  /**
   * Build the energy needed to get the preconditioner in the case
   * it is required just one derivative.
   *
   * This function is used to build the energy associated to the preconditioner
   * in the case it is required just one derivative.
   * It takes as argument a reference to the active cell
   * (DoFHandler<dim,spacedim>::active_cell_iterator), all the informations of the
   * system  such as fe values, quadrature points, SAKData (Scratch),
   * all the informations related to the PDE (CopySystem) and the energy
   * (Sdouble)
   */
  virtual void get_preconditioner_energy(const typename DoFHandler<dim,spacedim>::active_cell_iterator &,
                                         Scratch &,
                                         CopySystem &,
                                         Sdouble &) const
  {
    Assert(false, ExcPureFunctionCalled ());
  }

  /**
   * Build the energy needed to get the preconditioner in the case
   * it is required two derivatives.
   *
   * This function is used to build the energy associated to the preconditioner
   *in the case the Jacobian is automatically constructed using the derivative of the residual.
   * It takes as argument a reference to the active cell
   * (DoFHandler<dim,spacedim>::active_cell_iterator), all the informations of the
   * system  such as fe values, quadrature points, SAKData (Scratch),
   * all the informations related to the PDE (CopySystem) and the energy
   * (SSdouble)
   */
  virtual void get_preconditioner_energy(const typename DoFHandler<dim,spacedim>::active_cell_iterator &,
                                         Scratch &,
                                         CopyPreconditioner &,
                                         SSdouble &)  const
  {
    Assert(false, ExcPureFunctionCalled ());
  }

  /**
   * Build the energy needed to get the system matrix in the case
   * it is required two derivatives.
   *
   * This function is used to build the energy associated to the system matrix
   *in the case two derivatives are required.
   * It takes as argument a reference to the active cell
   * (DoFHandler<dim,spacedim>::active_cell_iterator), all the informations of the
   * system  such as fe values, quadrature points, SAKData (Scratch),
   * all the informations related to the PDE (CopySystem) and the energy
   * (SSdouble)
   */
  virtual void get_system_energy(const typename DoFHandler<dim,spacedim>::active_cell_iterator &,
                                 Scratch &,
                                 CopySystem &,
                                 Sdouble &) const
  {
    Assert(false, ExcPureFunctionCalled ());
  }

  /**
   * Build the energy needed to get the system matrix in the case
   * it is required two derivatives.
   *
   * This function is used to build the energy associated to the system matrix
   *in the case two derivatives are required.
   * It takes as argument a reference to the active cell
   * (DFHandler<dim,spacedim>::active_cell_iterator), all the informations of the
   * system  such as fe values, quadrature points, SAKData (Scratch),
   * all the informations related to the PDE (CopySystem) and the energy
   * (SSdouble)
   */
  virtual void get_system_energy(const typename DoFHandler<dim,spacedim>::active_cell_iterator &,
                                 Scratch &,
                                 CopySystem &,
                                 SSdouble &)  const
  {
    Assert(false, ExcPureFunctionCalled ());
  }

  /**
   * Build the residual needed to get the system matrix in the case
   * it is required two derivatives.
   *
   * This function is used to build the residual associated to the system
   *in the case two derivatives are required.
   * It takes as argument a reference to the active cell
   * @p cell, all the informations of the system @p scratch ( fe values,
   * quadrature points, SAKData ), all the informations related to the PDE
   * @p data and a reference to the local residual @p local_residual.
   */
  virtual void get_system_residual (const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                                    Scratch &scratch,
                                    CopySystem &data,
                                    std::vector<Sdouble> &local_residual) const
  {
    SSdouble energy;
    get_system_energy(cell, scratch, data, energy);

    apply_forcing_terms(cell,scratch,data,energy);

    if (cell->at_boundary())
      apply_neumann_bcs(cell,scratch, data, energy);

    for (unsigned int i=0; i<local_residual.size(); ++i)
      {
        local_residual[i] = energy.dx(i);
      }
  }

  /**
   * Build the residual needed to get the system matrix in the case
   * it is required just one derivative.
   *
   * This function is used to build the residual associated to the system
   * in the case it is required just one derivatice.
   * It takes as argument a reference to the active cell
   * @p cell, all the informations of the system @p scratch ( fe values,
   * quadrature points, SAKData ), all the informations related to the PDE
   * @p data and a reference to the local residual @p local_residual.
   */
  virtual void get_system_residual (const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                                    Scratch &scratch,
                                    CopySystem &data,
                                    std::vector<double> &local_residual) const
  {
    Sdouble energy;
    get_system_energy(cell, scratch, data, energy);

    apply_forcing_terms(cell,scratch,data,energy);
    if (cell->at_boundary())
      apply_neumann_bcs(cell,scratch, data, energy);

    for (unsigned int i=0; i<local_residual.size(); ++i)
      {
        local_residual[i] = energy.dx(i);
      }
  }

  /**
   * Build the residual needed to get the preconditioner matrix in the case
   * two derivatives are required.
   *
   * This function is used to build the residual associated to the preconditioner
   * in the case it is required just one derivatice.
   * It takes as argument a reference to the active cell
   * @p cell, all the informations of the system @p scratch ( fe values,
   * quadrature points, SAKData ), all the informations related to the PDE
   * @p data and a reference to the local residual @p local_residual.
   */
  virtual void get_preconditioner_residual (const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                                            Scratch &scratch,
                                            CopyPreconditioner &data,
                                            std::vector<Sdouble> &local_residual) const
  {
    SSdouble energy;
    get_preconditioner_energy(cell, scratch, data, energy);
    for (unsigned int i=0; i<local_residual.size(); ++i)
      {
        local_residual[i] = energy.dx(i);
      }
  }
  /**
   * Compute linear operators needed by the problem
   *
   * This function is used to assemble linear operators related
   * to the problem.
   * It takes a reference to DoF Handler, two references
   * to block sparse matrices representing the system matrix and
   * the preconditioner, and two references to LinearOperator.
   */
  virtual void compute_system_operators(const DoFHandler<dim,spacedim> &,
                                        const typename LAC::BlockMatrix &,
                                        const typename LAC::BlockMatrix &,
                                        LinearOperator<typename LAC::VectorType> &,
                                        LinearOperator<typename LAC::VectorType> &) const
  {
    Assert(false, ExcPureFunctionCalled ());
  }


  virtual void assemble_local_system (const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                                      Scratch &scratch,
                                      CopySystem &data) const
  {
    const unsigned dofs_per_cell = data.local_dof_indices.size();

    cell->get_dof_indices (data.local_dof_indices);
    data.local_matrix = 0;

    get_system_residual(cell, scratch, data, data.sacado_residual);

    for (unsigned int i=0; i<dofs_per_cell; ++i)
      for (unsigned int j=0; j<dofs_per_cell; ++j)
        data.local_matrix(i,j) = data.sacado_residual[i].dx(j);
  }


  virtual void assemble_local_preconditioner (const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                                              Scratch &scratch,
                                              CopyPreconditioner &data) const
  {
    const unsigned dofs_per_cell = data.local_dof_indices.size();
    cell->get_dof_indices (data.local_dof_indices);

    data.local_matrix = 0;

    get_preconditioner_residual(cell, scratch, data, data.sacado_residual);

    for (unsigned int i=0; i<dofs_per_cell; ++i)
      for (unsigned int j=0; j<dofs_per_cell; ++j)
        data.local_matrix(i,j) = data.sacado_residual[i].dx(j);

  }


  virtual shared_ptr<Mapping<dim,spacedim> > get_mapping(const DoFHandler<dim,spacedim> &,
                                                         const typename LAC::VectorType &) const
  {
    return shared_ptr<Mapping<dim,spacedim> >(new MappingQ<dim,spacedim>(1));
  }

  virtual UpdateFlags get_jacobian_flags() const
  {
    return (update_quadrature_points |
            update_JxW_values |
            update_values |
            update_gradients);
  }

  virtual UpdateFlags get_residual_flags() const
  {
    return get_jacobian_flags();
  }

  virtual UpdateFlags get_jacobian_preconditioner_flags() const
  {
    return (update_JxW_values |
            update_values |
            update_gradients);
  }

  virtual UpdateFlags get_face_flags() const
  {
    return get_jacobian_flags();
  }

  void fix_solution_dot_derivative(FEValuesCache<dim,spacedim> &, double) const
  {}

  void fix_solution_dot_derivative(FEValuesCache<dim,spacedim> &fe_cache, Sdouble alpha) const
  {
    auto &sol = fe_cache.get_current_independent_local_dofs("solution", alpha);
    auto &sol_dot = fe_cache.get_current_independent_local_dofs("solution_dot", alpha);

    for (unsigned int i=0; i<sol.size(); ++i)
      sol_dot[i] = alpha.val()*sol[i] + (sol_dot[i].val() - alpha.val()*sol[i].val());
  }


  void fix_solution_dot_derivative(FEValuesCache<dim,spacedim> &fe_cache, SSdouble alpha) const
  {
    auto &sol = fe_cache.get_current_independent_local_dofs("solution", alpha);
    auto &sol_dot = fe_cache.get_current_independent_local_dofs("solution_dot", alpha);

    for (unsigned int i=0; i<sol.size(); ++i)
      sol_dot[i] = (alpha.val().val()*sol[i]) + (sol_dot[i].val().val() - alpha.val().val()*sol[i].val().val());
  }

  template<typename Number>
  void reinit(const Number &alpha,
              const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
              FEValuesCache<dim,spacedim> &fe_cache) const
  {
    fe_cache.reinit(cell);
    fe_cache.cache_local_solution_vector("solution", *this->solution, alpha);
    fe_cache.cache_local_solution_vector("solution_dot", *this->solution_dot, alpha);
    this->fix_solution_dot_derivative(fe_cache, alpha);
  };


  template<typename Number>
  void reinit(const Number &alpha,
              const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
              const unsigned int face_no,
              FEValuesCache<dim,spacedim> &fe_cache) const
  {
    fe_cache.reinit(cell, face_no);
    fe_cache.cache_local_solution_vector("solution", *this->solution, alpha);
    fe_cache.cache_local_solution_vector("solution_dot", *this->solution_dot, alpha);
    this->fix_solution_dot_derivative(fe_cache, alpha);
  };




protected:
  mutable ParsedMappedFunctions<spacedim,n_components>  forcing_terms; // on the volume
  mutable ParsedMappedFunctions<spacedim,n_components>  neumann_bcs;
  mutable ParsedDirichletBCs<dim,spacedim,n_components> dirichlet_bcs;
  std::string str_diff_comp;
  std::vector<unsigned int> _diff_comp;

  mutable const typename LAC::VectorType *solution;
  mutable const typename LAC::VectorType *solution_dot;

  mutable double alpha;
  mutable double t;
  mutable unsigned int dofs_per_cell;
  mutable unsigned int n_q_points;
  mutable unsigned int n_face_q_points;

};


template<int dim, int spacedim, int n_components, typename LAC>
void Interface<dim,spacedim,n_components,LAC>::initialize_data(const typename LAC::VectorType &solution,
    const typename LAC::VectorType &solution_dot,
    const double t,
    const double alpha) const
{
  this->solution = &solution;
  this->solution_dot = &solution_dot;
  this->alpha = alpha;
  this->t = t;
}

template<int dim, int spacedim, int n_components, typename LAC>
void Interface<dim,spacedim,n_components,LAC>::declare_parameters(ParameterHandler &prm)
{
  ParsedFiniteElement<dim,spacedim>::declare_parameters(prm);
  this->add_parameter(prm, &_diff_comp, "Block of differential components", str_diff_comp,
                      Patterns::List(Patterns::Integer(0,1),this->n_blocks(),this->n_blocks(),","),
                      "Set the blocks of differential components to 1"
                      "0 for algebraic");
}

template<int dim, int spacedim, int n_components, typename LAC>
void Interface<dim,spacedim,n_components,LAC>::parse_parameters_call_back()
{
  ParsedFiniteElement<dim,spacedim>::parse_parameters_call_back();
}

template<int dim, int spacedim, int n_components, typename LAC>
const std::vector<unsigned int> Interface<dim,spacedim,n_components,LAC>
::get_differential_blocks() const
{
  return _diff_comp;
}

#endif
