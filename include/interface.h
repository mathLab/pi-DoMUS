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
 *    - coupling
 *  - Preconditioner:
 *    - preconditioner coupling
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

#include "dof_utilities.h"
#include "parsed_finite_element.h"
#include "sak_data.h"
#include "parsed_function.h"
#include "parsed_mapped_functions.h"
#include "parsed_dirichlet_bcs.h"
#include "assembly.h"

template <int dim,int spacedim=dim, int n_components=1>
class Interface : public ParsedFiniteElement<dim,spacedim>
{
  typedef Assembly::Scratch::NFields<dim,spacedim> Scratch;
  typedef Assembly::CopyData::NFieldsPreconditioner<dim,spacedim> CopyPreconditioner;
  typedef Assembly::CopyData::NFieldsSystem<dim,spacedim> CopySystem;
public:
  Interface(const std::string &name="",
            const std::string &default_fe="FE_Q(1)",
            const std::string &default_component_names="u",
            const std::string &default_coupling="",
            const std::string &default_preconditioner_coupling="") :
    ParsedFiniteElement<dim,spacedim>(name, default_fe, default_component_names,
                                      n_components, default_coupling, default_preconditioner_coupling),
    forcing_terms("Forcing terms", default_component_names, "0=ALL"),
    neumann_bcs("Neumann boundary conditions", default_component_names, "0=ALL"),
    dirichlet_bcs("Dirichlet boundary conditions", default_component_names, "0=ALL")
  {};

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
    dirichlet_bcs.interpolate_boundary_values(dof_handler,constraints);
  };

  /**
   * Applies Neumann boundary conditions
   *
   * This function is used to applies Neumann boundary conditions.
   * It takes as argument a DoF handler @p dof_handler and a constraint 
	 * matrix @p constraints.
   */
  virtual void apply_neumann_bcs (const DoFHandler<dim,spacedim> &dof_handler,
                                    ConstraintMatrix &constraints) const
  {
    Assert(false, ExcPureFunctionCalled ());
  };

  virtual void set_time(const double t) const
  {
    // boundary_conditions.set_time(t);
    forcing_term.set_time(t);
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
=======

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
>>>>>>> first version of parsed BC and forcing terms
  virtual void initialize_data(const unsigned int &dofs_per_cell,
                               const unsigned int &n_q_points,
                               const unsigned int &n_face_q_points,
                               const TrilinosWrappers::MPI::BlockVector &solution,
                               const TrilinosWrappers::MPI::BlockVector &solution_dot,
                               const double t,
                               const double alpha,
                               SAKData &d) const;

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
   * (DoFHandler<dim,spacedim>::active_cell_iterator), all the informations of the
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
                                        const TrilinosWrappers::BlockSparseMatrix &,
                                        const TrilinosWrappers::BlockSparseMatrix &,
                                        LinearOperator<TrilinosWrappers::MPI::BlockVector> &,
                                        LinearOperator<TrilinosWrappers::MPI::BlockVector> &) const
  {
    Assert(false, ExcPureFunctionCalled ());
  }


  virtual void assemble_local_system (const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                                      Scratch &scratch,
                                      CopySystem &data) const
  {

    const unsigned int   dofs_per_cell   = scratch.fe_values.dofs_per_cell;
    const unsigned int   n_q_points      = scratch.fe_values.n_quadrature_points;

    scratch.fe_values.reinit (cell);
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

    const unsigned int   dofs_per_cell   = scratch.fe_values.dofs_per_cell;
    const unsigned int   n_q_points      = scratch.fe_values.n_quadrature_points;

    scratch.fe_values.reinit (cell);
    cell->get_dof_indices (data.local_dof_indices);

    data.local_matrix = 0;

    get_preconditioner_residual(cell, scratch, data, data.sacado_residual);

    for (unsigned int i=0; i<dofs_per_cell; ++i)
      for (unsigned int j=0; j<dofs_per_cell; ++j)
        data.local_matrix(i,j) = data.sacado_residual[i].dx(j);

  }

  typedef TrilinosWrappers::MPI::BlockVector VEC;

  virtual shared_ptr<Mapping<dim,spacedim> > get_mapping(const DoFHandler<dim,spacedim> &,
                                                         const VEC &) const
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


protected:
  ParsedMappedFunctions<spacedim,n_components>  forcing_terms; // on the volume
  ParsedMappedFunctions<spacedim,n_components>  neumann_bcs;
private:
  ParsedDirichletBCs<dim,spacedim,n_components> dirichlet_bcs;


};


template<int dim, int spacedim, int n_components>
void Interface<dim,spacedim,n_components>::initialize_data(const unsigned int &dofs_per_cell,
                                                           const unsigned int &n_q_points,
                                                           const unsigned int &n_face_q_points,
                                                           const TrilinosWrappers::MPI::BlockVector &solution,
                                                           const TrilinosWrappers::MPI::BlockVector &solution_dot,
                                                           const double t,
                                                           const double alpha,
                                                           SAKData &d) const
{
  d.add_copy(dofs_per_cell, "dofs_per_cell");
  d.add_copy(n_q_points, "n_q_points");
  d.add_copy(n_face_q_points, "n_face_q_points");
  d.add_ref(solution, "solution");
  d.add_ref(solution_dot, "solution_dot");
  d.add_copy(t, "t");
  d.add_copy(alpha, "alpha");
}

#endif
