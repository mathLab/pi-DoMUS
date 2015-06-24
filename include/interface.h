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
    boundary_conditions("Dirichlet boundary conditions")
  {};

  virtual void apply_bcs (const DoFHandler<dim,spacedim> &dof_handler,
                          const FiniteElement<dim,spacedim> &fe,
                          ConstraintMatrix &constraints) const
  {
    VectorTools::interpolate_boundary_values (dof_handler,
                                              0,
                                              boundary_conditions,
                                              constraints);

  };


  virtual void initialize_data(const unsigned int &dofs_per_cell,
                               const unsigned int &n_q_points,
                               const unsigned int &n_face_q_points,
                               const std::vector<const TrilinosWrappers::MPI::BlockVector *> &sols,
                               SAKData &d) const;

  virtual void get_preconditioner_energy(const typename DoFHandler<dim,spacedim>::active_cell_iterator &,
                                         Scratch &,
                                         CopySystem &,
                                         Sdouble &) const
  {
    Assert(false, ExcPureFunctionCalled ());
  }

  virtual void get_preconditioner_energy(const typename DoFHandler<dim,spacedim>::active_cell_iterator &,
                                         Scratch &,
                                         CopyPreconditioner &,
                                         SSdouble &)  const
  {
    Assert(false, ExcPureFunctionCalled ());
  }

  virtual void get_system_energy(const typename DoFHandler<dim,spacedim>::active_cell_iterator &,
                                 Scratch &,
                                 CopySystem &,
                                 Sdouble &) const
  {
    Assert(false, ExcPureFunctionCalled ());
  }

  virtual void get_system_energy(const typename DoFHandler<dim,spacedim>::active_cell_iterator &,
                                 Scratch &,
                                 CopySystem &,
                                 SSdouble &)  const
  {
    Assert(false, ExcPureFunctionCalled ());
  }

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

  virtual void get_preconditioner_residual (const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                                            Scratch &scratch,
                                            CopyPreconditioner &data,
                                            std::vector<Sdouble> &local_residual) const
  {
    SSdouble energy;
    get_preconditioner_energy(cell, scratch, data,energy);
    for (unsigned int i=0; i<local_residual.size(); ++i)
      {
        local_residual[i] = energy.dx(i);
      }
  }

  virtual void compute_system_operators(const DoFHandler<dim,spacedim> &,
                                        const TrilinosWrappers::BlockSparseMatrix &,
                                        const TrilinosWrappers::BlockSparseMatrix &,
                                        LinearOperator<TrilinosWrappers::MPI::BlockVector> &,
                                        LinearOperator<TrilinosWrappers::MPI::BlockVector> &) const
  {
    Assert(false, ExcPureFunctionCalled ());
  }

protected:
  ParsedFunction<spacedim, n_components> boundary_conditions;

};


template<int dim, int spacedim, int n_components>
void Interface<dim,spacedim,n_components>::initialize_data(const unsigned int &dofs_per_cell,
                                                           const unsigned int &n_q_points,
                                                           const unsigned int &n_face_q_points,
                                                           const std::vector<const TrilinosWrappers::MPI::BlockVector *> &sols,
                                                           SAKData &d) const
{
  d.add_copy(dofs_per_cell, "dofs_per_cell");
  d.add_copy(n_q_points, "n_q_points");
  d.add_copy(n_face_q_points, "n_face_q_points");
  if (sols.size() > 1)
    for (unsigned int i=0; i<sols.size(); ++i)
      {
        d.add_ref(*sols[i], "sol["+Utilities::int_to_string(i)+"]");
      }
  else
    d.add_ref(*sols[0], "sol");
}

#endif
