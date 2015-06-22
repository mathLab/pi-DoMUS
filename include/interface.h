#ifndef _interface_h_
#define _interface_h_

#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/trilinos_block_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/block_linear_operator.h>

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
                                      n_components, default_coupling, default_preconditioner_coupling)
  {};

  void add_fe_data(const unsigned int &dofs_per_cell,
                   const unsigned int &n_q_points,
                   SAKData &d) const;

  void add_solution(const TrilinosWrappers::MPI::BlockVector &sol,
                    SAKData &d) const;

  virtual void initialize_preconditioner_data(SAKData &d) const
  {
    Assert(false, ExcPureFunctionCalled ());
  }

  virtual void initialize_system_data(SAKData &d) const
  {
    Assert(false, ExcPureFunctionCalled ());
  }

  virtual void fill_preconditioner_data(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                                        Scratch &scratch,
                                        CopyPreconditioner &data) const
  {
    Assert(false, ExcPureFunctionCalled ());
  }

  virtual void fill_system_data(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                                Scratch   &scratch,
                                CopySystem  &data) const
  {
    Assert(false, ExcPureFunctionCalled ());
  }

  virtual void get_preconditioner_energy(const Scratch &, Sdouble &) const
  {
    Assert(false, ExcPureFunctionCalled ());
  }

  virtual void get_preconditioner_energy(const Scratch &, SSdouble &)  const
  {
    Assert(false, ExcPureFunctionCalled ());
  }

  virtual void get_system_energy(const Scratch &, Sdouble &) const
  {
    Assert(false, ExcPureFunctionCalled ());
  }

  virtual void get_system_energy(const Scratch &, SSdouble &)  const
  {
    Assert(false, ExcPureFunctionCalled ());
  }

  virtual void get_system_residual (const Scratch &scratch,
                                    const CopySystem &data,
                                    const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                                    std::vector<Sdouble> &local_residual) const
  {
    SSdouble energy;
    get_system_energy(scratch, energy);
    for (unsigned int i=0; i<local_residual.size(); ++i)
      {
        local_residual[i] = energy.dx(i);
      }
  }

  virtual void get_system_residual (const Scratch &scratch,
                                    const CopySystem &data,
                                    const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                                    std::vector<double> &local_residual) const
  {
    Sdouble energy;
    get_system_energy(scratch, energy);
    for (unsigned int i=0; i<local_residual.size(); ++i)
      {
        local_residual[i] = energy.dx(i);
      }
  }

  virtual void get_preconditioner_residual (const Scratch &scratch, std::vector<Sdouble> &local_residual) const
  {
    SSdouble energy;
    get_preconditioner_energy(scratch, energy);
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

};


template<int dim, int spacedim, int n_components>
void Interface<dim,spacedim,n_components>::add_fe_data(const unsigned int &dofs_per_cell,
                                                       const unsigned int &n_q_points,
                                                       SAKData &d) const
{
  d.add_copy(dofs_per_cell, "dofs_per_cell");
  d.add_copy(n_q_points, "n_q_points");
}

template<int dim, int spacedim, int n_components>
void Interface<dim,spacedim,n_components>::add_solution(const TrilinosWrappers::MPI::BlockVector &sol,
                                                        SAKData &d) const
{
  d.add_ref(sol, "sol");
}

#endif
