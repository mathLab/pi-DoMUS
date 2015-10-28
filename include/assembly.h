/**
 * Assembly
 *
 * This namespace contains two sub namespaces: Scratch and CopyData.
 *
 * Goal: provide two structs data required in comunication process
 *       like WorkStream.
 */

#ifndef _ASSEMBLY_
#define _ASSEMBLY_

#include <deal.II/fe/fe_values.h>
#include "Sacado.hpp"
#include <deal2lkit/fe_values_cache.h>
#include <deal2lkit/any_data.h>

using namespace dealii;
using namespace deal2lkit;

typedef Sacado::Fad::DFad<double> Sdouble;
typedef Sacado::Fad::DFad<Sdouble> SSdouble;

namespace Assembly
{
  namespace CopyData
  {
    template <int dim, int spacedim>
    struct NFieldsPreconditioner
    {
      NFieldsPreconditioner (const FiniteElement<dim, spacedim> &fe);
      NFieldsPreconditioner (const NFieldsPreconditioner &data);

      FullMatrix<double>                    local_matrix;
      std::vector<types::global_dof_index>  local_dof_indices;
      std::vector<Sdouble>                  sacado_residual;
      std::vector<double>                   double_residual;
    };

    template <int dim, int spacedim>
    NFieldsPreconditioner<dim, spacedim>::
    NFieldsPreconditioner (const FiniteElement<dim, spacedim> &fe)
      :
      local_matrix (      fe.dofs_per_cell,
                          fe.dofs_per_cell),
      local_dof_indices ( fe.dofs_per_cell),
      sacado_residual (   fe.dofs_per_cell),
      double_residual (   fe.dofs_per_cell)
    {}

    template <int dim, int spacedim>
    NFieldsPreconditioner<dim, spacedim>::
    NFieldsPreconditioner (const NFieldsPreconditioner &data)
      :
      local_matrix (      data.local_matrix),
      local_dof_indices ( data.local_dof_indices),
      sacado_residual (   data.sacado_residual),
      double_residual (   data.double_residual)
    {}

    template <int dim, int spacedim>
    struct NFieldsSystem : public NFieldsPreconditioner<dim, spacedim>
    {
      NFieldsSystem (const FiniteElement<dim, spacedim> &fe);
      NFieldsSystem (const NFieldsSystem<dim, spacedim> &data);

      Vector<double> local_rhs;
    };

    template <int dim, int spacedim>
    NFieldsSystem<dim, spacedim>::
    NFieldsSystem (const FiniteElement<dim, spacedim> &fe)
      :
      NFieldsPreconditioner<dim, spacedim> (fe),
      local_rhs (fe.dofs_per_cell)
    {}

    template <int dim, int spacedim>
    NFieldsSystem<dim, spacedim>::
    NFieldsSystem (const NFieldsSystem<dim, spacedim> &data)
      :
      NFieldsPreconditioner<dim, spacedim> (data),
      local_rhs (data.local_rhs)
    {}

  }
}

#endif
