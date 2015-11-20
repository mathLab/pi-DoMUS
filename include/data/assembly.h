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
    struct piDoMUSPreconditioner
    {
      piDoMUSPreconditioner (const FiniteElement<dim, spacedim> &fe,
                             const unsigned int &n_aux);
      piDoMUSPreconditioner (const piDoMUSPreconditioner &data);

      FullMatrix<double>                    local_matrix;
      std::vector<types::global_dof_index>  local_dof_indices;
      std::vector<Sdouble>                  sacado_residual;
      std::vector<double>                   double_residual;
      std::vector<std::vector<double> >     double_residuals;
      std::vector<std::vector<Sdouble> >    sacado_residuals;
      std::vector<FullMatrix<double> >      local_matrices;
    };

    template <int dim, int spacedim>
    piDoMUSPreconditioner<dim, spacedim>::
    piDoMUSPreconditioner (const FiniteElement<dim, spacedim> &fe,
                           const unsigned int &n_aux)
      :
      local_matrix       (fe.dofs_per_cell,
                          fe.dofs_per_cell),
      local_dof_indices  (fe.dofs_per_cell),
      sacado_residual    (fe.dofs_per_cell),
      double_residual    (fe.dofs_per_cell),
      double_residuals   (n_aux, std::vector<double>(fe.dofs_per_cell)),
      sacado_residuals   (n_aux, std::vector<Sdouble>(fe.dofs_per_cell)),
      local_matrices     (n_aux, local_matrix)
    {}

    template <int dim, int spacedim>
    piDoMUSPreconditioner<dim, spacedim>::
    piDoMUSPreconditioner (const piDoMUSPreconditioner &data)
      :
      local_matrix       (data.local_matrix),
      local_dof_indices  (data.local_dof_indices),
      sacado_residual    (data.sacado_residual),
      double_residual    (data.double_residual),
      double_residuals   (data.double_residuals),
      sacado_residuals   (data.sacado_residuals),
      local_matrices     (data.local_matrices)
    {}

    template <int dim, int spacedim>
    struct piDoMUSSystem : public piDoMUSPreconditioner<dim, spacedim>
    {
      piDoMUSSystem (const FiniteElement<dim, spacedim> &fe,
                     const unsigned int &n_aux);
      piDoMUSSystem (const piDoMUSSystem<dim, spacedim> &data);

      Vector<double> local_rhs;
    };

    template <int dim, int spacedim>
    piDoMUSSystem<dim, spacedim>::
    piDoMUSSystem (const FiniteElement<dim, spacedim> &fe,
                   const unsigned int &n_aux)
      :
      piDoMUSPreconditioner<dim, spacedim> (fe, n_aux),
      local_rhs (fe.dofs_per_cell)
    {}

    template <int dim, int spacedim>
    piDoMUSSystem<dim, spacedim>::
    piDoMUSSystem (const piDoMUSSystem<dim, spacedim> &data)
      :
      piDoMUSPreconditioner<dim, spacedim> (data),
      local_rhs (data.local_rhs)
    {}

  }
}

#endif
