/**
 * Assembly
 *
 * This namespace contains two sub namespaces: Scratch and CopyData.
 *
 * Goal: provide two structs data required in comunication process
 *       like WorkStream.
 */

#ifndef _pidomus_copy_data_h
#define _pidomus_copy_data_h

#include <deal.II/fe/fe_values.h>
#include "Sacado.hpp"
#include <deal2lkit/fe_values_cache.h>

using namespace dealii;
using namespace deal2lkit;

typedef Sacado::Fad::DFad<double> Sdouble;
typedef Sacado::Fad::DFad<Sdouble> SSdouble;

struct CopyData
{
  CopyData (const unsigned int &dofs_per_cell,,
            const unsigned int &n_matrices);
  CopyData (const CopyData &data);

  FullMatrix<double>                    local_matrix;
  std::vector<types::global_dof_index>  local_dof_indices;
  std::vector<Sdouble>                  sacado_residual;
  std::vector<double>                   double_residual;
  std::vector<std::vector<double> >     double_residuals;
  std::vector<std::vector<Sdouble> >    sacado_residuals;
  std::vector<FullMatrix<double> >      local_matrices;
};


CopyData<dim, spacedim>::
CopyData (const unsigned int &dofs_per_cell,
          const unsigned int &n_matrices)
  :
  local_matrix       (dofs_per_cell,
                      dofs_per_cell),
  local_dof_indices  (dofs_per_cell),
  sacado_residual    (dofs_per_cell),
  double_residual    (dofs_per_cell),
  double_residuals   (n_matrices),
  sacado_residuals   (n_matrices),
  local_matrices     (n_matrices, local_matrix)
{}

CopyData<dim, spacedim>::
CopyData (const CopyData &data)
  :
  local_matrix       (data.local_matrix),
  local_dof_indices  (data.local_dof_indices),
  sacado_residual    (data.sacado_residual),
  double_residual    (data.double_residual),
  double_residuals   (data.double_residuals),
  sacado_residuals   (data.sacado_residuals),
  local_matrices     (data.local_matrices)
{}


#endif
