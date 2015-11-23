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
#include <deal2lkit/fe_values_cache.h>

using namespace dealii;
using namespace deal2lkit;

struct CopyData
{
  CopyData (const unsigned int &dofs_per_cell,,
            const unsigned int &n_matrices);
  CopyData (const CopyData &data);

  std::vector<types::global_dof_index>  local_dof_indices;
  std::vector<double>                   double_residual;
  std::vector<FullMatrix<double> >      local_matrices;
};


CopyData<dim, spacedim>::
CopyData (const unsigned int &dofs_per_cell,
          const unsigned int &n_matrices)
  :
  local_dof_indices  (dofs_per_cell),
  local_residual     (dofs_per_cell),
  local_matrices     (n_matrices,
                      FullMatrix<double>(dofs_per_cell,
                                         dofs_per_cell))
{}

CopyData<dim, spacedim>::
CopyData (const CopyData &data)
  :
  local_dof_indices  (data.local_dof_indices),
  local_residual     (data.local_residual),
  local_matrices     (data.local_matrices)
{}


#endif
