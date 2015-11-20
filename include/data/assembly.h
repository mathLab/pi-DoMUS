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

  std::vector<types::global_dof_index>  local_dof_indices;
  std::vector<std::vector<double> >     double_residuals;
  std::vector<std::vector<Sdouble> >    sacado_residuals; //TODO REMOVE
  std::vector<FullMatrix<double> >      local_matrices;
};


CopyData<dim, spacedim>::
CopyData (const unsigned int &dofs_per_cell,
          const unsigned int &n_matrices)
  :
  local_dof_indices  (dofs_per_cell),
  double_residuals   (n_matrices, std::vector<double>(dofs_per_cell)),
  sacado_residuals   (n_matrices, std::vector<Sdouble>(dofs_per_cell)),
  local_matrices     (n_matrices,
                      FullMatrix<double>(dofs_per_cell,
                                         dofs_per_cell))
{}

CopyData<dim, spacedim>::
CopyData (const CopyData &data)
  :
  local_dof_indices  (data.local_dof_indices),
  double_residuals   (data.double_residuals),
  sacado_residuals   (data.sacado_residuals),
  local_matrices     (data.local_matrices)
{}


#endif
