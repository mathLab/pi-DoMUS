#include "pidomus.h"
#include "pidomus_macros.h"

using namespace dealii;
using namespace deal2lkit;

// This file contains the implementation of the functions
// called for applying Dirichlet and Neumann BCs and forcing
// terms specified **in the parameter file**.


template <int dim, int spacedim, typename LAC>
void
piDoMUS<dim,spacedim,LAC>::
apply_neumann_bcs (
  const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
  FEValuesCache<dim,spacedim> &scratch,
  std::vector<double> &local_residual) const
{

  double dummy = 0.0;

  for (unsigned int face=0; face < GeometryInfo<dim>::faces_per_cell; ++face)
    {
      unsigned int face_id = cell->face(face)->boundary_id();
      if (cell->face(face)->at_boundary() && neumann_bcs.acts_on_id(face_id))
        {
          interface.reinit(dummy, cell, face, scratch);

          auto &fev = scratch.get_current_fe_values();
          auto &q_points = scratch.get_quadrature_points();
          auto &JxW = scratch.get_JxW_values();

          for (unsigned int q=0; q<q_points.size(); ++q)
            {
              Vector<double> T(interface.n_components);
              neumann_bcs.get_mapped_function(face_id)->vector_value(q_points[q], T);

              for (unsigned int i=0; i<local_residual.size(); ++i)
                for (unsigned int c=0; c<interface.n_components; ++c)
                  local_residual[i] -= T[c]*fev.shape_value_component(i,q,c)*JxW[q];

            }// end loop over quadrature points

          break;

        } // endif face->at_boundary

    }// end loop over faces

}// end function definition



template <int dim, int spacedim, typename LAC>
void
piDoMUS<dim,spacedim,LAC>::
apply_forcing_terms (const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                     FEValuesCache<dim,spacedim> &scratch,
                     std::vector<double> &local_residual) const
{
  unsigned cell_id = cell->material_id();
  if (forcing_terms.acts_on_id(cell_id))
    {
      double dummy = 0.0;
      interface.reinit(dummy, cell, scratch);

      auto &fev = scratch.get_current_fe_values();
      auto &q_points = scratch.get_quadrature_points();
      auto &JxW = scratch.get_JxW_values();
      for (unsigned int q=0; q<q_points.size(); ++q)
        for (unsigned int i=0; i<local_residual.size(); ++i)
          for (unsigned int c=0; c<interface.n_components; ++c)
            {
              double B = forcing_terms.get_mapped_function(cell_id)->value(q_points[q],c);
              local_residual[i] -= B*fev.shape_value_component(i,q,c)*JxW[q];
            }
    }
}


template <int dim, int spacedim, typename LAC>
void
piDoMUS<dim,spacedim,LAC>::
apply_dirichlet_bcs (const DoFHandler<dim,spacedim> &dof_handler,
                     const ParsedDirichletBCs<dim,spacedim> &bc,
                     ConstraintMatrix &constraints) const
{
  try
    {
      bc.interpolate_boundary_values(interface.get_bc_mapping(),dof_handler,constraints);
    }
  catch (...)
    {
      AssertThrow(!we_are_parallel,
                  ExcMessage("You called VectorTools::project_boundary_values(), which is not \n"
                             "currently supported on deal.II in parallel settings.\n"
                             "Feel free to submit a patch :)"));
      const QGauss<dim-1> quad(fe->degree+1);
      bc.project_boundary_values(interface.get_bc_mapping(),dof_handler,quad,constraints);
    }
  unsigned int codim = spacedim - dim;
  if (codim == 0)
    bc.compute_nonzero_normal_flux_constraints(dof_handler,interface.get_bc_mapping(),constraints);
}



#define INSTANTIATE(dim,spacedim,LAC) \
  template class piDoMUS<dim,spacedim,LAC>;


PIDOMUS_INSTANTIATE(INSTANTIATE)

