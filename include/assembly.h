#ifndef _ASSEMBLY_
#define _ASSEMBLY_

#include <deal.II/fe/fe_values.h>
#include "sak_data.h"
#include "Sacado.hpp"

using namespace dealii;
typedef Sacado::Fad::DFad<double> Sdouble;
typedef Sacado::Fad::DFad<Sdouble> SSdouble;

namespace Assembly
{
  namespace Scratch
  {
    template <int dim>
    struct NavierStokesPreconditioner
    {
      NavierStokesPreconditioner (  const FiniteElement<dim> &navier_stokes_fe,
                                    const Quadrature<dim>    &navier_stokes_quadrature,
                                    const Mapping<dim>       &mapping,
                                    const UpdateFlags         update_flags);

      NavierStokesPreconditioner (const NavierStokesPreconditioner &data);

      FEValues<dim>               navier_stokes_fe_values;

      std::vector<Tensor<2,dim> > grad_phi_u;
      std::vector<double>         phi_p;
    };

    template <int dim>
    NavierStokesPreconditioner<dim>::
    NavierStokesPreconditioner (  const FiniteElement<dim> &navier_stokes_fe,
                                  const Quadrature<dim>    &navier_stokes_quadrature,
                                  const Mapping<dim>       &mapping,
                                  const UpdateFlags         update_flags)
      :
      navier_stokes_fe_values   (mapping, navier_stokes_fe, navier_stokes_quadrature,
                                 update_flags),
      grad_phi_u          (navier_stokes_fe.dofs_per_cell),
      phi_p             (navier_stokes_fe.dofs_per_cell)
    {}

    template <int dim>
    NavierStokesPreconditioner<dim>::
    NavierStokesPreconditioner (const NavierStokesPreconditioner &scratch)
      :
      navier_stokes_fe_values ( scratch.navier_stokes_fe_values.get_mapping(),
                                scratch.navier_stokes_fe_values.get_fe(),
                                scratch.navier_stokes_fe_values.get_quadrature(),
                                scratch.navier_stokes_fe_values.get_update_flags()),
      grad_phi_u (scratch.grad_phi_u),
      phi_p (scratch.phi_p)
    {}

    template <int dim>
    struct NavierStokesSystem : public NavierStokesPreconditioner<dim>
    {
      NavierStokesSystem (    const FiniteElement<dim> &navier_stokes_fe,
                              const Mapping<dim>       &mapping,
                              const Quadrature<dim>    &navier_stokes_quadrature,
                              const UpdateFlags         navier_stokes_update_flags
                         );

      NavierStokesSystem (const NavierStokesSystem<dim> &data);

      std::vector<Tensor<1,dim> >          phi_u;
      std::vector<SymmetricTensor<2,dim> > grads_phi_u;
      std::vector<double>                  div_phi_u;

      std::vector<double>                  old_temperature_values;
    };

    template <int dim>
    NavierStokesSystem<dim>::
    NavierStokesSystem (    const FiniteElement<dim> &navier_stokes_fe,
                            const Mapping<dim>       &mapping,
                            const Quadrature<dim>    &navier_stokes_quadrature,
                            const UpdateFlags         navier_stokes_update_flags)
      :
      NavierStokesPreconditioner<dim> ( navier_stokes_fe, navier_stokes_quadrature,
                                        mapping, navier_stokes_update_flags),
      phi_u       (navier_stokes_fe.dofs_per_cell),
      grads_phi_u   (navier_stokes_fe.dofs_per_cell),
      div_phi_u     (navier_stokes_fe.dofs_per_cell)
    {}

    template <int dim>
    NavierStokesSystem<dim>::
    NavierStokesSystem (const NavierStokesSystem<dim> &scratch)
      :
      NavierStokesPreconditioner<dim> (scratch),
      phi_u     (scratch.phi_u),
      grads_phi_u (scratch.grads_phi_u),
      div_phi_u   (scratch.div_phi_u)
    {}

  }


  namespace CopyData
  {
    template <int dim>
    struct NavierStokesPreconditioner
    {
      NavierStokesPreconditioner (const FiniteElement<dim> &navier_stokes_fe);
      NavierStokesPreconditioner (const NavierStokesPreconditioner &data);

      FullMatrix<double>          local_matrix;
      std::vector<types::global_dof_index> local_dof_indices;
    };

    template <int dim>
    NavierStokesPreconditioner<dim>::
    NavierStokesPreconditioner (const FiniteElement<dim> &navier_stokes_fe)
      :
      local_matrix (navier_stokes_fe.dofs_per_cell,
                    navier_stokes_fe.dofs_per_cell),
      local_dof_indices (navier_stokes_fe.dofs_per_cell)
    {}

    template <int dim>
    NavierStokesPreconditioner<dim>::
    NavierStokesPreconditioner (const NavierStokesPreconditioner &data)
      :
      local_matrix (data.local_matrix),
      local_dof_indices (data.local_dof_indices)
    {}

    template <int dim>
    struct NavierStokesSystem : public NavierStokesPreconditioner<dim>
    {
      NavierStokesSystem (const FiniteElement<dim> &navier_stokes_fe);
      NavierStokesSystem (const NavierStokesSystem<dim> &data);

      Vector<double> local_rhs;
    };

    template <int dim>
    NavierStokesSystem<dim>::
    NavierStokesSystem (const FiniteElement<dim> &navier_stokes_fe)
      :
      NavierStokesPreconditioner<dim> (navier_stokes_fe),
      local_rhs (navier_stokes_fe.dofs_per_cell)
    {}

    template <int dim>
    NavierStokesSystem<dim>::
    NavierStokesSystem (const NavierStokesSystem<dim> &data)
      :
      NavierStokesPreconditioner<dim> (data),
      local_rhs (data.local_rhs)
    {}

  }
}


namespace Assembly
{
  namespace Scratch
  {
    template <int dim, int spacedim>
    struct NFields
    {
      NFields     (const SAKData &anydata,
                   const FiniteElement<dim, spacedim> &fe,
                   const Quadrature<dim>    &quadrature,
                   const Mapping<dim, spacedim>       &mapping,
                   const UpdateFlags         update_flags,
                   const Quadrature<dim-1>    &face_quadrature,
                   const UpdateFlags         face_update_flags);

      NFields     (const NFields &scratch);

      SAKData                     anydata;
      FEValues<dim, spacedim>     fe_values;
      FEFaceValues<dim, spacedim>     fe_face_values;
    };
    template <int dim, int spacedim>
    NFields<dim,spacedim>::NFields (const SAKData &data,
                                    const FiniteElement<dim, spacedim> &fe,
                                    const Quadrature<dim>    &quadrature,
                                    const Mapping<dim, spacedim>       &mapping,
                                    const UpdateFlags         update_flags,
                                    const Quadrature<dim-1>    &face_quadrature,
                                    const UpdateFlags         face_update_flags)
      :
      anydata         (data),
      fe_values       (mapping, fe, quadrature, update_flags),
      fe_face_values   (mapping, fe, face_quadrature, face_update_flags)
    {}

    template <int dim, int spacedim>
    NFields<dim,spacedim>::NFields (const NFields &scratch)
      :
      anydata   (scratch.anydata),
      fe_values ( scratch.fe_values.get_mapping(),
                  scratch.fe_values.get_fe(),
                  scratch.fe_values.get_quadrature(),
                  scratch.fe_values.get_update_flags()),
      fe_face_values ( scratch.fe_values.get_mapping(),
                       scratch.fe_values.get_fe(),
                       scratch.fe_face_values.get_quadrature(),
                       scratch.fe_face_values.get_update_flags())
    {}


  }



  namespace CopyData
  {
    template <int dim, int spacedim>
    struct NFieldsPreconditioner
    {
      NFieldsPreconditioner (const FiniteElement<dim, spacedim> &fe);
      NFieldsPreconditioner (const NFieldsPreconditioner &data);

      FullMatrix<double>          local_matrix;
      std::vector<types::global_dof_index> local_dof_indices;
      std::vector<Sdouble> sacado_residual;
      std::vector<double> double_residual;
    };

    template <int dim, int spacedim>
    NFieldsPreconditioner<dim, spacedim>::
    NFieldsPreconditioner (const FiniteElement<dim, spacedim> &fe)
      :
      local_matrix (fe.dofs_per_cell,
                    fe.dofs_per_cell),
      local_dof_indices (fe.dofs_per_cell),
      sacado_residual (fe.dofs_per_cell),
      double_residual (fe.dofs_per_cell)
    {}

    template <int dim, int spacedim>
    NFieldsPreconditioner<dim, spacedim>::
    NFieldsPreconditioner (const NFieldsPreconditioner &data)
      :
      local_matrix (data.local_matrix),
      local_dof_indices (data.local_dof_indices),
      sacado_residual (data.sacado_residual),
      double_residual (data.double_residual)
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
