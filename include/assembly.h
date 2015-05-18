#ifndef _ASSEMBLY_
#define _ASSEMBLY_

// #include <deal.II/fe/fe_q.h>
// #include <deal.II/fe/fe_dgq.h>
// #include <deal.II/fe/fe_dgp.h>
// #include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
// #include <deal.II/fe/mapping_q.h>

using namespace dealii;

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

#endif
