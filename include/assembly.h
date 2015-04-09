#ifndef ASSEMBLY 
#define ASSEMBLY

using namespace dealii;

namespace Assembly
{
	namespace Scratch
	{
		template <int dim>
		struct StokesPreconditioner
		{
		StokesPreconditioner (	const FiniteElement<dim> &stokes_fe,
								const Quadrature<dim>    &stokes_quadrature,
								const Mapping<dim>       &mapping,
								const UpdateFlags         update_flags);

		StokesPreconditioner (const StokesPreconditioner &data);


		FEValues<dim>               stokes_fe_values;

		std::vector<Tensor<2,dim> > grad_phi_u;
		std::vector<double>         phi_p;
		};

		template <int dim>
		StokesPreconditioner<dim>::
		StokesPreconditioner (const FiniteElement<dim> &stokes_fe,
							const Quadrature<dim>    &stokes_quadrature,
							const Mapping<dim>       &mapping,
							const UpdateFlags         update_flags)
		:
		stokes_fe_values (mapping, stokes_fe, stokes_quadrature,
							update_flags),
		grad_phi_u (stokes_fe.dofs_per_cell),
		phi_p (stokes_fe.dofs_per_cell)
		{}

		template <int dim>
		StokesPreconditioner<dim>::
		StokesPreconditioner (const StokesPreconditioner &scratch)
		:
		stokes_fe_values (scratch.stokes_fe_values.get_mapping(),
							scratch.stokes_fe_values.get_fe(),
							scratch.stokes_fe_values.get_quadrature(),
							scratch.stokes_fe_values.get_update_flags()),
		grad_phi_u (scratch.grad_phi_u),
		phi_p (scratch.phi_p)
		{}

		template <int dim>
		struct StokesSystem : public StokesPreconditioner<dim>
		{
		StokesSystem (  const FiniteElement<dim> &stokes_fe,
						const Mapping<dim>       &mapping,
						const Quadrature<dim>    &stokes_quadrature,
						const UpdateFlags         stokes_update_flags
						// ,
						// const FiniteElement<dim> &temperature_fe,
						// const UpdateFlags         temperature_update_flags
						);

		StokesSystem (const StokesSystem<dim> &data);


		// FEValues<dim>                        temperature_fe_values;

		std::vector<Tensor<1,dim> >          phi_u;
		std::vector<SymmetricTensor<2,dim> > grads_phi_u;
		std::vector<double>                  div_phi_u;

		std::vector<double>                  old_temperature_values;
		};

		template <int dim>
		StokesSystem<dim>::
		StokesSystem (  const FiniteElement<dim> &stokes_fe,
						const Mapping<dim>       &mapping,
						const Quadrature<dim>    &stokes_quadrature,
						const UpdateFlags         stokes_update_flags)
		:
		StokesPreconditioner<dim> (stokes_fe, stokes_quadrature,
									mapping,
									stokes_update_flags),
		// temperature_fe_values (mapping, temperature_fe, stokes_quadrature,
								// temperature_update_flags),
		phi_u (stokes_fe.dofs_per_cell),
		grads_phi_u (stokes_fe.dofs_per_cell),
		div_phi_u (stokes_fe.dofs_per_cell)//,
		// old_temperature_values (stokes_quadrature.size()
		// )
		{}

		template <int dim>
		StokesSystem<dim>::
		StokesSystem (const StokesSystem<dim> &scratch)
		:
		StokesPreconditioner<dim> (scratch),
		// temperature_fe_values (scratch.temperature_fe_values.get_mapping(),
		// 						scratch.temperature_fe_values.get_fe(),
		// 						scratch.temperature_fe_values.get_quadrature(),
		// 						scratch.temperature_fe_values.get_update_flags()),
		phi_u (scratch.phi_u),
		grads_phi_u (scratch.grads_phi_u),
		div_phi_u (scratch.div_phi_u)
		// old_temperature_values (scratch.old_temperature_values)
		{}

	}


	namespace CopyData
	{
		template <int dim>
		struct StokesPreconditioner
		{
		StokesPreconditioner (const FiniteElement<dim> &stokes_fe);
		StokesPreconditioner (const StokesPreconditioner &data);

		FullMatrix<double>          local_matrix;
		std::vector<types::global_dof_index> local_dof_indices;
		};

		template <int dim>
		StokesPreconditioner<dim>::
		StokesPreconditioner (const FiniteElement<dim> &stokes_fe)
		:
		local_matrix (stokes_fe.dofs_per_cell,
						stokes_fe.dofs_per_cell),
		local_dof_indices (stokes_fe.dofs_per_cell)
		{}

		template <int dim>
		StokesPreconditioner<dim>::
		StokesPreconditioner (const StokesPreconditioner &data)
		:
		local_matrix (data.local_matrix),
		local_dof_indices (data.local_dof_indices)
		{}

		template <int dim>
		struct StokesSystem : public StokesPreconditioner<dim>
		{
		StokesSystem (const FiniteElement<dim> &stokes_fe);
		StokesSystem (const StokesSystem<dim> &data);

		Vector<double> local_rhs;
		};

		template <int dim>
		StokesSystem<dim>::
		StokesSystem (const FiniteElement<dim> &stokes_fe)
		:
		StokesPreconditioner<dim> (stokes_fe),
		local_rhs (stokes_fe.dofs_per_cell)
		{}

		template <int dim>
		StokesSystem<dim>::
		StokesSystem (const StokesSystem<dim> &data)
		:
		StokesPreconditioner<dim> (data),
		local_rhs (data.local_rhs)
		{}

	}
}

#endif
