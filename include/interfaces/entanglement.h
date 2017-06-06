#ifndef _pidoums_entanglement_h_
#define _pidoums_entanglement_h_

#include "pde_system_interface.h"
#include <deal2lkit/sacado_tools.h>



template <int dim, int spacedim, typename LAC=LADealII>
class EntanglementInterface : public PDESystemInterface<dim,spacedim, EntanglementInterface<dim,spacedim,LAC>, LAC>
{

public:
  ~EntanglementInterface () {};
  EntanglementInterface ();

  /* void declare_parameters (ParameterHandler &prm); */
  /* void parse_parameters_call_back (); */


  virtual UpdateFlags get_update_flags() const
  {
    return (update_values             |
            update_gradients          |
            update_quadrature_points  |
            // update_normal_vectors     |
            update_jacobians          |
            update_JxW_values);
  }



  // interface with the PDESystemInterface :)


  template <typename EnergyType, typename ResidualType>
  void energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                              FEValuesCache<dim,spacedim> &scratch,
                              std::vector<EnergyType> &energies,
                              std::vector<std::vector<ResidualType> > &local_residuals,
                              bool compute_only_system_terms) const;

  virtual void connect_to_signals() const
  {
    // first of all we get the struct Signals from pidomus
    auto &signals = this->get_signals();

    // we can connect calling .connect( and defining a lambda
    signals.fix_initial_conditions.connect(
      [this](typename LAC::VectorType &, typename LAC::VectorType &)
    {
      std::cout << "ciao mondo" << std::endl;
    }
    );

    // or we can define a lambda first
    auto l =  [this](typename LAC::VectorType &, typename LAC::VectorType &)
    {
      std::cout << "ho raffinato" << std::endl;
    };

    // and then attach the just defined lambda
    signals.fix_solutions_after_refinement.connect(l);


    // herebelow, we connect to all the begin_* signals available in piDoMUS
    auto &pcout = this->get_pcout();
    signals.begin_make_grid_fe.connect(
      [&]()
    {
      pcout << "#########  make_grid_fe"<<std::endl;
    });
    signals.begin_setup_dofs.connect(
      [&]()
    {
      pcout << "#########  setup_dofs"<<std::endl;
    });
    signals.begin_refine_mesh.connect(
      [&]()
    {
      pcout << "#########  refine_mesh"<<std::endl;
    });
    signals.begin_setup_jacobian.connect(
      [&]()
    {
      pcout << "#########  setup_jacobian"<<std::endl;
    });
    signals.begin_residual.connect(
      [&]()
    {
      pcout << "#########  residual"<<std::endl;
    });
    signals.begin_solve_jacobian_system.connect(
      [&]()
    {
      pcout << "#########  solve_jacobian_system"<<std::endl;
    });
    signals.begin_refine_and_transfer_solutions.connect(
      [&]()
    {
      pcout << "#########  refine_and_transfer_solutions"<<std::endl;
    });
    signals.begin_assemble_matrices.connect(
      [&]()
    {
      pcout << "#########  assemble_matrices"<<std::endl;
    });
    signals.begin_solver_should_restart.connect(
      [&]()
    {
      pcout << "#########  solver_should_restart"<<std::endl;
    });

  }




};

template <int dim, int spacedim, typename LAC>
EntanglementInterface<dim,spacedim, LAC>::
EntanglementInterface():
  PDESystemInterface<dim,spacedim,EntanglementInterface<dim,spacedim,LAC>, LAC >("Entanglement",
      3,1,
      "FESystem[FE_Q<2,3>(1)^3]",
      "u,u,u","1")
{}

namespace d2kinternal {
    template <int dim, int spacedim, typename Number>
    inline
    Number determinant (const DerivativeForm<1,dim,spacedim,Number> &DF) {
        const DerivativeForm<1,spacedim,dim,Number> DF_t = DF.transpose();
        Tensor<2,dim,Number> G; //First fundamental form
        for (unsigned int i=0; i<dim; ++i)
            for (unsigned int j=0; j<dim; ++j)
                G[i][j] = DF_t[i] * DF_t[j];

        return ( sqrt(determinant(G)) );
    }
}


template <int dim, int spacedim, typename LAC>
template <typename EnergyType, typename ResidualType>
void
EntanglementInterface<dim,spacedim,LAC>::
energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                       FEValuesCache<dim,spacedim> &fe_cache,
                       std::vector<EnergyType> &energies,
                       std::vector<std::vector<ResidualType> > &/*local_residuals*/,
                       bool ) const
{
  const FEValuesExtractors::Vector u(0);
  {
    EnergyType et = 0; // dummy number to define the type of variables
    double dut=0;
    this->reinit (et, cell, fe_cache);
    auto &eus = fe_cache.get_values("explicit_solution", "u", u, dut);
    auto &us = fe_cache.get_values("solution", "u", u, et);
    auto &grad_us = fe_cache.get_gradients("solution", "grad_u",u, et);

    const unsigned int n_q_points = us.size();
    auto &JxW = fe_cache.get_JxW_values();
    auto &jacs = fe_cache.get_current_fe_values().get_jacobians();
    auto fev = dynamic_cast<const FEValues<dim,spacedim> *>(&(fe_cache.get_current_fe_values()));
    Assert(fev != nullptr, ExcInternalError());

    auto &points = fev->get_quadrature_points();

    for (unsigned int q=0; q<n_q_points; ++q)
      {
        auto &uz = us[q][spacedim-1];
        auto &gradu = grad_us[q];
        auto &jac   = jacs[q];
        auto &p     = points[q];

        DerivativeForm<1,dim,spacedim,EnergyType> X;
        for(unsigned int a=0; a<dim; ++a)
            for(unsigned int i=0; i<spacedim; ++i) {
                X[i][a] = jac[i][a];
                for(unsigned int j=0; j<spacedim; ++j)
                    X[i][a] += jac[j][a]*gradu[i][j];
            }

        EnergyType z = p[spacedim-1]+uz;
        EnergyType psi = d2kinternal::determinant(X)/(z*z);

        double W = fev->get_quadrature().weight(q);

        energies[0] += (psi*W);

      
      }


  }

}


#endif
