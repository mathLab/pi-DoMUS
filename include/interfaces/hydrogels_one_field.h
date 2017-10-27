/*! \addtogroup equations
 *  @{
 */

#ifndef _hydrogels_one_field_h_
#define _hydrogels_one_field_h_

#include "pde_system_interface.h"
#include <deal2lkit/parsed_function.h>
#include <deal2lkit/sacado_tools.h>

#include <deal.II/grid/grid_tools.h>

template <int dim, int spacedim, typename LAC>
class HydroGelOneField : public PDESystemInterface<dim,spacedim, HydroGelOneField<dim,spacedim,LAC>, LAC>
{
public:

  ~HydroGelOneField() {};

  HydroGelOneField();

  void declare_parameters (ParameterHandler &prm);
  void parse_parameters_call_back ();


  template <typename EnergyType, typename ResidualType>
  void energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                              FEValuesCache<dim,spacedim> &scratch,
                              std::vector<EnergyType> &energies,
                              std::vector<std::vector<ResidualType> > &local_residuals,
                              bool compute_only_system_terms) const;


  void compute_system_operators(const std::vector<shared_ptr<LATrilinos::BlockMatrix> > &,
                                LinearOperator<LATrilinos::VectorType> &,
                                LinearOperator<LATrilinos::VectorType> &,
                                LinearOperator<LATrilinos::VectorType> &) const;


  virtual void connect_to_signals() const
  {
    auto &signals = this->get_signals();
    if (this->wrinkling)
      {
        auto &pcout = this->get_pcout();
        pcout << "applying random distortion to grid" <<std::endl;
        signals.postprocess_newly_created_triangulation.connect(
          [&](Triangulation<dim,spacedim> *tria)
        {
          GridTools::distort_random(factor,*tria,false);
        });
      }

  }

private:
  double T;
  double Omega;
  double G;
  double chi;
  double l0;

  double mu0;
  double l02;
  double l03;
  double l0_3;
  double l0_6;
  const double R=8.314;


  double factor;
  bool wrinkling;



  mutable shared_ptr<TrilinosWrappers::PreconditionJacobi> preconditioner;

};

template <int dim, int spacedim, typename LAC>
HydroGelOneField<dim,spacedim,LAC>::HydroGelOneField() :
  PDESystemInterface<dim,spacedim,HydroGelOneField<dim,spacedim,LAC>, LAC>("Free Swelling One Field",
      dim,2,
      "FESystem[FE_Q(2)^d]",
      "u,u,u","1")
{}


template <int dim, int spacedim, typename LAC>
template <typename EnergyType, typename ResidualType>
void
HydroGelOneField<dim,spacedim,LAC>::
energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                       FEValuesCache<dim,spacedim> &fe_cache,
                       std::vector<EnergyType> &energies,
                       std::vector<std::vector<ResidualType> > &residuals,
                       bool compute_only_system_terms) const
{
  EnergyType alpha = 0;

  this->reinit(alpha, cell, fe_cache);

  const FEValuesExtractors::Vector displacement(0);

  auto &us = fe_cache.get_values("solution", "u", displacement, alpha);
  auto &Fs = fe_cache.get_deformation_gradients("solution", "Fu", displacement, alpha);
  auto &fev = fe_cache.get_current_fe_values();
  auto &JxW = fe_cache.get_JxW_values();

  const unsigned int n_q_points = JxW.size();

  std::vector<Tensor<2,dim,ResidualType> > Fs_res = SacadoTools::val(Fs);

  for (unsigned int q=0; q<n_q_points; ++q)
    {
      auto &u = us[q];
      const Tensor<2, dim, EnergyType>  &F = Fs[q];
      const Tensor<2, dim, EnergyType>   C = transpose(F)*F;

      const EnergyType I = trace(C);
      const EnergyType J = determinant(F);


      EnergyType psi = ( 0.5*G*l0_3*(l02*I - dim)

                         + (l0_3*R*T/Omega)*((J*l03-1.)*std::log((J*l03-1.)/(J*l03))
                                             + chi*((J*l03-1.)/(J*l03)) )

                         - (mu0)*(J*l03-1.)/(Omega*l03)) ;

      energies[0] += psi*JxW[q];

      auto &F_res = Fs_res[q];
      const Tensor<2,dim,ResidualType> F_star = J.val()*transpose(invert(F_res));
      for (unsigned int i=0; i<residuals[0].size(); ++i)
        {
          auto grad_v = fev[displacement].gradient(i,q);
          residuals[0][i] += (mu0/Omega)*(this->get_current_time())*SacadoTools::scalar_product(F_star,grad_v)*JxW[q];
        }

      if (!compute_only_system_terms)
        energies[1] += 0.5*(u*u)*JxW[q];
    }

}

template <int dim, int spacedim, typename LAC>
void HydroGelOneField<dim,spacedim,LAC>::declare_parameters (ParameterHandler &prm)
{
  PDESystemInterface<dim,spacedim, HydroGelOneField<dim,spacedim,LAC>, LAC>::declare_parameters(prm);
  this->add_parameter(prm, &T, "T", "298.0", Patterns::Double(0.0));
  this->add_parameter(prm, &Omega, "Omega", "1e-5", Patterns::Double(0.0));
  this->add_parameter(prm, &chi, "chi", "0.1", Patterns::Double(0.0));
  this->add_parameter(prm, &l0, "l0", "1.5", Patterns::Double(1.0));
  this->add_parameter(prm, &G, "G", "10e3", Patterns::Double(0.0));
  this->add_parameter(prm, &factor, "distortion factor", "1e-4", Patterns::Double(0.0));
  this->add_parameter(prm, &wrinkling, "distort triangulation", "false", Patterns::Bool());
}

template <int dim, int spacedim, typename LAC>
void HydroGelOneField<dim,spacedim,LAC>::parse_parameters_call_back ()
{
  l02 = l0*l0;
  l03 = l02*l0;
  l0_3 = 1./l03;
  l0_6 = 1./(l03*l03);

  mu0 = R*T*(std::log((l03-1.)/l03) + l0_3 + chi*l0_6) + G*Omega/l0;
}

template <int dim, int spacedim, typename LAC>
void
HydroGelOneField<dim,spacedim,LAC>::compute_system_operators(const std::vector<shared_ptr<LATrilinos::BlockMatrix> > &matrices,
    LinearOperator<LATrilinos::VectorType> &system_op,
    LinearOperator<LATrilinos::VectorType> &prec_op,
    LinearOperator<LATrilinos::VectorType> &) const
{
  preconditioner.reset  (new TrilinosWrappers::PreconditionJacobi());
  preconditioner->initialize(matrices[0]->block(0,0));

  auto P = linear_operator<LATrilinos::VectorType::BlockType>(matrices[0]->block(0,0));

  auto A  = linear_operator<LATrilinos::VectorType::BlockType>( matrices[0]->block(0,0) );

  static ReductionControl solver_control_pre(5000, 1e-8);
  static SolverCG<LATrilinos::VectorType::BlockType> solver_CG(solver_control_pre);
  auto P_inv     = inverse_operator( P, solver_CG, *preconditioner);

  auto P00 = P_inv;

  // ASSEMBLE THE PROBLEM:
  system_op  = block_operator<1, 1, LATrilinos::VectorType>({{
      {{ A }}
    }
  });

  prec_op = block_operator<1, 1, LATrilinos::VectorType>({{
      {{ P00 }} ,
    }
  });
}


#endif
/*! @} */
