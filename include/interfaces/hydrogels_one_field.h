/*! \addtogroup equations
 *  @{
 */

#ifndef _hydrogels_one_field_h_
#define _hydrogels_one_field_h_

#include "pde_system_interface.h"
#include <deal2lkit/parsed_function.h>


#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/trilinos_block_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>

#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/block_linear_operator.h>
#include <deal.II/lac/packaged_operation.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>

#include "lac/lac_type.h"


template <int dim, int spacedim, typename LAC>
class HydroGelOneField : public PDESystemInterface<dim,spacedim, HydroGelOneField<dim,spacedim,LAC>, LAC>
{
public:

  ~HydroGelOneField() {};

  HydroGelOneField();

  void declare_parameters (ParameterHandler &prm);
  void parse_parameters_call_back ();

//  void set_matrix_couplings(std::vector<std::string> &couplings) const;

  template <typename EnergyType, typename ResidualType>
  void energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                              FEValuesCache<dim,spacedim> &scratch,
                              std::vector<EnergyType> &energies,
                              std::vector<std::vector<ResidualType> > &local_residuals,
                              bool compute_only_system_terms) const;


  /*void compute_system_operators(const DoFHandler<dim,spacedim> &,
                                const std::vector<shared_ptr<LATrilinos::BlockMatrix> >,
                                LinearOperator<LATrilinos::VectorType> &,
                                LinearOperator<LATrilinos::VectorType> &) const;
  */


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


/*  mutable shared_ptr<TrilinosWrappers::PreconditionAMG> U_prec;
  mutable shared_ptr<TrilinosWrappers::PreconditionJacobi> p_prec;
  mutable shared_ptr<TrilinosWrappers::PreconditionJacobi> c_prec;
*/

};

template <int dim, int spacedim, typename LAC>
HydroGelOneField<dim,spacedim,LAC>::HydroGelOneField() :
  PDESystemInterface<dim,spacedim,HydroGelOneField<dim,spacedim,LAC>, LAC>("Free Swelling One Field",
      dim,1,
      "FESystem[FE_Q(2)^d]",
      "u,u,u","1")
{}

/*
template <int dim, int spacedim, typename LAC>
void
HydroGelOneField<dim,spacedim,LAC>::
set_matrix_couplings(std::vector<std::string> &couplings) const
{
  couplings[0] = "1,1,0;1,0,1;0,1,1";
  couplings[1] = "0,0,0;0,1,0;0,0,0";
}
*/

template <int dim, int spacedim, typename LAC>
template <typename EnergyType, typename ResidualType>
void
HydroGelOneField<dim,spacedim,LAC>::
energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                       FEValuesCache<dim,spacedim> &fe_cache,
                       std::vector<EnergyType> &energies,
                       std::vector<std::vector<ResidualType> > &,
                       bool compute_only_system_terms) const
{
  EnergyType alpha = this->alpha;
  this->reinit(alpha, cell, fe_cache);

  const FEValuesExtractors::Vector displacement(0);

  auto &Fs = fe_cache.get_deformation_gradients("solution", "Fu", displacement, alpha);

  auto &JxW = fe_cache.get_JxW_values();

  for (unsigned int q=0; q<n_q_points; ++q)
    {
      const Tensor<2, dim, EnergyType>  &F = Fs[q];
      const Tensor<2, dim, EnergyType>   C = transpose(F)*F;

      const EnergyType I = trace(C);
      const EnergyType J = determinant(F);


      EnergyType psi = ( 0.5*G*l0_3*(l02*I - dim)

                         + (l0_3*R*T/Omega)*((J*l03-1.)*std::log((J*l03-1.)/(J*l03))
                                             + chi*((J*l03-1.)/(J*l03)) )

                         - (mu0)*(J*l03-1)/(Omega*l03)) ;

      energies[0] += psi*JxW[q];

      //if (!compute_only_system_terms)
      //  energies[1] += 0.5*(u*u)*JxW[q];
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

/*
template <int dim, int spacedim, typename LAC>
void
HydroGelOneField<dim,spacedim,LAC>::
compute_system_operators(const DoFHandler<dim,spacedim> &dh,
                         const std::vector<shared_ptr<LATrilinos::BlockMatrix> > matrices,
                         LinearOperator<LATrilinos::VectorType> &system_op,
                         LinearOperator<LATrilinos::VectorType> &prec_op) const
{

  std::vector<std::vector<bool> > constant_modes;
  FEValuesExtractors::Vector displacement(0);
  DoFTools::extract_constant_modes (dh, dh.get_fe().component_mask(displacement),
                                    constant_modes);

  p_prec.reset (new TrilinosWrappers::PreconditionJacobi());
  c_prec.reset (new TrilinosWrappers::PreconditionJacobi());
  U_prec.reset (new TrilinosWrappers::PreconditionAMG());

  TrilinosWrappers::PreconditionAMG::AdditionalData Amg_data;
  Amg_data.constant_modes = constant_modes;
  Amg_data.elliptic = true;
  Amg_data.higher_order_elements = true;
  Amg_data.smoother_sweeps = 2;
  Amg_data.aggregation_threshold = 0.02;


  U_prec->initialize (matrices[0]->block(0,0), Amg_data);
  p_prec->initialize (matrices[1]->block(1,1));
  c_prec->initialize (matrices[0]->block(2,2));


  // SYSTEM MATRIX:
  auto A   =   linear_operator< LATrilinos::VectorType::BlockType >( matrices[0]->block(0,0) );
  auto Bt  =   linear_operator< LATrilinos::VectorType::BlockType >( matrices[0]->block(0,1) );
  auto Z02 = 0*linear_operator< LATrilinos::VectorType::BlockType >( matrices[0]->block(0,2) );

  auto B   =   linear_operator< LATrilinos::VectorType::BlockType >( matrices[0]->block(1,0) );
  auto Z11 = 0*linear_operator< LATrilinos::VectorType::BlockType >( matrices[0]->block(1,1) );
  auto C   =   linear_operator< LATrilinos::VectorType::BlockType >( matrices[0]->block(1,2) );

  auto Z20 = 0*linear_operator< LATrilinos::VectorType::BlockType >( matrices[0]->block(2,0) );
  auto D   =   linear_operator< LATrilinos::VectorType::BlockType >( matrices[0]->block(2,1) );
  auto E   =   linear_operator< LATrilinos::VectorType::BlockType >( matrices[0]->block(2,2) );

  auto P0  =  linear_operator< LATrilinos::VectorType::BlockType >( matrices[0]->block(0,0));
  auto P1  =  linear_operator< LATrilinos::VectorType::BlockType >( matrices[1]->block(1,1));
  auto P2  =  linear_operator< LATrilinos::VectorType::BlockType >( matrices[0]->block(2,2));


  static ReductionControl solver_control_pre(5000, 1e-3);
  static SolverCG<LATrilinos::VectorType::BlockType> solver_CG(solver_control_pre);

  auto P0_inv = inverse_operator( P0, solver_CG, *U_prec);
  auto P1_inv = inverse_operator( P1, solver_CG, *p_prec);
  auto P2_inv = inverse_operator( P2, solver_CG, *c_prec);

  auto P0_i = P0_inv;
  auto P1_i = P1_inv;
  auto P2_i = P2_inv;


  const std::array<std::array<LinearOperator<LATrilinos::VectorType::BlockType>, 3 >, 3 > matrix_array = {{
      {{ A,   Bt   , Z02 }},
      {{ B,   Z11  , C   }},
      {{ Z20, D    , E   }}
    }
  };

  system_op  = block_operator<3, 3, LATrilinos::VectorType >(matrix_array);

  const std::array<LinearOperator<LATrilinos::VectorType::BlockType>, 3 > diagonal_array = {{ P0_i, P1_i, P2_i }};


  prec_op = block_diagonal_operator<3,LATrilinos::VectorType>(diagonal_array);

}
*/


#endif
/*! @} */
