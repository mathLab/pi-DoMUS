/*! \addtogroup equations
 *  @{
 */

#ifndef _hydrogels_three_fields_h_
#define _hydrogels_three_fields_h_

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
  class HydroGelThreeFields : public PDESystemInterface<dim,spacedim, HydroGelThreeFields<dim,spacedim,LAC>, LAC>
{
public:

  ~HydroGelThreeFields() {};

  HydroGelThreeFields();

  void declare_parameters (ParameterHandler &prm);
  void parse_parameters_call_back ();

  void set_matrix_couplings(std::vector<std::string> &couplings) const;

  template <typename EnergyType, typename ResidualType>
  void energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                              FEValuesCache<dim,spacedim> &scratch,
                              std::vector<EnergyType> &energies,
                              std::vector<std::vector<ResidualType> > &local_residuals,
                              bool compute_only_system_terms) const;


  void compute_system_operators(const DoFHandler<dim,spacedim> &,
                                const std::vector<shared_ptr<LATrilinos::BlockMatrix> >,
                                LinearOperator<LATrilinos::VectorType> &,
                                LinearOperator<LATrilinos::VectorType> &) const;



private:
  double T;
  double Omega;
  double G;
  double chi;
  double l0;

  double mu;
  double mu0;
  double l02;
  double l03;
  double l0_3;
  double l0_6;
  const double R=8.314;


  mutable shared_ptr<TrilinosWrappers::PreconditionJacobi> U_prec;
  mutable shared_ptr<TrilinosWrappers::PreconditionJacobi> p_prec;
  mutable shared_ptr<TrilinosWrappers::PreconditionJacobi> c_prec;

};

template <int dim, int spacedim, typename LAC>
  HydroGelThreeFields<dim,spacedim,LAC>::HydroGelThreeFields() :
  PDESystemInterface<dim,spacedim,HydroGelThreeFields<dim,spacedim,LAC>, LAC>("Free Swelling Three Fields",
										 dim+2,2,
										 "FESystem[FE_Q(2)^d-FE_Q(1)-FE_Q(2)]",
										 "u,u,u,p,c","1,0,0")
{}

template <int dim, int spacedim, typename LAC>
 void
  HydroGelThreeFields<dim,spacedim,LAC>::
  set_matrix_couplings(std::vector<std::string> &couplings) const
{
  couplings[0] = "1,1,0;1,0,1;0,1,1";
  couplings[1] = "1,0,0;0,1,0;0,0,1";
}

template <int dim, int spacedim, typename LAC>
  template <typename EnergyType, typename ResidualType>
  void
  HydroGelThreeFields<dim,spacedim,LAC>::
  energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
			 FEValuesCache<dim,spacedim> &fe_cache,
			 std::vector<EnergyType> &energies,
			 std::vector<std::vector<ResidualType> > &,
			 bool compute_only_system_terms) const
{
  EnergyType alpha = this->alpha;
  this->reinit(alpha, cell, fe_cache);

  const FEValuesExtractors::Vector displacement(0);
  const FEValuesExtractors::Scalar pressure(dim);
  const FEValuesExtractors::Scalar concentration(dim+1);

  auto &us = fe_cache.get_values("solution", "u", displacement, alpha);
  auto &Fs = fe_cache.get_deformation_gradients("solution", "Fu", displacement, alpha);

  auto &ps = fe_cache.get_values("solution", "p", pressure, alpha);
  
  auto &cs = fe_cache.get_values("solution", "c", concentration, alpha);

  const unsigned int n_q_points = us.size();

  auto &JxW = fe_cache.get_JxW_values();
  
  for (unsigned int q=0; q<n_q_points; ++q)
    {
      const Tensor<1, dim, EnergyType>  &u = us[q];
      const Tensor<2, dim, EnergyType>  &F = Fs[q];
      const Tensor<2, dim, EnergyType>   C = transpose(F)*F;
      const EnergyType &c = cs[q];
      const EnergyType &p = ps[q];

      const EnergyType I = trace(C);
      const EnergyType J = determinant(F);


      EnergyType psi = ( 0.5*G*l0_3*(l02*I - dim)

                     + (l0_3*R*T/Omega)*((Omega*l03*c)*std::log((Omega*l03*c)/(1.+Omega*l03*c))
                                         + chi*((Omega*l03*c)/(1.+Omega*l03*c)) )

                     - (mu0)*c - p*(J-l0_3-Omega*c)) ;

      energies[0] += psi*JxW[q];

      if (!compute_only_system_terms)
	energies[1] += 0.5*(u*u
			    +p*p
			    +c*c)*JxW[q];
    }

}

template <int dim, int spacedim, typename LAC>
  void HydroGelThreeFields<dim,spacedim,LAC>::declare_parameters (ParameterHandler &prm)
{
  PDESystemInterface<dim,spacedim, HydroGelThreeFields<dim,spacedim,LAC>, LAC>::declare_parameters(prm);
  this->add_parameter(prm, &T, "T", "298.0", Patterns::Double(0.0));
  this->add_parameter(prm, &Omega, "Omega", "1e-5", Patterns::Double(0.0));
  this->add_parameter(prm, &chi, "chi", "0.1", Patterns::Double(0.0));
  this->add_parameter(prm, &l0, "l0", "1.5", Patterns::Double(1.0));
  this->add_parameter(prm, &G, "G", "10e3", Patterns::Double(0.0));
  this->add_parameter(prm, &mu, "mu", "-1.8e-22", Patterns::Double(-1000.0));
}

template <int dim, int spacedim, typename LAC>
  void HydroGelThreeFields<dim,spacedim,LAC>::parse_parameters_call_back ()
{
  l02 = l0*l0;
  l03 = l02*l0;
  l0_3 = 1./l03;
  l0_6 = 1./(l03*l03);

  mu0 = R*T*(std::log((l03-1.)/l03) + l0_3 + chi*l0_6) + G*Omega/l0;
}


template <int dim, int spacedim, typename LAC>
  void
  HydroGelThreeFields<dim,spacedim,LAC>::
  compute_system_operators(const DoFHandler<dim,spacedim> &,
			   const std::vector<shared_ptr<LATrilinos::BlockMatrix> > matrices,
			   LinearOperator<LATrilinos::VectorType> & system_op,
			   LinearOperator<LATrilinos::VectorType> & prec_op) const
{

  /* std::vector<std::vector<bool> > constant_modes; */
  /* FEValuesExtractors::Vector velocity_components(0); */
  /* DoFTools::extract_constant_modes (dh, dh.get_fe().component_mask(velocity_components), */
  /*                                   constant_modes); */

  p_prec.reset (new TrilinosWrappers::PreconditionJacobi());
  c_prec.reset (new TrilinosWrappers::PreconditionJacobi());
  U_prec.reset (new TrilinosWrappers::PreconditionJacobi());

//  TrilinosWrappers::PreconditionAMG::AdditionalData Amg_data;
//  Amg_data.constant_modes = constant_modes;
//  Amg_data.elliptic = true;
//  Amg_data.higher_order_elements = true;
//  Amg_data.smoother_sweeps = 2;
//  Amg_data.aggregation_threshold = 0.02;
//

  U_prec->initialize (matrices[1]->block(0,0));
  p_prec->initialize (matrices[1]->block(1,1));
  c_prec->initialize (matrices[1]->block(2,2));


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


  static ReductionControl solver_control_pre(5000, 1e-4);
  static SolverCG<LATrilinos::VectorType::BlockType> solver_CG(solver_control_pre);

  auto P0_inv = inverse_operator( P0, solver_CG, *U_prec);
  auto P1_inv = inverse_operator( P1, solver_CG, *p_prec);
  auto P2_inv = inverse_operator( P2, solver_CG, *c_prec);

  auto P0_i = P0_inv;
  auto P1_i = P1_inv;
  auto P2_i = P2_inv;


//  auto P0_inv = linear_operator( matrix.block(0,0), *U_prec);
//  auto P1_inv = linear_operator( matrix.block(1,1), *p_prec);
//  auto P2_inv = linear_operator( matrix.block(2,2), *c_prec);


  const std::array<std::array<LinearOperator<LATrilinos::VectorType::BlockType>, 3 >, 3 > matrix_array = {{
      {{ A,   Bt   , Z02 }},
      {{ B,   Z11  , C   }},
      {{ Z20, D    , E   }}
    }
  };

  auto sys_op = block_operator<3, 3, LATrilinos::VectorType>(matrix_array);
  
  system_op  = block_operator<3, 3, LATrilinos::VectorType >(matrix_array);

    /* {{ */
    /*   {{ A,   Bt   , Z02 }}, */
    /*   {{ B,   Z11  , C   }}, */
    /*   {{ Z20, D    , E   }} */
    /*   }}); */

  const std::array<LinearOperator<LATrilinos::VectorType::BlockType>, 3 > diagonal_array = {{ P0_i, P1_i, P2_i }};

  // system_op = linear_operator<VEC, VEC>(matrix);

  auto diag_op = block_diagonal_operator<3,LATrilinos::VectorType>(diagonal_array);
  
  //  prec_op = block_back_substitution<3, LATrilinos::VectorType::BlockType>(matrix_array, diagonal_array);
  
  auto p_op = block_forward_substitution<>(
					   BlockLinearOperator<LATrilinos::VectorType>(matrix_array),
					   BlockLinearOperator<LATrilinos::VectorType>(diagonal_array));

  prec_op = p_op;
}



#endif
/*! @} */
