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

  double mu0;
  double l02;
  double l03;
  double l0_3;
  double l0_6;
  const double R=8.314;


  mutable shared_ptr<TrilinosWrappers::PreconditionAMG> U_prec;
  mutable shared_ptr<TrilinosWrappers::PreconditionAMG> c_prec_amg;
  mutable shared_ptr<TrilinosWrappers::PreconditionJacobi> p_prec;
  mutable shared_ptr<TrilinosWrappers::PreconditionJacobi> c_prec;

};

template <int dim, int spacedim, typename LAC>
HydroGelThreeFields<dim,spacedim,LAC>::HydroGelThreeFields() :
  PDESystemInterface<dim,spacedim,HydroGelThreeFields<dim,spacedim,LAC>, LAC>("Free Swelling Three Fields",
      dim+2,2,
      "FESystem[FE_Q(2)^d-FE_Q(1)-FE_Q(2)]",
      "u,u,u,c,p","1,0,0")
{}

template <int dim, int spacedim, typename LAC>
void
HydroGelThreeFields<dim,spacedim,LAC>::
set_matrix_couplings(std::vector<std::string> &couplings) const
{
  couplings[0] = "1,0,1;0,1,1;1,1,0";
  couplings[1] = "0,0,0;0,0,0;0,0,1";
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
  const FEValuesExtractors::Scalar concentration(dim);
  const FEValuesExtractors::Scalar pressure(dim+1);

  auto &Fs = fe_cache.get_deformation_gradients("solution", "Fu", displacement, alpha);

  auto &ps = fe_cache.get_values("solution", "p", pressure, alpha);
  auto &grad_ps = fe_cache.get_gradients("solution", "gradp", pressure, alpha);

  auto &cs = fe_cache.get_values("solution", "c", concentration, alpha);

  const unsigned int n_q_points = ps.size();

  auto &JxW = fe_cache.get_JxW_values();

  for (unsigned int q=0; q<n_q_points; ++q)
    {
      const Tensor<2, dim, EnergyType>  &F = Fs[q];
      const Tensor<2, dim, EnergyType>   C = transpose(F)*F;
      const EnergyType &c = cs[q];
      const EnergyType &p = ps[q];
      const Tensor<1,dim,EnergyType> &grad_p = grad_ps[q];

      const EnergyType I = trace(C);
      const EnergyType J = determinant(F);


      EnergyType psi = ( 0.5*G*l0_3*(l02*I - dim)

                         + (l0_3*R*T/Omega)*((Omega*l03*c)*std::log((Omega*l03*c)/(1.+Omega*l03*c))
                                             + chi*((Omega*l03*c)/(1.+Omega*l03*c)) )

                         - (mu0)*c - p*(J-l0_3-Omega*c)) ;

      energies[0] += psi*JxW[q];

      if (!compute_only_system_terms)
        {
          EnergyType pp = p*p;
          energies[1] += pp*JxW[q];
        }
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
                         LinearOperator<LATrilinos::VectorType> &system_op,
                         LinearOperator<LATrilinos::VectorType> &prec_op) const
{


  p_prec.reset (new TrilinosWrappers::PreconditionJacobi());
  c_prec.reset (new TrilinosWrappers::PreconditionJacobi());
  U_prec.reset (new TrilinosWrappers::PreconditionAMG());
  c_prec_amg.reset (new TrilinosWrappers::PreconditionAMG());

  TrilinosWrappers::PreconditionAMG::AdditionalData Amg_data;
  Amg_data.elliptic = true;
  Amg_data.higher_order_elements = true;
  Amg_data.smoother_sweeps = 2;
  Amg_data.aggregation_threshold = 1;
  TrilinosWrappers::PreconditionAMG::AdditionalData c_amg_data;
  c_amg_data.elliptic = true;
  c_amg_data.higher_order_elements = true;
  c_amg_data.smoother_sweeps = 2;
  c_amg_data.aggregation_threshold = 2;


  U_prec->initialize (matrices[0]->block(0,0), Amg_data);
  p_prec->initialize (matrices[1]->block(2,2));
  c_prec->initialize (matrices[0]->block(1,1));
  c_prec_amg->initialize (matrices[0]->block(1,1), c_amg_data);



  // SYSTEM MATRIX:
  auto A   =   linear_operator< LATrilinos::VectorType::BlockType >( matrices[0]->block(0,0) );
  auto Z01 = 0*linear_operator< LATrilinos::VectorType::BlockType >( matrices[0]->block(0,1) );
  auto Bt  =   linear_operator< LATrilinos::VectorType::BlockType >( matrices[0]->block(0,2) );

  auto Z10 = 0*linear_operator< LATrilinos::VectorType::BlockType >( matrices[0]->block(1,0) );
  auto C   =   linear_operator< LATrilinos::VectorType::BlockType >( matrices[0]->block(1,1) );
  auto Et   =  linear_operator< LATrilinos::VectorType::BlockType >( matrices[0]->block(1,2) );

  auto B   =   linear_operator< LATrilinos::VectorType::BlockType >( matrices[0]->block(2,0) );
  auto E   =   linear_operator< LATrilinos::VectorType::BlockType >( matrices[0]->block(2,1) );
  auto Z22 = 0*linear_operator< LATrilinos::VectorType::BlockType >( matrices[0]->block(2,2) );

  auto PA  =   linear_operator< LATrilinos::VectorType::BlockType >( matrices[0]->block(0,0));
  auto PE  =   linear_operator< LATrilinos::VectorType::BlockType >( matrices[0]->block(1,1));
  auto Pp  =   linear_operator< LATrilinos::VectorType::BlockType >( matrices[1]->block(2,2));


  const std::array<std::array<LinearOperator<LATrilinos::VectorType::BlockType>, 3 >, 3 > matrix_array = {{
      {{ A   , Z01 ,  Bt  }},
      {{ Z10 ,   C ,  Et  }},
      {{ B   ,   E ,  Z22 }}
    }
  };

  system_op  = block_operator<3, 3, LATrilinos::VectorType >(matrix_array);





  //  static ReductionControl solver_control_pre(5000, 1e-9);
  static IterationNumberControl solver_control_pre(1);

  static SolverCG<LATrilinos::VectorType::BlockType> solver_CG(solver_control_pre);

  auto A_inv = inverse_operator( PA, solver_CG, *U_prec);
  auto C_inv = inverse_operator( PE, solver_CG, *c_prec_amg);
  auto P_inv = inverse_operator( Pp, solver_CG, *p_prec);
  /* auto A_inv = linear_operator<LATrilinos::VectorType::BlockType>( matrices[0]->block(0,0), *U_prec); */
  /* auto E_inv = linear_operator<LATrilinos::VectorType::BlockType>( matrices[0]->block(1,1), *c_prec); */
  /* auto P_inv = linear_operator<LATrilinos::VectorType::BlockType>( matrices[1]->block(2,2), *p_prec); */


  auto P0_i = A_inv;
  auto P1_i = C_inv;
  auto P2_i = P_inv;





  auto L00 = identity_operator(A.reinit_range_vector);
  auto L11 = identity_operator(E.reinit_range_vector);
  auto L22 = identity_operator(Z22.reinit_range_vector);

  auto L02 = null_operator(Bt);
  auto L12 = null_operator(Et);
  LinearOperator<LATrilinos::VectorType::BlockType> L20 = -1.0*(B*A_inv);
  LinearOperator<LATrilinos::VectorType::BlockType> L21 = -1.0*(E*C_inv);



  const std::array<std::array<LinearOperator<LATrilinos::VectorType::BlockType>, 3 >, 3 > L_inv_array = {{
      {{ L00  ,  Z01  , L02 }},
      {{ Z10  ,  L11  , L12 }},
      {{ L20  ,  L21  , L22 }}
    }
  };

  LinearOperator<LATrilinos::VectorType> L_inv_op =   block_operator<3, 3, LATrilinos::VectorType >(L_inv_array);


  auto U02 = -1.0*(A_inv*Bt);
  auto U12 = -1.0*(C_inv*Et);
  auto U20 = 0*B;
  auto U21 = 0*C;

  const std::array<std::array<LinearOperator<LATrilinos::VectorType::BlockType>, 3 >, 3 > U_inv_array = {{
      {{ L00  ,  Z01  , U02 }},
      {{ Z10  ,  L11, U12 }},
      {{ U20  ,  U21  , L22          }}
    }
  };
  LinearOperator<LATrilinos::VectorType> U_inv_op =  block_operator<3, 3, LATrilinos::VectorType >(U_inv_array);



  const std::array<LinearOperator<LATrilinos::VectorType::BlockType>, 3 > diagonal_array = {{ P0_i, P1_i, P2_i }};


  LinearOperator<LATrilinos::VectorType> D_inv_op = block_diagonal_operator<3,LATrilinos::VectorType>(diagonal_array);

  prec_op = U_inv_op*D_inv_op*L_inv_op;
//prec_op = U_inv_op;
//  prec_op = L_inv_op;
// prec_op = D_inv_op;


}



#endif
/*! @} */
