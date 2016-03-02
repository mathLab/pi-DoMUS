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
#include <deal.II/lac/solver_bicgstab.h>



#include<deal.II/lac/schur_complement.h>

#include "lac/lac_type.h"

#include <time.h>



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


  void compute_system_operators(const std::vector<shared_ptr<LATrilinos::BlockMatrix> >,
                                LinearOperator<LATrilinos::VectorType> &,
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
  mutable shared_ptr<TrilinosWrappers::PreconditionSSOR> p_prec_ssor;

  unsigned int it_c_lumped;
  unsigned int it_s_approx;
  unsigned int it_s;

  bool elliptic;
  bool high_order_elements;
  unsigned int n_cycles;
  bool w_cycle;
  double agg;
  unsigned int smoother_sweeps;
  unsigned int smoother_overlap;
  bool output_details;
  std::string smoother_type;
  std::string coarse_type;

  ConditionalOStream pcout;

};

template <int dim, int spacedim, typename LAC>
HydroGelThreeFields<dim,spacedim,LAC>::HydroGelThreeFields() :
  PDESystemInterface<dim,spacedim,HydroGelThreeFields<dim,spacedim,LAC>, LAC>("Free Swelling Three Fields",
      dim+2,2,
      "FESystem[FE_Q(1)^d-FE_DGPMonomial(0)-FE_DGPMonomial(0)]",
      "u,u,u,c,p","1,0,0"),
  pcout(std::cout,
        (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
         == 0))
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

  auto &cs = fe_cache.get_values("solution", "c", concentration, alpha);

  const unsigned int n_q_points = ps.size();

  auto &JxW = fe_cache.get_JxW_values();

  for (unsigned int q=0; q<n_q_points; ++q)
    {
      const Tensor<2, dim, EnergyType>  &F = Fs[q];
      const Tensor<2, dim, EnergyType>   C = transpose(F)*F;
      const EnergyType &c = cs[q];
      const EnergyType &p = ps[q];


      const EnergyType I = trace(C);
      const EnergyType J = determinant(F);


      EnergyType psi = ( 0.5*G*l0_3*(l02*I - dim)

                         + (l0_3*R*T/Omega)*(
                           (Omega*l03*c)*std::log(
                             (Omega*l03*c)/(1.+Omega*l03*c)
                           )
                           + chi*(
                             (Omega*l03*c)/(1.+Omega*l03*c)
                           )
                         )

                         - (mu0)*c - p*(J-l0_3-Omega*c)
                       ) ;

      energies[0] += psi*JxW[q];

      if (!compute_only_system_terms)
        {
          EnergyType pp = 0.5*p*p ;
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
  this->add_parameter(prm, &it_c_lumped, "iteration c lumped", "10", Patterns::Integer(1));
  this->add_parameter(prm, &it_s_approx, "iteration s approx", "10", Patterns::Integer(1));
  this->add_parameter(prm, &it_s, "iteration s", "10", Patterns::Integer(1));
  this->add_parameter(prm, &agg, "aggregation", "0.8", Patterns::Double(0.0));

  this->add_parameter(prm, &elliptic, "elliptic", "true", Patterns::Bool());
  this->add_parameter(prm, &high_order_elements, "high_order_elements", "true", Patterns::Bool());
  this->add_parameter(prm, &w_cycle, "w_cycle", "true", Patterns::Bool());
  this->add_parameter(prm, &output_details, "output_details", "true", Patterns::Bool());

  this->add_parameter(prm, &n_cycles, "n_cycles", "2", Patterns::Integer(1));
  this->add_parameter(prm, &smoother_sweeps, "smoother_sweeps", "2", Patterns::Integer(1));
  this->add_parameter(prm, &smoother_overlap, "smoother_overlap", "1", Patterns::Integer(0));

  this->add_parameter(prm, &smoother_type, "smoother_type", "Chebyshev", Patterns::Selection("Aztec|IFPACK|Jacobi|ML symmetric Gauss-Seidel|symmetric Gauss-Seidel|ML Gauss-Seidel|Gauss-Seidel|block Gauss-Seidel|symmetric block Gauss-Seidel|Chebyshev|MLS|Hiptmair|Amesos-KLU|Amesos-Superlu|Amesos-UMFPACK|Amesos-Superludist|Amesos-MUMPS|user-defined|SuperLU|IFPACK-Chebyshev|self|do-nothing|IC|ICT|ILU|ILUT|Block Chebyshev|IFPACK-Block Chebyshev"));
  this->add_parameter(prm, &coarse_type, "coarse_type", "Amesos-KLU",Patterns::Selection("Aztec|IFPACK|Jacobi|ML symmetric Gauss-Seidel|symmetric Gauss-Seidel|ML Gauss-Seidel|Gauss-Seidel|block Gauss-Seidel|symmetric block Gauss-Seidel|Chebyshev|MLS|Hiptmair|Amesos-KLU|Amesos-Superlu|Amesos-UMFPACK|Amesos-Superludist|Amesos-MUMPS|user-defined|SuperLU|IFPACK-Chebyshev|self|do-nothing|IC|ICT|ILU|ILUT|Block Chebyshev|IFPACK-Block Chebyshev"));



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
compute_system_operators(const std::vector<shared_ptr<LATrilinos::BlockMatrix> > matrices,
                         LinearOperator<LATrilinos::VectorType> &system_op,
                         LinearOperator<LATrilinos::VectorType> &prec_op,
                         LinearOperator<LATrilinos::VectorType> &) const
{

  clock_t inizio = clock();
  clock_t totale = clock();


  double tempo;


  p_prec_ssor.reset (new TrilinosWrappers::PreconditionSSOR());
  c_prec_amg.reset (new TrilinosWrappers::PreconditionAMG());

//    if (U_prec == 0)
//      {
  U_prec.reset (new TrilinosWrappers::PreconditionAMG());

  TrilinosWrappers::PreconditionAMG::AdditionalData U_amg_data;
  U_amg_data.elliptic = elliptic;
  U_amg_data.higher_order_elements = high_order_elements;
  U_amg_data.n_cycles = n_cycles;
  U_amg_data.w_cycle = w_cycle;
  U_amg_data.smoother_sweeps = smoother_sweeps;
  U_amg_data.smoother_overlap = smoother_overlap;
  U_amg_data.output_details = output_details;
  U_amg_data.smoother_type = smoother_type.c_str();
  U_amg_data.coarse_type = coarse_type.c_str();
  U_amg_data.aggregation_threshold = agg;


  /* U_amg_data.elliptic = true; */
  /* U_amg_data.higher_order_elements = true; */
  /* U_amg_data.smoother_sweeps = 2; */
  /* U_amg_data.coarse_type = "Amesos-MUMPS"; */

  U_prec->initialize (matrices[0]->block(0,0), U_amg_data);

  tempo =  double(clock() - inizio)/(double)CLOCKS_PER_SEC;
  inizio = clock();
  pcout << "u amg " << tempo << " seconds" << std::endl;
  //      }


  TrilinosWrappers::PreconditionAMG::AdditionalData c_amg_data;
  c_amg_data.elliptic = true;
  c_amg_data.higher_order_elements = true;
  c_amg_data.smoother_sweeps = 1;
  c_amg_data.aggregation_threshold = 0.03;
  //  c_amg_data.coarse_type = "Amesos-MUMPS";


  c_prec_amg->initialize (matrices[0]->block(1,1), c_amg_data);

  tempo =  double(clock() - inizio)/(double)CLOCKS_PER_SEC;
  inizio = clock();

  pcout << "c amg " << tempo << " seconds" << std::endl;

  p_prec_ssor->initialize (matrices[1]->block(2,2));
  tempo =  double(clock() - inizio)/(double)CLOCKS_PER_SEC;
  inizio = clock();

  pcout << "p ssor " << tempo << " seconds" << std::endl;


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

  static auto C_lumped =   linear_operator< LATrilinos::VectorType::BlockType >( matrices[0]->block(1,1) );
  /*  static auto A_lumped =   linear_operator< LATrilinos::VectorType::BlockType >( matrices[0]->block(0,0) ); */

////////////////// C_lumped
  LATrilinos::VectorType::BlockType c_ones;
  C_lumped.reinit_domain_vector(c_ones, false);
  c_ones = 1.0;

  static  LATrilinos::VectorType::BlockType vec;
  vec.reinit(c_ones);
  vec=c_ones;

  C_lumped.vmult(vec, c_ones);

  C_lumped.vmult = [&vec] (LATrilinos::VectorType::BlockType &dst,
                           const LATrilinos::VectorType::BlockType &src)
  {
    dst = src;
    dst.scale(vec);
  };
  ///////////////

  const std::array<std::array<LinearOperator<LATrilinos::VectorType::BlockType>, 3 >, 3 > matrix_array = {{
      {{ A   , Z01 ,  Bt  }},
      {{ Z10 ,   C ,  Et  }},
      {{ B   ,   E ,  Z22 }}
    }
  };

  system_op  = block_operator<3, 3, LATrilinos::VectorType >(matrix_array);

  tempo =  double(clock() - inizio)/(double)CLOCKS_PER_SEC;
  inizio = clock();

  pcout << "system " << tempo << " seconds" << std::endl;


  static ReductionControl solver_control_pre(5000, 1e-6);
  static IterationNumberControl solver_control_c_lumped(it_c_lumped);

  static SolverCG<LATrilinos::VectorType::BlockType> solver_CG(solver_control_pre);
  /* //  static SolverBicgstab<LATrilinos::VectorType::BlockType> solver_c_lumped(solver_control_c_lumped); */
  /* //    static SolverFGMRES<LATrilinos::VectorType::BlockType> solver_c_lumped(solver_control_c_lumped); */
  static SolverCG<LATrilinos::VectorType::BlockType> solver_c_lumped(solver_control_c_lumped);


  //  auto A_inv = inverse_operator( PA, solver_CG, *U_prec);
  /* auto C_inv = inverse_operator( PE, solver_CG, *c_prec); */
  //  auto P_inv = inverse_operator( Pp, solver_CG, *p_prec_ssor);
  auto P_inv = linear_operator<LATrilinos::VectorType::BlockType>( matrices[1]->block(2,2), *p_prec_ssor);
  auto A_inv = linear_operator<LATrilinos::VectorType::BlockType>( matrices[0]->block(0,0), *U_prec);
  auto C_inv = linear_operator<LATrilinos::VectorType::BlockType>( matrices[0]->block(1,1), *c_prec_amg);
  //  auto P_inv = linear_operator<LATrilinos::VectorType::BlockType>( matrices[1]->block(2,2), *p_prec);


  auto P0_i = A_inv;
  auto P1_i = C_inv;
  auto P2_i = P_inv;


  ///////////////////////////////////////////////////////////////////////////
  auto L00 = identity_operator(A.reinit_range_vector);
  auto L11 = identity_operator(E.reinit_range_vector);
  auto L22 = identity_operator(Z22.reinit_range_vector);

  auto L02 = null_operator(Bt);
  auto L12 = null_operator(Et);
  LinearOperator<LATrilinos::VectorType::BlockType> L20 = null_operator(B) - B*A_inv;
  LinearOperator<LATrilinos::VectorType::BlockType> L21 = null_operator(E) - E*C_inv;




  const std::array<std::array<LinearOperator<LATrilinos::VectorType::BlockType>, 3 >, 3 > L_inv_array = {{
      {{ L00  ,  Z01  , L02 }},
      {{ Z10  ,  L11  , L12 }},
      {{ L20  ,  L21  , L22 }}
    }
  };

  LinearOperator<LATrilinos::VectorType> L_inv_op =   block_operator<3, 3, LATrilinos::VectorType >(L_inv_array);

  tempo =  double(clock() - inizio)/(double)CLOCKS_PER_SEC;
  inizio = clock();

  pcout << "L inv " << tempo << " seconds" << std::endl;


  auto U02 = null_operator(Bt) - A_inv*Bt;
  auto U12 = null_operator(Et) - C_inv*Et;
  auto U20 = null_operator(B);
  auto U21 = null_operator(C);

  const std::array<std::array<LinearOperator<LATrilinos::VectorType::BlockType>, 3 >, 3 > U_inv_array = {{
      {{ L00  ,  Z01  , U02 }},
      {{ Z10  ,  L11, U12 }},
      {{ U20  ,  U21  , L22          }}
    }
  };
  LinearOperator<LATrilinos::VectorType> U_inv_op =  block_operator<3, 3, LATrilinos::VectorType >(U_inv_array);


  tempo =  double(clock() - inizio)/(double)CLOCKS_PER_SEC;
  inizio = clock();

  pcout << "u inv " << tempo << " seconds" << std::endl;


  ///////////////////////////////////////////////////////////////

  auto S1 = schur_complement(A_inv,Bt,B,Z22);
  auto S2 = schur_complement(C_inv,Et,E,Z22);

  /* LinearOperator<LATrilinos::VectorType::BlockType> S1 = B*A_inv*Bt; */
  /* LinearOperator<LATrilinos::VectorType::BlockType> S2 = E*C_inv*Et; */
  /* S1 *= -1.0; */
  /* S2 *= -1.0; */

  /* auto S1_inv = inverse_operator(S2, solver_CG, *p_prec); */
  /* auto S2_inv = inverse_operator(S2, solver_CG, *p_prec); */

  auto C_lumped_inv = inverse_operator(C_lumped, solver_c_lumped, *p_prec_ssor);
  //   auto C_lumped_inv = inverse_operator(C_lumped, solver_c_lumped, *c_prec);
  /* auto A_lumped_inv = inverse_operator(A_lumped,solver_CG_it, *U_prec); */

  auto S2_approx = schur_complement(C_lumped_inv,Et,E,Z22);
  auto S_approx = S1 + S2_approx;
  //   auto S_approx =null_operator(E)- B*A_lumped_inv*Bt - E*C_lumped_inv*Et;

  auto S = S1 + S2;
  //  auto S_prec = P2_i;



  static IterationNumberControl schur_control_approx(it_s_approx);
  static IterationNumberControl schur_control(it_s);

  // static SolverBicgstab<LATrilinos::VectorType::BlockType> solver_schur_approx(schur_control_approx);

  static SolverFGMRES<LATrilinos::VectorType::BlockType> solver_schur_approx(schur_control_approx);

  static SolverFGMRES<LATrilinos::VectorType::BlockType> solver_schur(schur_control);
  // static SolverBicgstab<LATrilinos::VectorType::BlockType> solver_schur(schur_control);


  tempo =  double(clock() - inizio)/(double)CLOCKS_PER_SEC;
  inizio = clock();

  pcout << "definizione di schur " << tempo << " seconds" << std::endl;



  auto S_approx_inv = inverse_operator(S1, solver_schur_approx, *p_prec_ssor);
  tempo =  double(clock() - inizio)/(double)CLOCKS_PER_SEC;
  inizio = clock();

  pcout << "S approx inv " << tempo << " seconds" << std::endl;




  static  LinearOperator<LATrilinos::VectorType::BlockType> S_preconditioner;

  S_preconditioner       = S_approx_inv;




  auto S_inv = inverse_operator(S, solver_schur, S_preconditioner);
  tempo =  double(clock() - inizio)/(double)CLOCKS_PER_SEC;
  inizio = clock();

  pcout << "S  inv " << tempo << " seconds" << std::endl;



  const std::array<LinearOperator<LATrilinos::VectorType::BlockType>, 3 > diagonal_array = {{ P0_i, P1_i, S_inv }};


  LinearOperator<LATrilinos::VectorType> D_inv_op = block_diagonal_operator<3,LATrilinos::VectorType>(diagonal_array);

  tempo =  double(clock() - inizio)/(double)CLOCKS_PER_SEC;
  inizio = clock();

  pcout << "D inv " << tempo << " seconds" << std::endl;

  prec_op = U_inv_op*D_inv_op*L_inv_op;

  tempo =  double(clock() - inizio)/(double)CLOCKS_PER_SEC;
  inizio = clock();

  pcout << "prodotto " << tempo << " seconds" << std::endl;

  pcout << "TOTALE  " << double(clock() - totale)/(double)CLOCKS_PER_SEC << std::endl;
//prec_op = U_inv_op;
//  prec_op = L_inv_op;
// prec_op = D_inv_op;

}



#endif
/*! @} */
