/*! \addtogroup equations
 *  @{
 */

#ifndef _hydrogels_three_fields_h_
#define _hydrogels_three_fields_h_

#include "pde_system_interface.h"
#include <deal2lkit/parsed_function.h>
#include <deal2lkit/parsed_preconditioner/amg.h>
#include <deal2lkit/parsed_mapped_functions.h>

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


  virtual UpdateFlags get_face_update_flags() const
  {
    return (update_values             |
            update_gradients          | /* this is the new entry */
            update_quadrature_points  |
            update_normal_vectors     |
            update_JxW_values);
  }


  void declare_parameters (ParameterHandler &prm);
  void parse_parameters_call_back ();

  void set_matrix_couplings(std::vector<std::string> &couplings) const;

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
    auto &pcout = this->get_pcout();
    if (this->wrinkling)
      {
        pcout << "applying random distortion to grid" <<std::endl;
        signals.postprocess_newly_created_triangulation.connect(
          [&](Triangulation<dim,spacedim> *tria)
        {
          GridTools::distort_random(factor,*tria,false);
        });
      }
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

  mutable ParsedAMGPreconditioner U_prec;
  mutable ParsedAMGPreconditioner c_prec_amg;

  mutable shared_ptr<TrilinosWrappers::PreconditionSSOR> p_prec_ssor;

  unsigned int it_c_lumped;
  unsigned int it_s_approx;
  unsigned int it_s;

  double gamma;
  mutable  ParsedMappedFunctions<spacedim> nitsche;


};

template <int dim, int spacedim, typename LAC>
HydroGelThreeFields<dim,spacedim,LAC>::HydroGelThreeFields() :
  PDESystemInterface<dim,spacedim,HydroGelThreeFields<dim,spacedim,LAC>, LAC>("Free Swelling Three Fields",
      dim+2,2,
      "FESystem[FE_Q(1)^d-FE_DGPMonomial(0)-FE_DGPMonomial(0)]",
      "u,u,u,c,p","1,0,0"),
  U_prec("AMG for U"),
  c_prec_amg("AMG for c"),
  nitsche("Nitsche boundary conditions",
          this->n_components,
          this->get_component_names(),
          "" /* do nothing by default */
         )

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
  EnergyType alpha = 0;
  this->reinit(alpha, cell, fe_cache);

  const FEValuesExtractors::Vector displacement(0);
  const FEValuesExtractors::Scalar concentration(dim);
  const FEValuesExtractors::Scalar pressure(dim+1);
  {
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

  /// (quasi) automatic nitsche bcs
  {
    EnergyType dummy;
    const double h = cell->diameter();

    for (unsigned int face=0; face < GeometryInfo<dim>::faces_per_cell; ++face)
      {
        unsigned int face_id = cell->face(face)->boundary_id();
        if (cell->face(face)->at_boundary() && nitsche.acts_on_id(face_id))
          {
            this->reinit(dummy, cell, face, fe_cache);
            auto &ps = fe_cache.get_values("solution", "p", pressure, dummy);
            auto &us = fe_cache.get_values("solution", "u", displacement, dummy);
            auto &Fs = fe_cache.get_deformation_gradients("solution", "Fu", displacement, dummy);


            auto &fev = fe_cache.get_current_fe_values();
            auto &q_points = fe_cache.get_quadrature_points();
            auto &JxW = fe_cache.get_JxW_values();

            for (unsigned int q=0; q<q_points.size(); ++q)
              {
                auto &u = us[q];
                auto &p = ps[q];
                const Tensor<1,spacedim> n = fev.normal_vector(q);

                auto &F = Fs[q];
                Tensor<2,dim,EnergyType> C = transpose(F)*F;
                Tensor<2,dim,EnergyType> F_inv=invert(F);

                EnergyType Ic = trace(C);
                EnergyType J = determinant(F);
                EnergyType lnJ = std::log (J);


                Tensor<2,dim,EnergyType> S = invert(transpose(F));
                S *= -p*J;
                S += G/l0*F;

                // update time for nitsche_bcs
                nitsche.set_time(this->get_current_time());

                // get mapped function acting on this face_id
                Vector<double> func(this->n_components);
                nitsche.get_mapped_function(face_id)->vector_value(q_points[q], func);

                Tensor<1,spacedim> u0;

                for (unsigned int c=0; c<spacedim; ++c)
                  u0[c] = func[c];

                energies[0] +=(
                                (S*n)*(u-u0)

                                + (1.0/(2.0*gamma*h))*(u-u0)*(u-u0)
                              )*JxW[q];


              }// end loop over quadrature points

            break;

          } // endif face->at_boundary

      }// end loop over faces
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

  this->add_parameter(prm, &gamma, "Gamma", "0.001", Patterns::Double(0));
  this->add_parameter(prm, &factor, "distortion factor", "1e-4", Patterns::Double(0.0));
  this->add_parameter(prm, &wrinkling, "distort triangulation", "false", Patterns::Bool());

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
compute_system_operators(const std::vector<shared_ptr<LATrilinos::BlockMatrix> > &matrices,
                         LinearOperator<LATrilinos::VectorType> &system_op,
                         LinearOperator<LATrilinos::VectorType> &prec_op,
                         LinearOperator<LATrilinos::VectorType> &) const
{

  auto &pcout = this->get_pcout();
  clock_t inizio = clock();


  double tempo;

  //  auto &fe = this->pfe;

  p_prec_ssor.reset (new TrilinosWrappers::PreconditionSSOR());

  U_prec.initialize_preconditioner(matrices[0]->block(0,0));

  tempo =  double(clock() - inizio)/(double)CLOCKS_PER_SEC;
  inizio = clock();
  pcout << "u amg " << tempo << " seconds" << std::endl;
  //      }


  c_prec_amg.initialize_preconditioner (matrices[0]->block(1,1));

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
  /* LATrilinos::VectorType::BlockType c_ones; */
  /* C_lumped.reinit_domain_vector(c_ones, false); */
  /* c_ones = 1.0; */

  /* static  LATrilinos::VectorType::BlockType vec; */
  /* vec.reinit(c_ones); */
  /* vec=c_ones; */

  /* C_lumped.vmult(vec, c_ones); */

  /* C_lumped.vmult = [&vec] (LATrilinos::VectorType::BlockType &dst, */
  /*                          const LATrilinos::VectorType::BlockType &src) */
  /* { */
  /*   dst = src; */
  /*   dst.scale(vec); */
  /* }; */
  /* /////////////// */

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
  auto A_inv = linear_operator<LATrilinos::VectorType::BlockType>( matrices[0]->block(0,0), U_prec);
  auto C_inv = linear_operator<LATrilinos::VectorType::BlockType>( matrices[0]->block(1,1), c_prec_amg);
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

  (void)tempo;

  pcout << "prodotto " << tempo << " seconds" << std::endl;

  //  pcout << "TOTALE  " << double(clock() - totale)/(double)CLOCKS_PER_SEC << std::endl;
//prec_op = U_inv_op;
//  prec_op = L_inv_op;
// prec_op = D_inv_op;

}



#endif
/*! @} */
