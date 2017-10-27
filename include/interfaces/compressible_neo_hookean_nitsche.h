#ifndef _pidoums_compressible_neo_hookean_h_
#define _pidoums_compressible_neo_hookean_h_

#include "pde_system_interface.h"



template <int dim, int spacedim, typename LAC=LATrilinos>
class CompressibleNeoHookeanInterface : public PDESystemInterface<dim,spacedim, CompressibleNeoHookeanInterface<dim,spacedim,LAC>, LAC>
{

public:
  ~CompressibleNeoHookeanInterface () {}
  CompressibleNeoHookeanInterface ();

  void declare_parameters (ParameterHandler &prm);
  void parse_parameters_call_back ();


  virtual UpdateFlags get_face_update_flags() const
  {
    return (update_values             |
            update_gradients          | /* this is the new entry */
            update_quadrature_points  |
            update_normal_vectors     |
            update_JxW_values);
  }



  // interface with the PDESystemInterface :)


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

private:
  double E;
  double nu;
  double mu;
  double lambda;

  mutable shared_ptr<TrilinosWrappers::PreconditionJacobi> preconditioner;
  mutable shared_ptr<TrilinosWrappers::PreconditionAMG> U_prec;

  double gamma;
  mutable  ParsedMappedFunctions<spacedim> nitsche;

};

template <int dim, int spacedim, typename LAC>
CompressibleNeoHookeanInterface<dim,spacedim, LAC>::
CompressibleNeoHookeanInterface():
  PDESystemInterface<dim,spacedim,CompressibleNeoHookeanInterface<dim,spacedim,LAC>, LAC >("Compressible NeoHookean Parameters",
      dim,2,
      "FESystem[FE_Q(1)^d]",
      "u,u,u","1"),
  nitsche("Nitsche boundary conditions",
          this->n_components,
          this->get_component_names(),
          "" /* do nothing by default */
         )

{}


template <int dim, int spacedim, typename LAC>
void CompressibleNeoHookeanInterface<dim,spacedim,LAC>::declare_parameters (ParameterHandler &prm)
{
  PDESystemInterface<dim,spacedim, CompressibleNeoHookeanInterface<dim,spacedim,LAC>,LAC >::declare_parameters(prm);
  this->add_parameter(prm, &E, "Young's modulus", "10.0", Patterns::Double(0.0));
  this->add_parameter(prm, &nu, "Poisson's ratio", "0.3", Patterns::Double(0.0));
  this->add_parameter(prm, &gamma, "gamma", "0.1", Patterns::Double(0.0));

}

template <int dim, int spacedim, typename LAC>
void CompressibleNeoHookeanInterface<dim,spacedim,LAC>::parse_parameters_call_back ()
{
  mu = E/(2.0*(1.+nu));
  lambda = (E *nu)/((1.+nu)*(1.-2.*nu));
}



template <int dim, int spacedim, typename LAC>
template <typename EnergyType, typename ResidualType>
void
CompressibleNeoHookeanInterface<dim,spacedim,LAC>::
energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                       FEValuesCache<dim,spacedim> &fe_cache,
                       std::vector<EnergyType> &energies,
                       std::vector<std::vector<ResidualType> > &/*local_residuals*/,
                       bool compute_only_system_terms) const
{
  const FEValuesExtractors::Vector displacement(0);
  {
    ////////// conservative section
    //
    EnergyType et = 0; // dummy number to define the type of variables
    this->reinit (et, cell, fe_cache);
    auto &us = fe_cache.get_values("solution", "u", displacement, et);
    auto &Fs = fe_cache.get_deformation_gradients("solution", "Fu", displacement, et);

    const unsigned int n_q_points = us.size();
    auto &JxW = fe_cache.get_JxW_values();


    for (unsigned int q=0; q<n_q_points; ++q)
      {
        ///////////////////////// energetic contribution
        auto &u = us[q];
        auto &F = Fs[q];
        auto C = transpose(F)*F;

        auto Ic = trace(C);
        auto J = determinant(F);
        auto lnJ = std::log (J);

        EnergyType psi = (mu/2.)*(Ic-dim) - mu*lnJ + (lambda/2.)*(lnJ)*(lnJ);

        energies[0] += (
                         psi

                       )*JxW[q];

        if (!compute_only_system_terms)
          energies[1] += 0.5*u*u*JxW[q];


      }
  }

  {
    EnergyType dummy;
    const double h = cell->diameter();
    for (unsigned int face=0; face < GeometryInfo<dim>::faces_per_cell; ++face)
      {
        unsigned int face_id = cell->face(face)->boundary_id();
        if (cell->face(face)->at_boundary() && nitsche.acts_on_id(face_id))
          {
            this->reinit(dummy, cell, face, fe_cache);
            auto &us = fe_cache.get_values("solution", "u", displacement, dummy);
            auto &Fs = fe_cache.get_deformation_gradients("solution", "Fu", displacement, dummy);


            auto &fev = fe_cache.get_current_fe_values();
            auto &q_points = fe_cache.get_quadrature_points();
            auto &JxW = fe_cache.get_JxW_values();

            for (unsigned int q=0; q<q_points.size(); ++q)
              {
                auto &u = us[q];
                const Tensor<1,spacedim> n = fev.normal_vector(q);

                auto &F = Fs[q];

                EnergyType J = determinant(F);
                EnergyType lnJ = std::log (J);

                Tensor<2,dim,EnergyType> S = invert(transpose(F));
                S *= -(mu-lambda*lnJ);
                S += mu*F;

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
void
CompressibleNeoHookeanInterface<dim,spacedim,LAC>::compute_system_operators(const std::vector<shared_ptr<LATrilinos::BlockMatrix> > &matrices,
    LinearOperator<LATrilinos::VectorType> &system_op,
    LinearOperator<LATrilinos::VectorType> &prec_op,
    LinearOperator<LATrilinos::VectorType> &) const
{
  U_prec.reset(new TrilinosWrappers::PreconditionAMG());
  TrilinosWrappers::PreconditionAMG::AdditionalData U_amg_data;

  U_amg_data.elliptic = true;
  U_amg_data.higher_order_elements = true;
  U_amg_data.smoother_sweeps = 2;
  //U_amg_data.coarse_type = "Amesos-MUMPS";
  U_prec->initialize (matrices[0]->block(0,0), U_amg_data);


  preconditioner.reset  (new TrilinosWrappers::PreconditionJacobi());
  preconditioner->initialize(matrices[1]->block(0,0));
  auto P = linear_operator<LATrilinos::VectorType::BlockType>(matrices[1]->block(0,0));

  auto A  = linear_operator<LATrilinos::VectorType::BlockType>( matrices[0]->block(0,0) );

  static ReductionControl solver_control_pre(5000, 1e-4);
  static SolverCG<LATrilinos::VectorType::BlockType> solver_CG(solver_control_pre);
  /* auto P_inv     = inverse_operator( P, solver_CG, *preconditioner); */
  auto P_inv     = linear_operator<LATrilinos::VectorType::BlockType>(matrices[0]->block(0,0), *U_prec);

  auto P00 = P_inv;

  // ASSEMBLE THE PROBLEM:
  system_op  = block_operator<1, 1, LATrilinos::VectorType>({{
      {{ A }}
    }
  });

  prec_op = block_operator<1, 1, LATrilinos::VectorType>({{
      {{ P00}} ,
    }
  });
}

#endif
