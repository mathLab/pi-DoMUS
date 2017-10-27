#ifndef _pidoums_template_h_
#define _pidoums_template_h_

#include "pde_system_interface.h"

template <int dim, int spacedim, typename LAC>
class ProblemTemplate : public PDESystemInterface<dim,spacedim,ProblemTemplate<dim,spacedim,LAC> LAC>
{
public:
  ~ProblemTemplate () {};
  ProblemTemplate ();

  void declare_parameters (ParameterHandler &prm);
  void parse_parameters_call_back ();

  // interface with the PDESystemInterface :)
  template <typename EnergyType, typename ResidualType>
  void energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                              FEValuesCache<dim,spacedim> &scratch,
                              std::vector<EnergyType> &energies,
                              std::vector<std::vector<ResidualType> > &local_residuals,
                              bool compute_only_system_terms) const;


  // this function is needed only for the iterative solver
  void compute_system_operators(const std::vector<shared_ptr<LATrilinos::BlockMatrix> > &,
                                LinearOperator<LATrilinos::VectorType> &,
                                LinearOperator<LATrilinos::VectorType> &,
                                LinearOperator<LATrilinos::VectorType> &) const;



  ////////////// optimization part

  /* Coupling between the blocks of the finite elements in the system: */
  /*  0: No coupling */
  /*  1: Full coupling */
  /*  2: Coupling only on faces */
  /* If your matrices are fully coupled (as in this case), you can skip the */
  /* implementation because it is already set by default. */

  // void set_matrix_couplings (std::vector<std::string> > &couplings) const;


  /* this function allows to define the update_flags. */
  /* by default they are */
  /*  (update_quadrature_points | */
  /*  update_JxW_values | */
  /*  update_values | */
  /*  update_gradients) */

  //  UpdateFlags get_cell_update_flags() const;

  /* this function allows to set particular update_flags on the face */
  /* by default it returns */
  /* (update_values         | update_quadrature_points  | */
  /*  update_normal_vectors | update_JxW_values); */

  // UpdateFlags get_face_update_flags() const;

  /* this function defines Mapping used when Dirichlet boundary */
  /* conditions are applied, when the Initial solution is */
  /* interpolated, when the solution vector is stored in vtu format */
  /* and when the the error_from_exact is performed.  By default it */
  /* returns  StaticMappingQ1<dim,spacedim>::mapping; */

  // const Mapping<dim,spacedim> & get_default_mapping () const;

private:
// additional variables such as preconditioners
  mutable shared_ptr<TrilinosWrappers::PreconditionJacobi> preconditioner;

};

///////// constructor

template <int dim, int spacedim, typename LAC>
ProblemTemplate<dim,spacedim,LAC>::ProblemTemplate():
  PDESystemInterface<dim,spacedim,ProblemTemplate<dim,spacedim,LAC>,LAC>
  (\* section name in parameter file *\ "ProblemTemplate Parameters",
   \* n componenets *\ dim,
   \* n matrices *\ 2,
   \* finite element type *\ "FESystem[FE_Q(1)^d]",
   \* component names *\ "u,u,u",
   \* differential (1) and algebraic components (0) needed by IDA *\ "1")
{}

//////// additional parameters
template <int dim, int spacedim, typename LAC>
void ProblemTemplate<dim,spacedim,LAC>::declare_parameters (ParameterHandler &prm)
{
  PDESystemInterface<dim,spacedim, ProblemTemplate<dim,spacedim,LAC>,LAC >::declare_parameters(prm);

  this->add_parameter(....);
}

template <int dim, int spacedim, typename LAC>
void CompressibleNeoHookeanInterface<dim,spacedim,LAC>::parse_parameters_call_back ()
{
  // some operations with the just parsed parameters
  ... ;
}


///// definition of energies and residuals

template <int dim, int spacedim, typename LAC>
template <typename EnergyType, typename ResidualType>
void
ProblemTemplate<dim,spacedim>::
set_energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                           FEValuesCache<dim,spacedim> &fe_cache,
                           std::vector<EnergyType> &energies,
                           std::vector<std::vector<ResidualType> > &local_residuals,
                           bool compute_only_system_matrix) const
{

  const FEValuesExtractors::Vector displacement(0);

  ////////// conservative section
  //
  EnergyType et = 0; // dummy number to define the type of variables
  this->reinit (et, cell, fe_cache);
  auto &Fs = fe_cache.get_deformation_gradients("solution", "Fu", displacement, et);

  ////////// non-conservative section
  //
  ResidualType rt = 0;
  this->reinit (rt, cell, fe_cache);
  auto &us = fe_cache.get_values("solution", "u", displacement, rt);


  ////////// common variables
  //
  auto &fev = fe_cache.get_current_fe_values();
  const unsigned int n_q_points = us.size();
  auto &JxW = fe_cache.get_JxW_values();


  for (unsigned int q=0; q<n_q_points; ++q)
    {
      ///////////////////////// energetic contribution
      auto &F = Fs[q];
      auto C = transpose(F)*F;

      auto Ic = trace(C);
      auto J = determinant(F);
      auto lnJ = std::log (J);

      EnergyType psi = (mu/2.)*(Ic-dim) - mu*lnJ + (lambda/2.)*(lnJ)*(lnJ);

      energies[0] += (psi)*JxW[q];

      ////////////////////////// residual formulation
      auto &u = us[q];
      for (unsigned int i=0; i<local_residuals[0].size(); ++i)
        {
          auto v = fev[displacement] .value(i,q); // test function

          local_residuals[0][i] -= 0.1*v*u*JxW[q];

          // matrix[0] is assumed to be the system matrix other
          // matrices are either preconditioner or auxiliary matrices
          // needed to build it.
          //
          // if this function is called to evalute the system residual
          //  we do not need to assemble them so we guard them
          if (!compute_only_system_matrix)
            local_residuals[1][i] += v*u*JxW[q];
        }
    }

}


template <int dim, int spacedim, typename LAC>
void
ProblemTemplate<dim,spacedim,LAC>::compute_system_operators(const std::vector<shared_ptr<LATrilinos::BlockMatrix> > &matrices,
                                                            LinearOperator<LATrilinos::VectorType> &system_op,
                                                            LinearOperator<LATrilinos::VectorType> &prec_op,
                                                            LinearOperator<LATrilinos::VectorType> &) const
{

  preconditioner.reset  (new TrilinosWrappers::PreconditionJacobi());
  preconditioner->initialize(matrices[1]->block(0,0));
  auto P = linear_operator<LATrilinos::VectorType::BlockType>(matrices[1]->block(0,0));

  auto A  = linear_operator<LATrilinos::VectorType::BlockType>( matrices[0]->block(0,0) );

  static ReductionControl solver_control_pre(5000, 1e-4);
  static SolverCG<LATrilinos::VectorType::BlockType> solver_CG(solver_control_pre);
  auto P_inv     = inverse_operator( P, solver_CG, *preconditioner);

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
