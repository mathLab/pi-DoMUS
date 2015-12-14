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
  void set_energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                                  FEValuesCache<dim,spacedim> &scratch,
                                  std::vector<EnergyType> &energies,
                                  std::vector<std::vector<ResidualType> > &local_residuals,
                                  bool compute_only_system_matrix) const;

  void compute_system_operators(const DoFHandler<dim,spacedim> &,
                                const std::vector<shared_ptr<typename LAC::BlockMatrix> >,
                                LinearOperator<typename LAC::VectorType> &,
                                LinearOperator<typename LAC::VectorType> &) const;

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

  // const Mapping<dim,spacedim> & get_mapping () const;

};




template <int dim, int spacedim>
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

          local_residuals[0] -= 0.1*v*u;

          // matrix[0] is assumed to be the system matrix other
          // matrices are either preconditioner or auxiliary matrices
          // needed to build it.
          //
          // if this function is called to evalute the system residual
          //  we do not need to assemble them so we guard them
          if (!compute_only_system_matrix)
            local_residuals[1] += v*u;
        }
    }

}




#endif
