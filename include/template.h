#ifndef _pidoums_template_h_
#define _pidoums_template_h_

#include "interface.h"

template <int dim, int spacedim>
class ProblemTemplate : public Interface<dim,spacedim,/*n_components*/ dim, ProblemTemplate<dim,spacedim> /*LAC*/ >
{
public:
  ~ProblemTemplate () {};
  ProblemTemplate ();

  void declare_parameters (ParameterHandler &prm);
  void parse_parameters_call_back ();

  // interface with the Interface :)

   void compute_system_operators(const DoFHandler<dim,spacedim> &,
                                        const std::vector<shared_ptr<typename LAC::BlockMatrix> >,
                                        LinearOperator<typename LAC::VectorType> &,
                                        LinearOperator<typename LAC::VectorType> &) const;


  // this function allows to define the update_flags.
  // by default they are
  //  update_quadrature_points |
  //  update_JxW_values |
  //  update_values |
  //  update_gradients
  //  UpdateFlags get_matrices_update_flags() const;

  //   UpdateFlags get_face_update_flags() const;
   
   shared_ptr<Mapping<dim,spacedim> > set_mapping () const;

   // set the number of matrices to be assembled
   unsigned int get_number_of_matrices() const;


   set_matrices_coupling ();

  template <typename EnergyType, typename ResidualType>
  void set_energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                                  FEValuesCache<dim,spacedim> &scratch,
                                  std::vector<EnergyType> &energies,
                                  std::vector<std::vector<ResidualType> > &local_residuals,
                                  bool compute_only_system_matrix) const;
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

      auto psi = (mu/2.)*(Ic-dim) - mu*lnJ + (lambda/2.)*(lnJ)*(lnJ);

      energies[0] += (psi)*JxW[q];

      ////////////////////////// residual formulation
      auto &u = us[q];
      for (unsigned int i=0; i<fev.dofs_per_cell(); ++i)
        {
          auto v = fev[displacement] .value(i,q); // test function

          local_residuals[0] -= 0.1*v*u; // non-conservative load

          // matrix[0] is assumed to be the system matrix
          // other matrices are either preconditioner or
          // auxiliary matrices needed to build it.
          //
          // if this function is called to evalute the
          // residual we do not need to assemble them
          // so we guard them
          if (!compute_only_system_matrix)
            local_residuals[1] += v*u;
        }
    }

}




#endif
