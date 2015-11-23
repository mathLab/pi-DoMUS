/*
 *  Interface
 *
 *
 */

#ifndef _pidomus_interface_h
#define _pidomus_interface_h

#include "base_interface.h"

using namespace deal2lkit;
using namespace pidomus;

template<int dim, int spacedim, int n_components, class Implementation,  typename LAC=LATrilinos>
class Interface : public BaseInterface<dim,spacedim,n_components,LAC>
{


public:

  virtual ~Interface() {};

  Interface(const std::string &name="",
            const std::string &default_fe="FE_Q(1)",
            const std::string &default_component_names="u",
            const std::string &default_differential_components="") :
    BaseInterface<dim,spacedim,n_components,LAC>(name, default_fe, default_component_names,
                                                 default_differential_components) {};

  virtual void declare_parameters(ParameterHandler &prm)
  {
    BaseInterface<dim,spacedim,n_components,LAC>::declare_parameters(prm);
  }
  virtual void parse_parameters_call_back()
  {
    BaseInterface<dim,spacedim,n_components,LAC>::parse_parameters_call_back();
  }


  virtual void get_energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                                          FEValuesCache<dim,spacedim> &scratch,
                                          std::vector<Sdouble> &energies,
                                          std::vector<std::vector<double> > &local_residuals,
                                          bool compute_only_system_matrix) const
  {
    for (unsigned int i=0; i<energies.size(); ++i)
      energies[i] = 0.0;
    for (unsigned int i=0; i<local_residuals.size(); ++i)
      for (unsigned int j=0; j<local_residuals[0].size(); ++j)
        local_residuals[i][j] = 0.0;
    //    init_to_zero(energies);
    //    init_to_zero(local_residuals);
    static_cast<const Implementation *>(this)->set_energies_and_residuals(cell,
        scratch,
        energies,
        local_residuals,
        compute_only_system_matrix);
  }


  virtual void get_energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                                          FEValuesCache<dim,spacedim> &scratch,
                                          std::vector<SSdouble> &energies,
                                          std::vector<std::vector<Sdouble> > &local_residuals,
                                          bool compute_only_system_matrix) const
  {
    for (unsigned int i=0; i<energies.size(); ++i)
      energies[i] = 0.0;
    for (unsigned int i=0; i<local_residuals.size(); ++i)
      for (unsigned int j=0; j<local_residuals[0].size(); ++j)
        local_residuals[i][j] = 0.0;
    // init_to_zero(energies);
    // init_to_zero(local_residuals);
    static_cast<const Implementation *>(this)->set_energies_and_residuals(cell,
        scratch,
        energies,
        local_residuals,
        compute_only_system_matrix);
  }

};

#endif
