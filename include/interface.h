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

template<int dim, int spacedim, class Implementation,  typename LAC=LATrilinos>
class Interface : public BaseInterface<dim,spacedim,LAC>
{


public:

  virtual ~Interface() {};

  Interface(const unsigned int &n_components,
            const unsigned int &n_matrices,
            const std::string &name="",
            const std::string &default_fe="FE_Q(1)",
            const std::string &default_component_names="u",
            const std::string &default_differential_components="") :
    BaseInterface<dim,spacedim,LAC>(n_components,n_matrices, name,
                                    default_fe, default_component_names,
                                    default_differential_components) {};

  virtual void declare_parameters(ParameterHandler &prm)
  {
    BaseInterface<dim,spacedim,LAC>::declare_parameters(prm);
  }

  virtual void parse_parameters_call_back()
  {
    BaseInterface<dim,spacedim,LAC>::parse_parameters_call_back();
  }


  virtual void assemble_energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                                               FEValuesCache<dim,spacedim> &scratch,
                                               std::vector<Sdouble> &energies,
                                               std::vector<std::vector<double> > &local_residuals,
                                               bool compute_only_system_terms) const
  {
    static_cast<const Implementation *>(this)->energies_and_residuals(cell,
        scratch,
        energies,
        local_residuals,
        compute_only_system_terms);

  }


  virtual void assemble_energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                                               FEValuesCache<dim,spacedim> &scratch,
                                               std::vector<SSdouble> &energies,
                                               std::vector<std::vector<Sdouble> > &local_residuals,
                                               bool compute_only_system_terms) const
  {
    static_cast<const Implementation *>(this)->energies_and_residuals(cell,
        scratch,
        energies,
        local_residuals,
        compute_only_system_terms);

  }

};

#endif
