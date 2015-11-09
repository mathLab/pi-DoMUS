/**
 * Conservative Interface
 *
 * This class is a child of interface.h
 *
 * It is used to define a problem in the case whole the problem
 * could be stated in terms of energy and this energy can be derived
 * in ordert ot find a solution to the problem.
 *
 */

#ifndef _conservative_interface_h
#define _conservative_interface_h

#include "interfaces/interface.h"

using namespace deal2lkit;

template<int dim, int spacedim, int n_components, class Implementation, typename LAC=LATrilinos>
class ConservativeInterface : public Interface<dim,spacedim,n_components,LAC>
{

  typedef FEValuesCache<dim,spacedim> Scratch;
  typedef Assembly::CopyData::piDoMUSPreconditioner<dim,spacedim> CopyPreconditioner;
  typedef Assembly::CopyData::piDoMUSSystem<dim,spacedim> CopySystem;

public:

  virtual ~ConservativeInterface() {};
  virtual void declare_parameters(ParameterHandler &prm)
  {
    Interface<dim,spacedim,n_components,LAC>::declare_parameters(prm);
  }
  virtual void parse_parameters_call_back()
  {
    Interface<dim,spacedim,n_components,LAC>::parse_parameters_call_back();
  }


  ConservativeInterface(const std::string &name="",
                        const std::string &default_fe="FE_Q(1)",
                        const std::string &default_component_names="u",
                        const std::string &default_coupling="",
                        const std::string &default_preconditioner_coupling="",
                        const std::string &default_differential_components="") :
    Interface<dim,spacedim,n_components,LAC>(name, default_fe, default_component_names,
                                             default_coupling, default_preconditioner_coupling,
                                             default_differential_components) {};

  virtual void get_system_energy(const typename DoFHandler<dim,spacedim>::active_cell_iterator &c,
                                 Scratch &s,
                                 CopySystem &d,
                                 Sdouble &e) const
  {
    static_cast<const Implementation *>(this)->system_energy(c,s,d,e);
  }


  virtual void get_system_energy(const typename DoFHandler<dim,spacedim>::active_cell_iterator &c,
                                 Scratch &s,
                                 CopySystem &d,
                                 SSdouble &e) const
  {
    static_cast<const Implementation *>(this)->system_energy(c,s,d,e);
  }


  virtual void get_preconditioner_energy(const typename DoFHandler<dim,spacedim>::active_cell_iterator &c,
                                         Scratch &s,
                                         CopySystem &d,
                                         Sdouble &e) const
  {
    static_cast<const Implementation *>(this)->preconditioner_energy(c,s,d,e);
  }


  virtual void get_preconditioner_energy(const typename DoFHandler<dim,spacedim>::active_cell_iterator &c,
                                         Scratch &s,
                                         CopyPreconditioner &d,
                                         SSdouble &e) const
  {
    static_cast<const Implementation *>(this)->preconditioner_energy(c,s,d,e);
  }
};

#endif
