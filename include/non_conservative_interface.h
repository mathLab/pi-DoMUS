#include "interface.h"

template<int dim, int spacedim, int n_components, class Implementation>
class NonConservativeInterface : public Interface<dim,spacedim,n_components>
{

  typedef Assembly::Scratch::NFields<dim,spacedim> Scratch;
  typedef Assembly::CopyData::NFieldsPreconditioner<dim,spacedim> CopyPreconditioner;
  typedef Assembly::CopyData::NFieldsSystem<dim,spacedim> CopySystem;
public:

  NonConservativeInterface(const std::string &name="",
                           const std::string &default_fe="FE_Q(1)",
                           const std::string &default_component_names="u",
                           const std::string &default_coupling="",
                           const std::string &default_preconditioner_coupling="") :
    Interface<dim,spacedim,n_components>(name, default_fe, default_component_names,
                                         default_coupling, default_preconditioner_coupling) {};

  virtual void get_system_residual (const Scratch &scratch,
                                    const CopySystem &data,
                                    const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                                    std::vector<double> &local_residual) const
  {
    static_cast<const Implementation *>(this)->system_residual(scratch, data, cell, local_residual);
  }

  virtual void get_system_residual (const Scratch &scratch,
                                    const CopySystem &data,
                                    const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                                    std::vector<Sdouble> &local_residual) const
  {
    static_cast<const Implementation *>(this)->system_residual(scratch, data, cell, local_residual);
  }

  virtual void get_preconditioner_residual(const Scratch &d, std::vector<Sdouble> &local_residual) const
  {
    static_cast<const Implementation *>(this)->preconditioner_residual(d, local_residual);
  }
};

