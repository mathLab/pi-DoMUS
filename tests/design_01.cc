//-----------------------------------------------------------
//
//    Copyright (C) 2014 by the deal.II authors
//
//    This file is subject to LGPL and may not be distributed
//    without copyright and license information. Please refer
//    to the file deal.II/doc/license.html for the  text  and
//    further information on this license.
//
//-----------------------------------------------------------

#include "tests.h"
#include <deal2lkit/parameter_acceptor.h>
#include "Sacado.hpp"
#include <deal2lkit/dof_utilities.h>
#include <deal2lkit/parsed_finite_element.h>

class Scratch
{
public:
  double  data;
};

template<int dim, int spacedim>
class Interface : public ParsedFiniteElement<dim,spacedim>
{

public:
  Interface(const std::string &name="",
            const std::string &default_fe="FE_Q(1)",
            const std::string &default_component_names="u",
            const unsigned int n_components=1,
            const std::string &default_coupling="",
            const std::string &default_preconditioner_coupling="") :
    ParsedFiniteElement<dim,spacedim>(name, default_fe, default_component_names,
                                      n_components, default_coupling, default_preconditioner_coupling) {};

  virtual void get_sacado_energy(const Scratch &, Sdouble &) const
  {
    Assert(false, ExcPureFunctionCalled ());
  };

  virtual void get_sacado_energy(const Scratch &, SSdouble &)  const
  {
    Assert(false, ExcPureFunctionCalled ());
  };

  virtual void get_sacado_residual(const Scratch &d, std::vector<Sdouble> &local_residual) const
  {
    SSdouble energy;
    get_sacado_energy(d, energy);
    for (unsigned int i=0; i<local_residual.size(); ++i)
      {
        local_residual[i] = energy.dx(i);
      }
  };

};

template<int dim, int spacedim, class Implementation>
class ConservativeInterface : public Interface<dim,spacedim>
{

public:

  ConservativeInterface(const std::string &name="",
                        const std::string &default_fe="FE_Q(1)",
                        const std::string &default_component_names="u",
                        const unsigned int n_components=1,
                        const std::string &default_coupling="",
                        const std::string &default_preconditioner_coupling="") :
    Interface<dim,spacedim>(name, default_fe, default_component_names,
                            n_components, default_coupling, default_preconditioner_coupling) {};

  virtual void get_sacado_energy(const Scratch &d, Sdouble &energy) const
  {
    static_cast<const Implementation *>(this)->get_energy(d, energy);
  }


  virtual void get_sacado_energy(const Scratch &d, SSdouble &energy) const
  {
    static_cast<const Implementation *>(this)->get_energy(d, energy);
  }
};


template<int dim, int spacedim, class Implementation>
class NonConservativeInterface : public Interface<dim,spacedim>
{

public:

  NonConservativeInterface(const std::string &name="",
                           const std::string &default_fe="FE_Q(1)",
                           const std::string &default_component_names="u",
                           const unsigned int n_components=1,
                           const std::string &default_coupling="",
                           const std::string &default_preconditioner_coupling="") :
    Interface<dim,spacedim>(name, default_fe, default_component_names,
                            n_components, default_coupling, default_preconditioner_coupling) {};

  virtual void get_sacado_residual(const Scratch &d, std::vector<Sdouble> &local_residual) const
  {
    static_cast<const Implementation *>(this)->get_sacado_residual(d, local_residual);
  }
};



template<int dim>
class StokesInterface : public ConservativeInterface<dim,dim, StokesInterface<dim> >
{
public:
  StokesInterface() :
    ConservativeInterface<dim,dim,StokesInterface<dim> >("Stokes Inteface",
                                                         "FESystem[FE_Q(2)^d-FE_Q(1)]",
                                                         "u,u,p", 3, "1,1; 1,0", "1,0; 0,1") {};
  template<typename Number>
  void get_energy(const Scratch &d, Number &energy) const
  {
    energy = d.data;
  }
};




template<int dim>
class NavierStokesInterface : public NonConservativeInterface<dim,dim,NavierStokesInterface<dim> >
{
public:
  NavierStokesInterface() :
    NonConservativeInterface<dim,dim, NavierStokesInterface<dim> >("Navier Stokes Inteface",
        "FESystem[FE_Q(2)^d-FE_Q(1)]",
        "u,u,p", 3, "1,1; 1,0", "1,0; 0,1") {};

  template<typename Number>
  void get_residual(const Scratch &d, std::vector<Number> &local_residual) const
  {
    local_residual[0] = d.data;
  }
};



template <int dim, int spacedim>
class piDoMUS
{
public:
  piDoMUS(const Interface<dim,spacedim> &pb) :
    problem(pb),
    names(pb.get_component_names())
  {
  };

  const Interface<dim,spacedim> &problem;
  const std::string names;
};

int main ()
{
  initlog();
  NavierStokesInterface<2> interface0;
  StokesInterface<2> interface1;

  piDoMUS<2,2> navier_stoks_driver(interface0);
  piDoMUS<2,2> stokes_driver(interface1);

  deallog << navier_stoks_driver.names << std::endl;
  deallog << navier_stoks_driver.problem.n_components() << std::endl;
  deallog << navier_stoks_driver.problem.n_blocks() << std::endl;

  ParameterAcceptor::initialize();

}
