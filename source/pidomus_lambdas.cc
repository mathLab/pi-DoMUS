#include "pidomus_lambdas.h"
#include "pidomus.h"
#include "pidomus_macros.h"

template <int dim, int spacedim, typename LAC>
Lambdas<dim,spacedim,LAC>::Lambdas ()
{}


template <int dim, int spacedim, typename LAC>
Lambdas<dim,spacedim,LAC>::
Lambdas (piDoMUS<dim,spacedim,LAC> &simulator_object)
  :
  simulator (&simulator_object)
{}


template <int dim, int spacedim, typename LAC>
Lambdas<dim,spacedim,LAC>::~Lambdas ()
{}


template <int dim, int spacedim, typename LAC>
void
Lambdas<dim,spacedim,LAC>::
initialize_simulator (piDoMUS<dim,spacedim,LAC> &simulator_object)
{
  simulator = &simulator_object;
}


template <int dim, int spacedim, typename LAC>
void
Lambdas<dim,spacedim,LAC>::
set_functions_to_default()
{
  create_new_vector = [this]() ->shared_ptr<typename LAC::VectorType>
  {
    return this->simulator->create_new_vector();
  };

  residual = [this](const double t,
                    const typename LAC::VectorType &y,
                    const typename LAC::VectorType &y_dot,
                    typename LAC::VectorType &residual) ->int
  {
    return this->simulator->residual(t,y,y_dot,residual);
  };

  setup_jacobian = [this](const double t,
                          const typename LAC::VectorType &y,
                          const typename LAC::VectorType &y_dot,
                          const double alpha) ->int
  {
    shared_ptr<typename LAC::VectorType> res;
    return this->simulator->setup_jacobian(t,y,y_dot,*res,alpha);
  };

  solve_jacobian_system = [this](const typename LAC::VectorType &rhs,
                                 typename LAC::VectorType &dst) ->int
  {
    shared_ptr<typename LAC::VectorType> y;
    shared_ptr<typename LAC::VectorType> y_d;
    shared_ptr<typename LAC::VectorType> re;
    return this->simulator->solve_jacobian_system(0,*y,*y_d,*re,0,rhs,dst);
  };

  output_step = [this](const double t,
                       const typename LAC::VectorType &y,
                       const typename LAC::VectorType &y_dot,
                       const unsigned int step_number)
  {
    this->simulator->output_step(t,y,y_dot,step_number,0);
  };

  solver_should_restart = [this](const double t,
                                 typename LAC::VectorType &y,
                                 typename LAC::VectorType &y_dot) ->bool
  {
    return this->simulator->solver_should_restart(t,0,0,y,y_dot);
  };

  differential_components = [this]() ->typename LAC::VectorType &
  {
    return this->simulator->differential_components();
  };

  get_local_tolerances = [this]() ->typename LAC::VectorType &
  {
    return this->simulator->get_local_tolerances();
  };

  get_lumped_mass_matrix = [this]() ->typename LAC::VectorType &
  {
    auto lm = this->create_new_vector();
    this->simulator->get_lumped_mass_matrix(*lm);
    return *lm;
  };

  jacobian_vmult = [this](const typename LAC::VectorType &src,
                          typename LAC::VectorType &dst) ->int
  {
      return this->simulator->jacobian_vmult(src,dst);
    };

}

#define INSTANTIATE(dim,spacedim,LAC) \
  template class Lambdas<dim,spacedim,LAC>;

PIDOMUS_INSTANTIATE(INSTANTIATE)
