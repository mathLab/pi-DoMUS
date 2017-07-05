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
    int ret = this->simulator->residual(t,y,y_dot,residual);
    this->simulator->set_constrained_dofs_to_zero(residual);
    return ret;
  };

  setup_jacobian = [this](const double t,
                          const typename LAC::VectorType &y,
                          const typename LAC::VectorType &y_dot,
                          const double alpha) ->int
  {
    return this->simulator->setup_jacobian(t,y,y_dot,alpha);
  };

  solve_jacobian_system = [this](const typename LAC::VectorType &rhs,
                                 typename LAC::VectorType &dst) ->int
  {
    return this->simulator->solve_jacobian_system(rhs,dst);
  };

  output_step = [this](const double t,
                       const typename LAC::VectorType &y,
                       const typename LAC::VectorType &y_dot,
                       const unsigned int step_number)
  {
    this->simulator->output_step(t,y,y_dot,step_number);
  };

  solver_should_restart = [this](const double t,
                                 typename LAC::VectorType &y,
                                 typename LAC::VectorType &y_dot) ->bool
  {
    return this->simulator->solver_should_restart(t,y,y_dot);
  };

  differential_components = [this]() ->typename LAC::VectorType &
  {
    return this->simulator->differential_components();
  };

  get_local_tolerances = [this]() ->typename LAC::VectorType &
  {
    AssertThrow(false, ExcPureFunctionCalled("Please implement get_local_tolerances function."));
    static auto lt = this->create_new_vector();
    return *lt;
  };

  get_lumped_mass_matrix = [&]() ->typename LAC::VectorType &
  {
    static shared_ptr<typename LAC::VectorType> lm;
    lm = this->create_new_vector();
    this->simulator->get_lumped_mass_matrix(*lm);
    return *lm;
  };

  jacobian_vmult = [this](const typename LAC::VectorType &src,
                          typename LAC::VectorType &dst) ->int
  {
    return this->simulator->jacobian_vmult(src,dst);
  };

  vector_norm = [this](const typename LAC::VectorType &vector) ->double
  {
    return this->simulator->vector_norm(vector);
  };

}

#define INSTANTIATE(dim,spacedim,LAC) \
  template class Lambdas<dim,spacedim,LAC>;

PIDOMUS_INSTANTIATE(INSTANTIATE)
