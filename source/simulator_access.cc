#include "pidomus.h"

template <int dim, int spacedim, typename LAC>
SimulatorAccess<dim,spacedim,LAC>::SimulatorAccess ()
{}


template <int dim, int spacedim, typename LAC>
SimulatorAccess<dim,spacedim,LAC>::
SimulatorAccess (const piDoMUS<dim,spacedim,LAC> &simulator_object)
  :
  simulator (&simulator_object)
{}


template <int dim, int spacedim, typename LAC>
SimulatorAccess<dim,spacedim,LAC>::~SimulatorAccess ()
{}



template <int dim, int spacedim, typename LAC>
void
SimulatorAccess<dim,spacedim,LAC>::
initialize_simulator (const piDoMUS<dim,spacedim,LAC> &simulator_object) const
{
  simulator = &simulator_object;
}



template <int dim, int spacedim, typename LAC>
const piDoMUS<dim,spacedim,LAC> &
SimulatorAccess<dim,spacedim,LAC>::get_simulator() const
{
  return *simulator;
}



template <int dim, int spacedim, typename LAC>
Signals<dim,spacedim,LAC> &
SimulatorAccess<dim,spacedim,LAC>::get_signals() const
{
  // we need to connect to the signals so a const_cast is required
  return const_cast<Signals<dim,spacedim,LAC>&>(simulator->signals);
}


#ifdef DEAL_II_WITH_MPI
template <int dim, int spacedim, typename LAC>
MPI_Comm SimulatorAccess<dim,spacedim,LAC>::get_mpi_communicator () const
{
  return simulator->comm;
}
#endif


template <int dim, int spacedim, typename LAC>
const ConditionalOStream &
SimulatorAccess<dim,spacedim,LAC>::get_pcout () const
{
  return simulator->pcout;
}

template <int dim, int spacedim, typename LAC>
double SimulatorAccess<dim,spacedim,LAC>::get_current_time () const
{
  return simulator->current_time;
}

template <int dim, int spacedim, typename LAC>
double SimulatorAccess<dim,spacedim,LAC>::get_alpha () const
{
  return simulator->current_alpha;
}

template <int dim, int spacedim, typename LAC>
double SimulatorAccess<dim,spacedim,LAC>::get_timestep () const
{
  return simulator->current_dt;
}


// template <int dim, int spacedim, typename LAC>
// unsigned int SimulatorAccess<dim,spacedim,LAC>::get_timestep_number () const
// {
//   return simulator->timestep_number;
// }


template <int dim, int spacedim, typename LAC>
const parallel::distributed::Triangulation<dim,spacedim> &
SimulatorAccess<dim,spacedim,LAC>::get_triangulation () const
{
  return *simulator->triangulation;
}


template <int dim, int spacedim, typename LAC>
const typename LAC::VectorType &
SimulatorAccess<dim,spacedim,LAC>::get_solution () const
{
  return simulator->solution;
}

template <int dim, int spacedim, typename LAC>
const typename LAC::VectorType &
SimulatorAccess<dim,spacedim,LAC>::get_solution_dot () const
{
  return simulator->solution_dot;
}

template <int dim, int spacedim, typename LAC>
const typename LAC::VectorType &
SimulatorAccess<dim,spacedim,LAC>::get_locally_relevant_solution () const
{
  return simulator->locally_relevant_solution;
}

template <int dim, int spacedim, typename LAC>
const typename LAC::VectorType &
SimulatorAccess<dim,spacedim,LAC>::get_locally_relevant_solution_dot () const
{
  return simulator->locally_relevant_solution_dot;
}

template <int dim, int spacedim, typename LAC>
const typename LAC::VectorType &
SimulatorAccess<dim,spacedim,LAC>::get_locally_relevant_explicit_solution () const
{
  return simulator->locally_relevant_explicit_solution;
}


template <int dim, int spacedim, typename LAC>
const DoFHandler<dim,spacedim> &
SimulatorAccess<dim,spacedim,LAC>::get_dof_handler () const
{
  return *simulator->dof_handler;
}


template <int dim, int spacedim, typename LAC>
const FiniteElement<dim,spacedim> &
SimulatorAccess<dim,spacedim,LAC>::get_fe () const
{
  Assert (simulator->dof_handler->n_locally_owned_dofs() != 0,
          ExcMessage("You are trying to access the FiniteElement before the DOFs have been "
                     "initialized. This may happen when accessing the Simulator from a plugin "
                     "that gets executed early in some cases (like material models) or from "
                     "an early point in the core code."));
  return simulator->dof_handler->get_fe();
}

template <int dim, int spacedim, typename LAC>
const ParsedDirichletBCs<dim, spacedim> &
SimulatorAccess<dim,spacedim,LAC>::get_dirichlet_bcs() const
{
  return simulator->dirichlet_bcs;
}



template class SimulatorAccess<2, 2, LATrilinos>;
template class SimulatorAccess<2, 3, LATrilinos>;
template class SimulatorAccess<3, 3, LATrilinos>;

template class SimulatorAccess<2, 2, LADealII>;
template class SimulatorAccess<2, 3, LADealII>;
template class SimulatorAccess<3, 3, LADealII>;

