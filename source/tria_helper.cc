#include "pidomus_macros.h"
#include <tria_helper.h>
#include <deal.II/base/utilities.h>

#ifdef DEAL_II_WITH_MPI
template <int dim, int spacedim, typename LAC>
TriaHelper<dim,spacedim,LAC>::TriaHelper(const MPI_Comm _comm):
  comm(Utilities::MPI::duplicate_communicator(_comm)),
  pgg("Domain"),
  p_serial(nullptr),
  p_parallel(nullptr)
{}

template <int dim, int spacedim, typename LAC>
TriaHelper<dim,spacedim,LAC>::~TriaHelper()
{
  p_serial.reset();
  p_parallel.reset();
  MPI_Comm_free(&comm);
}

#else
template <int dim, int spacedim, typename LAC>
TriaHelper<dim,spacedim,LAC>::TriaHelper()
  :
  pgg("Domain"),
  p_serial(nullptr),
{}
#endif

template <int dim, int spacedim, typename LAC>
void TriaHelper<dim,spacedim,LAC>::make_grid()
{
  if (LAC::triatype == TriaType::serial)
    p_serial = shared_ptr<Triangulation<dim,spacedim> >(pgg.serial());
#ifdef DEAL_II_WITH_MPI
  else
    p_parallel = shared_ptr<parallel::distributed::Triangulation<dim,spacedim> >(pgg.distributed(comm));
#endif
}

template <int dim, int spacedim, typename LAC>
Triangulation<dim, spacedim> *
TriaHelper<dim,spacedim,LAC>::get_tria() const
{
  if (LAC::triatype == TriaType::serial)
    return p_serial.get();
#ifdef DEAL_II_WITH_MPI
  return p_parallel.get();
#endif
}

#define INSTANTIATE(dim,spacedim,LAC) \
  template class TriaHelper<dim,spacedim,LAC>;


PIDOMUS_INSTANTIATE(INSTANTIATE)
