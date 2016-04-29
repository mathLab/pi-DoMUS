
#include "pidomus.h"
#include <deal2lkit/utilities.h>

#include <deal.II/base/mpi.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/distributed/solution_transfer.h>

#ifdef DEAL_II_WITH_ZLIB
#  include <zlib.h>
#endif

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>


template <int dim, int spacedim, typename LAC>
template<class Archive>
void piDoMUS<dim,spacedim,LAC>::serialize (Archive &ar, const unsigned int)
{
  ar &current_time;
  ar &current_alpha;
  ar &current_dt;
  ar &step_number;
}


