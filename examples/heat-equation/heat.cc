#include <pidomus.h>
#include "heat_interface.h"

int main (int argc, char *argv[])
{

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                      numbers::invalid_unsigned_int);


  const int dim = 2;
  const int spacedim = 2;

  // for serial version using a direct solver use uncomment these two
  // lines
  // HeatEquation<dim,spacedim,LADealII> problem;
  // piDoMUS<dim,spacedim,LADealII> solver ("pidomus",problem);

  // for parallel version using an iterative solver uncomment these
  // two lines
  HeatEquation<dim,spacedim,LATrilinos> problem;
  piDoMUS<dim,spacedim,LATrilinos> solver ("pidomus",problem);

  ParameterAcceptor::initialize("heat.prm", "used_parameters.prm");

  solver.run ();

  return 0;
}
