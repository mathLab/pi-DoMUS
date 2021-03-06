#include "pidomus.h"
#include "interfaces/stokes_derived_interface.h"
#include "tests.h"

using namespace dealii;
int main (int argc, char *argv[])
{

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                      numbers::invalid_unsigned_int);

  initlog();

  const int dim = 2;

  StokesDerivedInterface<dim> energy;
  piDoMUS<dim,dim,dim+1> n_problem (energy);
  ParameterAcceptor::initialize(SOURCE_DIR "/parameters/pidomus_01.prm", "used_parameters.prm");


  n_problem.run ();

  return 0;
}
