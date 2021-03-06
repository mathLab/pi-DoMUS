#include "pidomus.h"
#include "interfaces/dynamic_stokes.h"
#include "tests.h"

using namespace dealii;
int main (int argc, char *argv[])
{

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                      numbers::invalid_unsigned_int);

  initlog();

  const int dim = 2;
  const int spacedim = 2;

  DynamicStokes<dim> energy;
  piDoMUS<dim,spacedim,dim+1> n_problem (energy);
  ParameterAcceptor::initialize(SOURCE_DIR "/parameters/poiseuille.prm", "used_parameters.prm");


  n_problem.run ();

  return 0;
}
