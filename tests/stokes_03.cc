/**
 *  This test solves a Stokes problem with time depending forcing term.
 */
#include "pidomus.h"
#include "interfaces/stokes_nc.h"
#include "tests.h"

using namespace dealii;
int main (int argc, char *argv[])
{

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                      numbers::invalid_unsigned_int);

  initlog();
  deallog.depth_file(2);

  const int dim = 2;
  const int spacedim = 2;

  StokesNC<dim> energy;
  piDoMUS<dim,spacedim,dim+1> n_problem (energy);
  ParameterAcceptor::initialize(SOURCE_DIR "/parameters/stokes_03.prm", "used_parameters.prm");

  n_problem.run ();

  return 0;
}
