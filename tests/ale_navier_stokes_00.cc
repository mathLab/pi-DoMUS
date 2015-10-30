#include "pidomus.h"
#include "interfaces/ale_navier_stokes.h"
#include "tests.h"

using namespace dealii;
int main (int argc, char *argv[])
{

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                      numbers::invalid_unsigned_int);

  initlog();

  const int dim = 2;
  const int spacedim = 2;

  ALENavierStokes<dim> energy;
  piDoMUSProblem<dim,spacedim,dim+dim+1> n_problem (energy);
  ParameterAcceptor::initialize(SOURCE_DIR "/parameters/ale_navier_stokes_00.prm", "used_parameters.prm");


  n_problem.run ();

  return 0;
}
