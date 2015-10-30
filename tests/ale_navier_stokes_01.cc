#include "n_fields_problem.h"
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
  NFieldsProblem<dim,spacedim,dim+dim+1> n_problem (energy);
  ParameterAcceptor::initialize(SOURCE_DIR "/parameters/ale_navier_stokes_01.prm", "used_parameters.prm");


  n_problem.run ();

  return 0;
}
