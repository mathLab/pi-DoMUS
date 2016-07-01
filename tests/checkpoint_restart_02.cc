#include "pidomus.h"
#include "interfaces/poisson_problem.h"
#include "tests.h"

using namespace dealii;
int main (int argc, char *argv[])
{

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                      numbers::invalid_unsigned_int);

  MPILogInitAll log;

  deallog.depth_file(1);
  const int dim = 2;
  const int spacedim = 2;

  PoissonProblem<dim,spacedim,LATrilinos> p;
  piDoMUS<dim,spacedim,LATrilinos> solver ("pidomus",p);
  ParameterAcceptor::initialize(SOURCE_DIR "/parameters/checkpoint_restart_02.prm", "used_parameters.prm");


  solver.run ();

  deallog.depth_file(10);

  deallog << "OK" << std::endl;

  return 0;
}
