#include "pidomus.h"
#include "interfaces/poisson_problem_signals.h"
#include "tests.h"


//test that we can resume a computation from snapshots
using namespace dealii;
int main (int argc, char *argv[])
{

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                      numbers::invalid_unsigned_int);

  initlog();
  deallog.depth_file(1);

  const int dim = 2;
  const int spacedim = 2;

  PoissonProblem<dim,spacedim,LADealII> p;
  piDoMUS<dim,spacedim,LADealII> solver ("pidomus",p);
  ParameterAcceptor::initialize(SOURCE_DIR "/parameters/checkpoint_restart_04.prm", "used_parameters.prm");


  solver.run ();

  auto sol = solver.get_solution();
  deallog.depth_file(10);
  deallog << sol.l2_norm() << std::endl;

  return 0;
}
