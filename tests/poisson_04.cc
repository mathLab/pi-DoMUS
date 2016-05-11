#include "pidomus.h"
#include "interfaces/poisson_problem.h"
#include "tests.h"

using namespace dealii;
int main (int argc, char *argv[])
{

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  initlog();
  deallog.depth_file(1);

  const int dim = 2;
  const int spacedim = 2;

  PoissonProblem<dim,spacedim,LAPETSc> p;
  piDoMUS<dim,spacedim,LAPETSc> solver ("pidomus",p);
  ParameterAcceptor::initialize(SOURCE_DIR "/parameters/poisson_problem_04.prm", "used_parameters.prm");


  solver.run ();

  auto sol = solver.get_solution();
  for (unsigned int i = 0; i<sol.size(); ++i)
    deallog << sol[i] << std::endl;

  return 0;
}
