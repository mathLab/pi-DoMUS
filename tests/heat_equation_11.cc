#include "pidomus.h"
#include "interfaces/heat_equation.h"
#include "tests.h"

typedef TrilinosWrappers::MPI::BlockVector VEC;



using namespace dealii;
int main (int argc, char *argv[])
{

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                      numbers::invalid_unsigned_int);

  initlog();

  const int dim = 2;

  HeatEquation<dim> energy;
  NFieldsProblem<dim,dim,1> n_problem (energy);
  ParameterAcceptor::initialize(SOURCE_DIR "/parameters/heat_equation_11.prm", "used_parameters.prm");

  n_problem.run();

  return 0;
}
