#include "pidomus.h"
#include "interfaces/free_swelling_three_fields.h"
#include "tests.h"

using namespace dealii;
int main (int argc, char *argv[])
{

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
//                                                      numbers::invalid_unsigned_int);

  initlog();

  const int dim = 3;
  const int spacedim = 3;

  FreeSwellingThreeFields<dim,spacedim> energy;
  NFieldsProblem<dim,spacedim,dim+2,LADealII> n_problem (energy);
  ParameterAcceptor::initialize(SOURCE_DIR "/parameters/free_swelling_01.prm", "used_parameters.prm");


  n_problem.run ();

  return 0;
}
