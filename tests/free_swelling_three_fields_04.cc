#include "n_fields_problem.h"
#include "interfaces/free_swelling.h"
#include "tests.h"

using namespace dealii;
int main (int argc, char *argv[])
{

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                      numbers::invalid_unsigned_int);

  initlog();
  deallog.depth_file(0);


  const int dim = 3;
  const int spacedim = 3;

  FreeSwellingThreeField<dim,spacedim> energy;
  NFieldsProblem<dim,spacedim,dim+2> n_problem (energy);
  ParameterAcceptor::initialize(SOURCE_DIR "/parameters/free_swelling_04.prm", "used_parameters.prm");


  n_problem.run ();

  return 0;
}
