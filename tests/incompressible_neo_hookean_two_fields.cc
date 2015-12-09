#include "pidomus.h"
#include "interfaces/neo_hookean_two_fields.h"
#include "tests.h"

using namespace dealii;
int main (int argc, char *argv[])
{

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                      numbers::invalid_unsigned_int);

  initlog();

  const int dim = 3;
  const int spacedim = 3;

  NeoHookeanTwoFieldsInterface<dim,spacedim,LADealII> energy;
  piDoMUS<dim,spacedim,LADealII> n_problem ("",energy);
  ParameterAcceptor::initialize(SOURCE_DIR "/parameters/incompressible_neo_hookean_two_fields.prm", "used_parameters.prm");


  n_problem.run ();

  return 0;
}
