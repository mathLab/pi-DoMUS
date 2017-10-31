#include "pidomus.h"
#include "interfaces/neo_hookean_two_fields.h"
#include "tests.h"

using namespace dealii;
int main (int argc, char *argv[])
{

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                      numbers::invalid_unsigned_int);

  initlog();

  deallog.depth_file(1);

  const int dim = 3;
  const int spacedim = 3;

  NeoHookeanTwoFieldsInterface<dim,spacedim,LATrilinos> energy;
  piDoMUS<dim,spacedim,LATrilinos> n_problem ("",energy);
  ParameterAcceptor::initialize(SOURCE_DIR "/parameters/incompressible_neo_hookean_two_fields_02.prm", "used_parameters.prm");

  n_problem.run ();

  auto &sol = n_problem.get_solution();
  for (unsigned int i = 0; i<sol.size(); ++i)
    deallog << sol[i] << std::endl;

  return 0;
}
