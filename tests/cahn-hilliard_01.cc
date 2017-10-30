#include "pidomus.h"
#include "interfaces/cahn-hilliard.h"
#include "tests.h"
#include <iomanip>

using namespace dealii;
int main (int argc, char *argv[])
{

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                      numbers::invalid_unsigned_int);

  initlog();
  deallog.depth_file(1);

  const int dim = 2;

  CahnHilliard<dim,LADealII> p;
  piDoMUS<dim,dim,LADealII> solver ("pidomus",p);
  ParameterAcceptor::initialize(SOURCE_DIR "/parameters/cahn_hilliard_01.prm", "used_parameters.prm");


  solver.run ();

  auto &sol = solver.get_solution();
  for (unsigned int i = 0; i<sol.size(); ++i)
    deallog << std::fixed << std::setprecision(3) << sol[i] << std::endl;

  return 0;
}
