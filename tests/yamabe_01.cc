#include "pidomus.h"
#include "interfaces/yamabe.h"
#include "tests.h"

using namespace dealii;
int main (int argc, char *argv[])
{

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                      numbers::invalid_unsigned_int);

  initlog();
  deallog.depth_file(1);

  const int dim = 2;
  const int spacedim = 2;

  YamabeProblem<dim,spacedim,LADealII> p;
  piDoMUS<dim,spacedim,LADealII> solver ("pidomus",p);
  ParameterAcceptor::initialize(SOURCE_DIR "/parameters/yamabe_01.prm", "used_parameters.prm");


  // solver.run ();

  // auto sol = solver.get_solution();
  // for (unsigned int i = 0; i<sol.size(); ++i)
  //   deallog << sol[i] << std::endl;

  deallog << "OK" << std::endl;

  return 0;
}
