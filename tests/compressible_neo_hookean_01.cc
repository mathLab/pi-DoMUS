#include "pidomus.h"
#include "interfaces/compressible_neo_hookean.h"
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

  CompressibleNeoHookeanInterface<dim,spacedim> cnh_body;
  piDoMUS<dim,spacedim> solver ("",cnh_body);
  ParameterAcceptor::initialize(SOURCE_DIR "/parameters/compressible_neo_hookean_01.prm", "used_parameters.prm");


  solver.run ();

  auto &sol = solver.get_solution();
  for (unsigned int i = 0; i<sol.size(); ++i)
    deallog << sol[i] << std::endl;

  return 0;
}
