#include "pidomus.h"
#include "interfaces/scalar_reaction_diffusion_convection_stabilized.h"
#include "tests.h"

/**
 * Test:     scalar_reaction_diffusion_convection_stabilized
 */

using namespace dealii;
int main (int argc, char *argv[])
{

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                      numbers::invalid_unsigned_int);

  initlog();
  deallog.depth_file(1);
  deallog.threshold_double(1.0e-3);

  ScalarReactionDiffusionConvection<2,2> energy;
  piDoMUS<2,2> problem ("",energy);
  ParameterAcceptor::initialize(
    SOURCE_DIR "/parameters/scalar_reaction_diffusion_convection_stabilized.prm",
    "used_parameters.prm");

  problem.run ();

  auto sol = problem.get_solution();
  for (unsigned int i = 0 ; i<sol.size(); ++i)
    {
      deallog << sol[i] << std::endl ;
    }

  return 0;
}
