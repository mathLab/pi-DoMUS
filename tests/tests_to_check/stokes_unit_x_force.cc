#include "navier_stokes.h"



template<int dim>
void test(NavierStokes<dim> &ns)
{
  ns.make_grid_fe();
  ns.setup_dofs();
  ns.assemble_navier_stokes_system();
  ns.build_navier_stokes_preconditioner();
  ns.solve();
  ns.output_results();
}

int main (int argc, char *argv[])
{
  using namespace dealii;

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                      numbers::invalid_unsigned_int);

  deallog.depth_console (0);

  const int dim = 2;
  NavierStokes<dim> flow_problem (NavierStokes<dim>::global_refinement);

  ParameterAcceptor::initialize(SOURCE_DIR "/parameters/stokes_unit_force_x.prm", "used_parameters.prm");
  ParameterAcceptor::prm.log_parameters(deallog);

  test<dim>(flow_problem);
  std::cout << std::endl;

  return 0;
}
