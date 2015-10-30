#include "pidomus.h"
#include "interfaces/heat_equation.h"
#include <deal.II/fe/component_mask.h>
#include "tests.h"

typedef TrilinosWrappers::MPI::BlockVector VEC;


template<int fdim, int fspacedim, int fn_components>
void test(piDoMUSProblem<fdim,fspacedim,fn_components> &pb)
{
  pb.make_grid_fe();
  pb.setup_dofs();

  std::vector<bool> m(1,true);
  ComponentMask mask(m);

  VEC &d = pb.differential_components();


  d.block(0) = 11;
  d.print(deallog.get_file_stream());


}


using namespace dealii;
int main (int argc, char *argv[])
{

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                      numbers::invalid_unsigned_int);

  initlog();

  const int dim = 2;

  HeatEquation<dim> energy;
  piDoMUSProblem<dim,dim,1> n_problem (energy);
  ParameterAcceptor::initialize(SOURCE_DIR "/parameters/heat_equation_01.prm", "used_parameters.prm");

  deallog << "All dofs on boundary, set BC => all zeros" << std::endl;

  test(n_problem);

  return 0;
}
