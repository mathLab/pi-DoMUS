#include "n_fields_problem.h"
#include "interfaces/heat_equation.h"
#include "tests.h"

typedef TrilinosWrappers::MPI::BlockVector VEC;


template<int fdim, int fspacedim, int fn_components>
void test(NFieldsProblem<fdim,fspacedim,fn_components> &pb)
{
//  pb.make_grid_fe();
//  pb.setup_dofs();
//
//  VEC &d = pb.differential_components();
//
//  d.print(deallog.get_file_stream());
	pb.run();

}


using namespace dealii;
int main (int argc, char *argv[])
{

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                      numbers::invalid_unsigned_int);

  initlog();

  const int dim = 2;

  HeatEquation<dim> energy;
  NFieldsProblem<dim,dim,1> n_problem (energy);
  ParameterAcceptor::initialize(SOURCE_DIR "/parameters/heat_equation_09.prm", "used_parameters.prm");


  test(n_problem);

  return 0;
}
