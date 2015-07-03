#include "n_fields_problem.h"
#include "interfaces/heat_equation.h"
#include "tests.h"

typedef TrilinosWrappers::MPI::BlockVector VEC;


template<int fdim, int fspacedim, int fn_components>
void test(NFieldsProblem<fdim,fspacedim,fn_components> &pb)
{
  pb.make_grid_fe();
  pb.setup_dofs();


  VEC &d = pb.differential_components();
	VEC *sol = &pb.solution;
	VEC *sol_dot = &pb.solution_dot;
	VEC residual = *sol;
	VEC new_sol = *sol;

  deallog << "differential components" <<std::endl;
  d.print(deallog.get_file_stream());

  deallog << "solution" <<std::endl;
	sol->print(deallog.get_file_stream());
  deallog << "solution_dot" <<std::endl;
  sol_dot->print(deallog.get_file_stream());
	pb.residual(0.0,*sol,*sol_dot,residual);
  deallog << "residual  " <<std::endl;
  residual.print(deallog.get_file_stream());

	deallog << "assemble preconditioner" <<std::endl;
	pb.assemble_jacobian_preconditioner(0.0, *sol, *sol_dot, 0.33);
	pb.assemble_jacobian(0.0, *sol, *sol_dot, 0.33);

	pb.solve_jacobian_system(new_sol, residual, 1e-5);

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
  ParameterAcceptor::initialize(SOURCE_DIR "/parameters/heat_equation_06.prm", "used_parameters.prm");

  test(n_problem);

  return 0;
}
