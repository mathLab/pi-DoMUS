#include "n_fields_problem.h"
#include "interfaces/dynamic_stokes.h"
#include "tests.h"

template<int dim, int spacedim, int n_components>
void test(NFieldsProblem<dim,spacedim,n_components> &pb)
{
  for (pb.current_cycle=0; pb.current_cycle<pb.n_cycles; ++pb.current_cycle)
    {
      if (pb.current_cycle == 0)
        {
          pb.make_grid_fe();
          pb.setup_dofs(true);
        }
      else
        pb.refine_mesh();

      pb.constraints.distribute(pb.solution);

      auto rhs = pb.create_new_vector();
      auto du = pb.create_new_vector();

      pb.residual(0.0, pb.solution, pb.solution_dot, *rhs);
      pb.setup_jacobian(0.0,pb.solution,pb.solution_dot, *rhs,1.0);
      pb.solve_jacobian_system(0.0, pb.solution, pb.solution_dot, *rhs,
                               1.0, *rhs, *du);

      pb.set_constrained_dofs_to_zero(*du);

      pb.solution += *du;

      pb.distributed_solution = pb.solution;

      pb.eh.error_from_exact(*pb.mapping, *pb.dof_handler, pb.distributed_solution, pb.exact_solution);

      pb.output_step(0.0, pb.solution, pb.solution_dot, 0, 0.0);
    }

  pb.eh.output_table(pb.pcout);

}

using namespace dealii;
int main (int argc, char *argv[])
{

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                      numbers::invalid_unsigned_int);

  initlog();

  const int dim = 2;
  const int spacedim = 2;

  DynamicStokes<dim> energy;
  NFieldsProblem<dim,spacedim,dim+1> n_problem (energy);
  ParameterAcceptor::initialize(SOURCE_DIR "/parameters/stokes_01.prm", "used_parameters.prm");


  test(n_problem);

  return 0;
}
