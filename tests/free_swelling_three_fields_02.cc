#include "pidomus.h"
#include "interfaces/free_swelling_three_fields.h"
#include "tests.h"
#include "lac_type.h"

template<int dim, int spacedim, int n_components,typename LAC>
void test(piDoMUS<dim,spacedim,n_components,LAC> &pb)
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
      *rhs *= -1.0;
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

  const int dim = 3;
  const int spacedim = 3;

  FreeSwellingThreeFields<dim,spacedim> energy;
  piDoMUS<dim,spacedim,dim+2,LADealII> n_problem (energy);
  ParameterAcceptor::initialize(SOURCE_DIR "/parameters/free_swelling_02.prm", "used_parameters.prm");


  test(n_problem);

  return 0;
}
