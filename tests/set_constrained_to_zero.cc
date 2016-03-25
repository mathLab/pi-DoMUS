#include "pidomus.h"
#include "interfaces/poisson_problem.h"
#include "tests.h"

using namespace dealii;
template<int fdim, int fspacedim, typename fn_LAC>
void test(piDoMUS<fdim,fspacedim,fn_LAC>  &pi_foo)
{
  pi_foo.constraints.clear();
  pi_foo.constraints.add_line(0);
  pi_foo.constraints.close();

  pi_foo.global_partitioning.clear();
  pi_foo.global_partitioning.set_size(2);

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
    pi_foo.global_partitioning.add_index(0);
  else
    pi_foo.global_partitioning.add_index(1);
  pi_foo.global_partitioning.compress();
  LATrilinos::VectorType foo_vector(std::vector<IndexSet> (1,pi_foo.global_partitioning),MPI_COMM_WORLD);
  foo_vector = 0;
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
    foo_vector[0] = 1.;
  else
    foo_vector[1] = 1.;
  foo_vector.compress(VectorOperation::insert);

  pi_foo.set_constrained_dofs_to_zero(foo_vector);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
    foo_vector[0] += 1.;
  else
    foo_vector[1] += 1.;
  foo_vector.compress(VectorOperation::add);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
    std::cout<<foo_vector[0]<<std::endl;
  MPI_Barrier(MPI_COMM_WORLD);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==1)
    std::cout<<foo_vector[1]<<std::endl;



}


int main (int argc, char *argv[])
{

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                      numbers::invalid_unsigned_int);


  const int dim = 2;
  const int spacedim = 3;

  PoissonProblem<dim,spacedim,LATrilinos> p;
  piDoMUS<dim,spacedim,LATrilinos> solver ("pidomus",p);
  test(solver);

  return 0;
}
