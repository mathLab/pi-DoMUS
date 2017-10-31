// test the SacadoUtilities functions
// for Number=double

#include "tests.h"
#include <deal.II/lac/full_matrix.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/vector.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_nedelec.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <fstream>

#include <deal2lkit/utilities.h>
#include "Sacado.hpp"

typedef Sacado::Fad::DFad<double> Sdouble;
typedef Sacado::Fad::DFad<Sdouble> SSdouble;


template<int dim>
void test (const Triangulation<dim> &tr,
           const FiniteElement<dim> &fe)
{
//  deallog << "FE=" << fe.get_name()
//          << std::endl;

  DoFHandler<dim> dof(tr);
  dof.distribute_dofs(fe);


  const QGauss<dim> quadrature(2);
  FEValues<dim> fe_values (fe, quadrature,
                           update_values | update_gradients);

  std::vector<types::global_dof_index>    local_dof_indices (fe_values.dofs_per_cell);
  std::vector<SSdouble> independent_local_dof_values (fe_values.dofs_per_cell);

  const unsigned int           dofs_per_cell = fe_values.dofs_per_cell;
  const unsigned int           n_q_points    = fe_values.n_quadrature_points;
  fe_values.reinit (dof.begin_active());
  dof.begin_active()->get_dof_indices (local_dof_indices);

  Vector<double> global_vector(dof.n_dofs());
  for (unsigned int i=0; i<dof.n_dofs(); ++i)
    global_vector[i] += i*i;

  SacadoUtilities::extract_local_dofs(global_vector, local_dof_indices, independent_local_dof_values);

//  for (unsigned int i=0; i<dof.n_dofs(); ++i)
//    deallog << independent_local_dof_values[i] << std::endl;


  FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
  FullMatrix<double>   sacado_matrix (dofs_per_cell, dofs_per_cell);

  std::vector <Tensor <2, dim, SSdouble> > grad_v(quadrature.size());
  std::vector<SSdouble>                    p(n_q_points);
  std::vector<SSdouble>                    div_u(n_q_points);

  std::vector <Tensor <2, dim, double> > grad_v_dealii(dofs_per_cell);
//  std::vector <SymmetricTensor <2, dim> > grad_v_dealii(dofs_per_cell);
  std::vector<double>                    p_dealii(dofs_per_cell);
  std::vector<double>                    div_u_dealii(dofs_per_cell);

  FEValuesExtractors::Scalar scalar (dim);
  FEValuesExtractors::Vector vector (0);

  SacadoUtilities::get_grad_values(fe_values, independent_local_dof_values, grad_v, vector);
  SacadoUtilities::get_values(fe_values, independent_local_dof_values, p, scalar);

  SSdouble en;
  for (unsigned int q=0; q<n_q_points; ++q)
    en += 0.5*(scalar_product(grad_v[q],grad_v[q])
               +
               p[q]*p[q])*fe_values.JxW(q);


  for (unsigned int i=0; i<dofs_per_cell; ++i)
    {
      for (unsigned int j=0; j<dofs_per_cell; ++j)
        {
          sacado_matrix(i,j) = en.dx(j).dx(i);
        }
    }

//  deallog <<std::endl;
//  deallog << "local _matrix sacado" << std::endl;
//  for (unsigned int i=0; i<dofs_per_cell; ++i)
//    {
//      for (unsigned int j=0; j<dofs_per_cell; ++j)
//        {
//          deallog << sacado_matrix(i,j) << " ";
//        }
//      deallog <<std::endl;
//    }
  for (unsigned int q=0; q<n_q_points; ++q)
    {
      for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
          grad_v_dealii[i] = fe_values[vector].gradient(i,q);
          div_u_dealii[i]  = fe_values[vector].divergence(i,q);
          p_dealii[i]  = fe_values[scalar].value (i, q);
          double diff=0;
          for (unsigned int k=0; k<dim; ++k)
            for (unsigned int j=0; j<dim; ++j)
              diff += std::abs(grad_v_dealii[i][k][j] - grad_v[q][k][j].dx(i).val());
          if (diff > 1e-10)
            {
              deallog << "k, " << i
                      << "q, " << q << std::endl
                      << "Gphi " << grad_v_dealii[i] << std::endl
                      << "GphiSS " << grad_v[q] << std::endl;
              AssertThrow(false, ExcMessage("Errore!!!"));
            }
        }

      for (unsigned int i=0; i<dofs_per_cell; ++i)
        for (unsigned int j=0; j<dofs_per_cell; ++j)
          local_matrix(i,j) += (scalar_product(grad_v_dealii[i],grad_v_dealii[j])
                                +p_dealii[i]*p_dealii[j])
                               *fe_values.JxW(q);

    }



//  deallog <<std::endl;
//  deallog << "local matrix -- dealii" << std::endl;
//  for (unsigned int i=0; i<dofs_per_cell; ++i)
//    {
//      for (unsigned int j=0; j<dofs_per_cell; ++j)
//        {
//          deallog << local_matrix(i,j) << " ";
//        }
//      deallog <<std::endl;
//    }

  double df;

  for (unsigned int i=0; i<dofs_per_cell; ++i)
    {
      for (unsigned int j=0; j<dofs_per_cell; ++j)
        {
          df = local_matrix(i,j) - sacado_matrix(i,j);
          if (df*df > 1e-10)
            {
              deallog << "(" << i<<", " << j<< ") --> diff = " << df <<std::endl;
            }
        }
    }
}




template<int dim>
void test_hyper_cube()
{
  Triangulation<dim> tr;
  GridGenerator::hyper_cube(tr);


  FESystem<dim> fe (FE_Q<dim>(2), dim,
                    FE_Q<dim>(1), 1);
  test(tr, fe);
}


int main()
{
  std::ofstream logfile ("output");
//  deallog << std::setprecision (3);

  deallog.attach(logfile);
  deallog.depth_console (0);

  test_hyper_cube<2>();
//  test_hyper_cube<3>();
}
