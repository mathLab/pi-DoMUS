#include "elastic_problem.h"
#include "boundary_values.h"
#include <iostream>

template <int dim>
class RightHandSide :  public Function<dim>
{
public:
  RightHandSide ();
  virtual void vector_value (const Point<dim> &p,
                             Vector<double>   &values) const;
  virtual void vector_value_list (const std::vector<Point<dim> > &points,
                                  std::vector<Vector<double> >   &value_list) const;
};

template <int dim>
RightHandSide<dim>::RightHandSide ()
  :
  Function<dim> (dim)
{}

template <int dim>
inline
void RightHandSide<dim>::vector_value (const Point<dim> &p,
                                       Vector<double>   &values) const
{
  Assert (values.size() == dim,
          ExcDimensionMismatch (values.size(), dim));
  Assert (dim >= 2, ExcNotImplemented());
  Point<dim> point_1, point_2;
  point_1(0) = 0.5;
  point_2(0) = -0.5;
  if (((p-point_1).norm_square() < 0.2*0.2) ||
      ((p-point_2).norm_square() < 0.2*0.2))
    values(0) = 1;
  else
    values(0) = 0;
  if (p.norm_square() < 0.2*0.2)
    values(1) = 1;
  else
    values(1) = 0;
}

template <int dim>
void RightHandSide<dim>::vector_value_list (const std::vector<Point<dim> > &points,
                                            std::vector<Vector<double> >   &value_list) const
{
  Assert (value_list.size() == points.size(),
          ExcDimensionMismatch (value_list.size(), points.size()));
  const unsigned int n_points = points.size();
  for (unsigned int p=0; p<n_points; ++p)
    RightHandSide<dim>::vector_value (points[p],
                                      value_list[p]);
}



template <int dim>
ElasticProblem<dim>::ElasticProblem ()
  :
  dof_handler (triangulation),
  fe (FE_Q<dim>(1), dim)
{}

template <int dim>
ElasticProblem<dim>::ElasticProblem (double timestep, double dt)
  :
  dof_handler (triangulation),
  fe (FE_Q<dim>(1), dim),
  timestep(timestep),
  dt(dt)
{}

template <int dim>
ElasticProblem<dim>::~ElasticProblem ()
{
  dof_handler.clear ();
}

template <int dim>
void ElasticProblem<dim>::setup_system ()
{
  dof_handler.distribute_dofs (fe);
  hanging_node_constraints.clear ();
  DoFTools::make_hanging_node_constraints (dof_handler,
                                           hanging_node_constraints);
  hanging_node_constraints.close ();
  DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler,
                                  dsp,
                                  hanging_node_constraints,
                                  /*keep_constrained_dofs = */ true);
  sparsity_pattern.copy_from (dsp);
  system_matrix.reinit (sparsity_pattern);
  solution.reinit (dof_handler.n_dofs());
  system_rhs.reinit (dof_handler.n_dofs());
}

template <int dim>
void ElasticProblem<dim>::assemble_system ()
{
  QGauss<dim>  quadrature_formula(2);
  FEValues<dim> fe_values (fe, quadrature_formula,
                           update_values   | update_gradients |
                           update_quadrature_points | update_JxW_values);
  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();
  FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       cell_rhs (dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
  std::vector<double>     lambda_values (n_q_points);
  std::vector<double>     mu_values (n_q_points);
  ConstantFunction<dim> lambda(0.), mu(1.);
  RightHandSide<dim>      right_hand_side;
  std::vector<Vector<double> > rhs_values (n_q_points,
                                           Vector<double>(dim));
  typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
                                                 endc = dof_handler.end();
  for (; cell!=endc; ++cell)
    {
      cell_matrix = 0;
      cell_rhs = 0;
      fe_values.reinit (cell);
      lambda.value_list (fe_values.get_quadrature_points(), lambda_values);
      mu.value_list     (fe_values.get_quadrature_points(), mu_values);
      right_hand_side.vector_value_list (fe_values.get_quadrature_points(),
                                         rhs_values);
      for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
          const unsigned int
          component_i = fe.system_to_component_index(i).first;
          for (unsigned int j=0; j<dofs_per_cell; ++j)
            {
              const unsigned int
              component_j = fe.system_to_component_index(j).first;
              for (unsigned int q_point=0; q_point<n_q_points;
                   ++q_point)
                {
                  cell_matrix(i,j)
                  +=
                    (
                      (fe_values.shape_grad(i,q_point)[component_i] *
                       fe_values.shape_grad(j,q_point)[component_j] *
                       lambda_values[q_point])
                      +
                      (fe_values.shape_grad(i,q_point)[component_j] *
                       fe_values.shape_grad(j,q_point)[component_i] *
                       mu_values[q_point])
                      +
                      ((component_i == component_j) ?
                       (fe_values.shape_grad(i,q_point) *
                        fe_values.shape_grad(j,q_point) *
                        mu_values[q_point])  :
                       0)
                    )
                    *
                    fe_values.JxW(q_point);
                }
            }
        }
      for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
          const unsigned int
          component_i = fe.system_to_component_index(i).first;
          for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
            cell_rhs(i) += fe_values.shape_value(i,q_point) *
                           rhs_values[q_point](component_i) *
                           fe_values.JxW(q_point);
        }
      cell->get_dof_indices (local_dof_indices);
      for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
          for (unsigned int j=0; j<dofs_per_cell; ++j)
            system_matrix.add (local_dof_indices[i],
                               local_dof_indices[j],
                               cell_matrix(i,j));
          system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }
    }
  hanging_node_constraints.condense (system_matrix);
  hanging_node_constraints.condense (system_rhs);
  std::map<types::global_dof_index,double> boundary_values;

  // top face
  VectorTools::interpolate_boundary_values (dof_handler,
                                            2,
                                            BoundaryValues<dim>(2),
                                            boundary_values);

  // bottom face
  VectorTools::interpolate_boundary_values (dof_handler,
                                            1,
                                            BoundaryValues<dim>(1, timestep, dt, false, 2),
                                            boundary_values);

  // hull
  VectorTools::interpolate_boundary_values (dof_handler,
                                            0,
                                            BoundaryValues<dim>(0, timestep, dt, true, 4),
                                            boundary_values);

  MatrixTools::apply_boundary_values (boundary_values,
                                      system_matrix,
                                      solution,
                                      system_rhs);
}

template <int dim>
void ElasticProblem<dim>::solve ()
{
  SolverControl           solver_control (1000, 1e-12);
  SolverCG<>              cg (solver_control);
  PreconditionSSOR<> preconditioner;
  preconditioner.initialize(system_matrix, 1.2);
  cg.solve (system_matrix, solution, system_rhs,
            preconditioner);
  hanging_node_constraints.distribute (solution);
}

template <int dim>
void ElasticProblem<dim>::refine_grid ()
{
  Vector<float> estimated_error_per_cell (triangulation.n_active_cells());
  KellyErrorEstimator<dim>::estimate (dof_handler,
                                      QGauss<dim-1>(2),
                                      typename FunctionMap<dim>::type(),
                                      solution,
                                      estimated_error_per_cell);
  GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                                                   estimated_error_per_cell,
                                                   0.3, 0.03);
  triangulation.execute_coarsening_and_refinement ();
}

template <int dim>
void ElasticProblem<dim>::run (Triangulation<dim> &external_tria)
{
  // create grid
  // cylinder from x = -3.5 to x = 1.6838 with radius = 1.3858
  // height: 1.6838 + 3.5 = 5.1838 --> half height: 2.5919
  //                      (triangulation, radius, half_length)
  GridGenerator::cylinder (triangulation, 1.3858, 2.5919);

  // shift mesh in x direction such that the top face is at x = 1.6838
  // 1.6838 - 2.5919 = -0.9081
  Tensor<1, 3> shift_vec;
  shift_vec[0] = -0.9081;
  GridTools::shift(shift_vec, triangulation);

  triangulation.set_all_manifold_ids_on_boundary(0,0);
  static const CylindricalManifold<dim> cylM;
  triangulation.set_manifold (0, cylM);

  triangulation.refine_global (2);

  // solve
  setup_system ();
  assemble_system ();
  solve ();

  external_tria.copy_triangulation(triangulation);
}

// Explicit instantiations
template class ElasticProblem<3>;
