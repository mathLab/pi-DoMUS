#include "navier_stokes.h"

template <int dim>
NavierStokes<dim>::Postprocessor::
Postprocessor (const unsigned int partition,
               const double       minimal_pressure)
  :
  partition (partition),
  minimal_pressure (minimal_pressure)
{}


template <int dim>
std::vector<std::string>
NavierStokes<dim>::Postprocessor::get_names() const
{
  std::vector<std::string> solution_names (dim, "velocity");
  solution_names.push_back ("p");
  solution_names.push_back ("partition");

  return solution_names;
}


template <int dim>
std::vector<DataComponentInterpretation::DataComponentInterpretation>
NavierStokes<dim>::Postprocessor::
get_data_component_interpretation () const
{
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  interpretation (dim,
                  DataComponentInterpretation::component_is_part_of_vector);
  interpretation.push_back (DataComponentInterpretation::component_is_scalar);
  interpretation.push_back (DataComponentInterpretation::component_is_scalar);

  return interpretation;
}


template <int dim>
UpdateFlags
NavierStokes<dim>::Postprocessor::get_needed_update_flags() const
{
  return update_values | update_gradients | update_q_points;
}


template <int dim>
void
NavierStokes<dim>::Postprocessor::
compute_derived_quantities_vector (const std::vector<Vector<double> >              &uh,
                                   const std::vector<std::vector<Tensor<1,dim> > > &/*duh*/,
                                   const std::vector<std::vector<Tensor<2,dim> > > &/*dduh*/,
                                   const std::vector<Point<dim> >                  &/*normals*/,
                                   const std::vector<Point<dim> >                  &/*evaluation_points*/,
                                   std::vector<Vector<double> >                    &computed_quantities) const
{
  const unsigned int n_quadrature_points = uh.size();
  Assert (computed_quantities.size() == n_quadrature_points,  ExcInternalError());
  Assert (uh[0].size() == dim+1,                              ExcInternalError());

  for (unsigned int q=0; q<n_quadrature_points; ++q)
    {
      for (unsigned int d=0; d<dim; ++d)
        computed_quantities[q](d)
          = (uh[q](d));

      const double pressure = (uh[q](dim)-minimal_pressure);
      computed_quantities[q](dim) = pressure;
      computed_quantities[q](dim+1) = partition;
    }
}

// template class NavierStokes<1>;
template class NavierStokes<2>;
// template class NavierStokes<3>;
