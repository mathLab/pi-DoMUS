#ifndef __boundary_values_h
#define __boundary_values_h

#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>
#include "heart_fe.h"

using namespace dealii;

template <int dim>
class BoundaryValues : public Function<dim>
{
public:
  BoundaryValues (int cl) : Function<dim>(dim), color(cl) {}
  BoundaryValues (int cl, int ts, bool side, int degree) 
    : 
    Function<dim>(dim),
    color(cl), 
    timestep(ts),
    heart(side, degree) 
  {}

  virtual double value (const Point<dim>   &p,
                        const unsigned int  component = 0) const;
  virtual void vector_value (const Point<dim> &p,
                             Vector<double>   &value) const;
private:
  int color;
  int timestep;
  Heart<2,3> heart; 
  void transform_to_polar_coord(const Point<3> &p, 
                                double rot, 
                                double &angle, 
                                double &height) const;
  void swap_coord(Point<3> &p) const;
};

#endif
