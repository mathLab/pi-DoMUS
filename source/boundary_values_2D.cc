#include "boundary_values_2D.h"
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/grid/grid_tools.h>
#define PI 3.14159265358979323846

using namespace dealii;

// - - - - -  public functions - - - - -
template <int dim>
double
BoundaryValues<dim>::value (const Point<dim>  &p,
                            const unsigned int component) const
{
  Assert (component < this->n_components,
          ExcIndexRange (component, 0, this->n_components));

  Vector<double> values(4);
  BoundaryValues<dim>::vector_value (p, values);

  return 0;
}

template <int dim>
void
BoundaryValues<dim>::vector_value (const Point<dim> &p,
                                   Vector<double>   &values) const
{
    if(derivative)
        BoundaryValues<dim>::get_values_dt(p, values);
    else
    {
        if (color == 3) {
            get_heartdelta(p, values, heartstep);
        }
        else
            BoundaryValues<dim>::get_values(p, values);
    }
}



// - - - - -  private functions - - - - -

template <int dim>
void
BoundaryValues<dim>::transform_to_polar_coord (const Point<3> &p,
                                               double rot,
                                               double &angle, 
                                               double &height) const
{
  // convert point to polar coordinates
  double x   = p[0],
         y   = p[1],
         z   = p[2],
         phi = atan2(y,z); // returns angle in the range from -Pi to Pi

  // need to rotate to match heart_fe
  phi = phi+rot;

  // angle needs to be in the range from 0 to 2Pi
  phi = (phi < 0) ? phi+2*PI : phi;
  
  angle = phi;
  height = x;
}

template <int dim>
void
BoundaryValues<dim>::swap_coord (Point<3> &p) const
{
  // swap x and z coordinate
  double tmp = p(0);
  //std::cout << "Hallo error1" << std::endl;
  p(0) = p(2);
  p(2) = tmp;
}

template <int dim>
void
BoundaryValues<dim>::setup_system()
{
  std::vector<unsigned int> subdivisions(1);
  subdivisions[0] = 1;
  const Point<1> p1 (-1);
  const Point<1> p2 (1);

  GridGenerator::subdivided_hyper_rectangle(triangulation, 
                                            subdivisions, 
                                            p1, p2, false);
  dof_handler.distribute_dofs(fe);
  Point<1> direction (1e5);
  DoFRenumbering::downstream (dof_handler, direction, true);
}

template <int dim>
void
BoundaryValues<dim>::get_heartdelta (const Point<dim> &p,
                                     Vector<double>   &values,
                                     int heartstep) const
{
  //onst Point<3> p (point(1), 0, point(0)); //?
  if (color == 3)         //////////////////////////////// top face 
  {
      // convert to polar coordinates and rotate 45 degrees
      double phi, h, rot = -PI/4;
      Point<3> artificial_p (p(1), p(0), 0);
      transform_to_polar_coord(artificial_p, rot, phi, h);

      // transform back to cartesian
      double x, y, r;
      x = p(0);
      y = 0;

      r = sqrt(x*x + y*y);
      x = r * cos(phi);
      y = r * sin(phi);
      
      // calc delta
      values(0) = y-p(0);
      values(1) = 0;
  }
  else if (color == 2)    //////////////////////////////// bottom face
  {
      double x, y;
      x = p(0);
      y = p(0);

      Point<2> two_dim_pnt (x, y);
      //std::cout << "(" << x << ", " << y << ")" << std::endl;
      //get heart boundary point
      Point<3> heart_p = heart.push_forward (two_dim_pnt, heartstep);
      swap_coord(heart_p);
      //std::cout << "Hallo error2" << std::endl;
      // calc delta
      values(0) = heart_p(1) - p(0);
      values(1) = heart_p(0) - p(1);
      //std::cout << "Hallo error3" << std::endl;
      //values(0) = 0;
      //values(1) = 0;
  }
  else if (color == 0 || color == 1)    //////////////////////////////// hull
  {
      // convert to polar coordinates and rotate -45 degrees
      double phi, h, rot = -PI/2;
      Point<3> artificial_p (p(1), p(0), 0);
      transform_to_polar_coord(artificial_p, rot, phi, h);
      Point<2> polar_pnt (phi, h);
      
      //get heart boundary point
      Point<3> heart_p = heart.push_forward (polar_pnt, heartstep);
      swap_coord(heart_p);

      // calc delta
      //values(0) = heart_p(0) - p(0);
      //values(1) = heart_p(1) - p(1);
      //values(2) = heart_p(2) - p(2);
      values(0) = heart_p(1) - artificial_p(1);
      values(1) = heart_p(0) - artificial_p(0);
  }
}

template <int dim>
void
BoundaryValues<dim>::get_values (const Point<dim> &p,
                                 Vector<double>   &values) const
{
    int n_dofs = dof_handler.n_dofs();
    Vector<double> solution(n_dofs);

    std::vector<Vector<double> > u(3);
    u[0].reinit(dim);
    u[1].reinit(dim);
    u[2].reinit(dim);

    //Vector<double> u_(dim);                                // u_t-1
    //Vector<double> u(dim);                                 // u_t
    //Vector<double> delta_u(dim);                             // u_t - u_t-1
    Point<1> substep (fmod (timestep, heartinterval)/heartinterval);

    BoundaryValues<dim>::get_heartdelta(p, u[0], 0);
    BoundaryValues<dim>::get_heartdelta(p, u[1], 1);
    BoundaryValues<dim>::get_heartdelta(p, u[2], 2);
    
    int counter = 0;
    for (int line = 0; line < 3; ++line)
    {
      for (int column = 0; column < dim; ++column)
      {
        solution(counter) = u[line](column);
        ++counter;
      }
    }
    auto cell = GridTools::find_active_cell_around_point (dof_handler,
                                                          substep);
    Point<1> scaled_point ( (substep(0) - cell->vertex(0)[0]) / (cell->vertex(1)[0]-cell->vertex(0)[0]) );
    
    Quadrature<1> quad(scaled_point);

    FEValues<1> fe_values(fe, quad, update_values);
    fe_values.reinit (cell);
    std::vector<Vector<double> > wert (quad.size(),
                                       Vector<double>(dim));
    fe_values.get_function_values(solution, wert);

    values(0) = wert[0](0);
    values(1) = wert[0](1);
    //values(2) = u_(2);
}



template <int dim>
void
BoundaryValues<dim>::get_values_dt (const Point<dim> &p,
                                    Vector<double>   &values) const
{
    Vector<double> u_minus(dim);       // u_t-1
    Vector<double> u(dim);             // u_t
    Vector<double> u_plus(dim);        // u_t+1
    Vector<double> v0(dim);            // u_t - u_t-1
    Vector<double> v1(dim);            // u_t+1 - u_t
    Vector<double> delta_v(dim);       // v1 - v0
    double substep = fmod (timestep, heartinterval)/heartinterval;

    BoundaryValues<dim>::get_heartdelta(p, u_minus, 0);
    BoundaryValues<dim>::get_heartdelta(p, u, 1);
    BoundaryValues<dim>::get_heartdelta(p, u_plus, 2);

    // (u_t - u_t-1) / h
    v0 = u;
    v0 -= u_minus;
    v0 /= heartinterval;
    // (u_t+1 - u_t) / h
    v1 = u_plus;
    v1 -= u;
    v1 /= heartinterval;
    // calculate & scale delta_v
    delta_v = v1;
    delta_v -= v0;
    delta_v *= substep;

    v0 += delta_v; // delta_u = delta_u*substep

    // scaling to achieve convergence
    v0 *= 0.1;
    
    values(0) = v0(0);
    values(1) = v0(1);
    values(2) = v0(0);
    values(3) = v0(1);
    //values(2) = u_(2);
}

// Explicit instantiations
template class BoundaryValues<2>;
