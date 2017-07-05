#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/manifold.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/base/function_lib.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/numerics/fe_field_function.h>

#include <iostream>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <string>

#include "../include/heart_fe.h"

using namespace dealii;

template <int dim, int spacedim>
Heart<dim,spacedim>::Heart() 
  :
  fe (FE_Q<dim>(2), spacedim),
  dof_handler (triangulation)
{}

template <int dim, int spacedim>
Heart<dim,spacedim>::Heart(bool side, const int degree)
  :
  fe (FE_Q<dim>(degree), spacedim),
  dof_handler (triangulation),
  solution(3),
  side(side)
{
  if (side)
  {
    run_side();
  }
  else
  {
    Assert (degree == 1 || degree == 2,
            ExcMessage("Fe degree must either be equal 1 or 2. Sorry for that!\n"
                       "Degree 3 or more is not availiable due to insufficient data"));
    run_bottom();
  }
}

template <int dim, int spacedim>
void Heart<dim,spacedim>::operator ()(unsigned heartstep){
  // set start vector = ...
  // make sure that we operate always on start_vector, startvector+3
}


template <int dim, int spacedim>
void Heart<dim,spacedim>::setup_system()
{
  dof_handler.distribute_dofs(fe);
  Point<dim> direction (1e-5,1e5);
  // lexicographical numbering of the dofs 
  // due to the heart point ordering
  DoFRenumbering::downstream (dof_handler, direction, true);
}

template <int dim, int spacedim>
void Heart<dim,spacedim>::reinit_data()
{
  std::string filename;
  if (side)
  {
    filename = "../source/side_boundary.txt";
  }
  else
  {
    filename =  "../source/bottom_boundary.txt";
  }
  std::fstream in (filename);
  std::string first;
  std::string second;
  std::string third;
  int n_dofs = dof_handler.n_dofs();
  // TODO: 
  // -jump in line heartstep-1
  // -read line heartstep-1 and heartstep
  if (heartstep==0)
  {
    in >> second;
    in >> third;
    //std::getline(in,second);
    for (int i = 1; i < 98; ++i)
    {
      std::getline(in,first);
    }
    in >> first;
  }
  else if (heartstep == 99)
  {
    in >> third;
    for (int i = 1; i < 98; ++i)
    {
      std::getline(in,first);
    }
    in >> first;
    in >> second;
  }
  else
  {
    for (int i = 0; i < heartstep-1; ++i)
    {
      std::getline(in,first);
    }
    in >> first;
    in >> second;
    in >> third;
    //std::getline(in,second);
  }
  // -split into 3675 or 363 pieces
  std::vector<std::vector<std::string> > splitted (3);
  
  boost::split(splitted[0], first, boost::is_any_of(";") );
  boost::split(splitted[1], second, boost::is_any_of(";") );
  boost::split(splitted[2], third, boost::is_any_of(";") );
  //std::cout << "size = " << splitted[0].size() << std::endl;
  for (int line = 0; line < 3; ++line)    
  {
    solution[line].reinit(n_dofs);

    // -write into solution[0] and solution[1]
    for (int column = 0; column < n_dofs; ++column)
    {
      solution[line][column] = std::stod(splitted[line][column]);
      //std::cout << "solution[" << line << "][" << column << "] = value " << solution[line][column] << std::endl;
    }
  }

}

template <int dim, int spacedim>
Point<spacedim> Heart<dim,spacedim>::push_forward(const Point<dim> chartpoint, 
                                                  const int timestep) const
{
  
  dealii::Functions::FEFieldFunction<dim, DoFHandler<dim>, Vector<double> > fe_field(dof_handler, solution[timestep]);
  Vector<double> wert (spacedim);
  fe_field.vector_value (chartpoint, wert);

/* this is what happens inside the FEFieldFunction :)
  auto cell = GridTools::find_active_cell_around_point (dof_handler,
                                                        chartpoint); 
  // identifying vertex orientation
  // redundance for readability !
  Point<2> lower_left  ( cell->vertex(0)[0], cell->vertex(0)[1] );
  Point<2> lower_right ( cell->vertex(1)[0], cell->vertex(1)[1] );
  Point<2> upper_left  ( cell->vertex(2)[0], cell->vertex(2)[1] );

  Point<2> scaled_point ( (chartpoint[0]  - lower_left[0])/ 
                          (lower_right[0] - lower_left[0])    ,
                          (chartpoint[1]  - lower_left[1])/ 
                          (upper_left[1]  - lower_left[1])    );
  // initializing quadrature by scaled point
  Quadrature<dim> quad(scaled_point);
  FEValues<dim> fe_values(fe, quad, update_values);
  fe_values.reinit (cell);

  std::vector<Vector<double> > wert (quad.size(),
                                     Vector<double>(spacedim));
  fe_values.get_function_values(solution[timestep], wert);
*/
  return Point<spacedim> (wert[0], wert[1], wert[2]);

}

template <int dim, int spacedim>
void Heart<dim,spacedim>::run_side()
{
  std::vector<unsigned int> subdivisions(2);
  subdivisions[0] = 48/fe.degree;
  subdivisions[1] = 24/fe.degree;
  const Point<dim> p1 (0,-4.3196);
  const Point<dim> p2 (2*numbers::PI,1.6838);

  GridGenerator::subdivided_hyper_rectangle(triangulation, 
                                            subdivisions, 
                                            p1, p2, false);
  setup_system();
  reinit_data();
}

template <int dim, int spacedim>
void Heart<dim,spacedim>::run_bottom()
{
  std::vector<unsigned int> subdivisions(2);
  subdivisions[0] = 10/fe.degree;
  subdivisions[1] = 10/fe.degree;
  const Point<dim> p1 (-1.3858, -1.3858);
  const Point<dim> p2 ( 1.3858,  1.3858);

  GridGenerator::subdivided_hyper_rectangle(triangulation, 
                                            subdivisions, 
                                            p1, p2, false);
  setup_system();
  reinit_data();
}

// Explicit instantiations
template class Heart<2,3>;
