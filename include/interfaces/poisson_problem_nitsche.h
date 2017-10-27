#ifndef _pidoums_poisson_nitsche_h_
#define _pidoums_poisson_nitsche_h_

#include "pde_system_interface.h"
#include <deal2lkit/sacado_tools.h>

template <int dim, int spacedim, typename LAC=LATrilinos>
class PoissonProblemNitsche : public PDESystemInterface<dim,spacedim, PoissonProblemNitsche<dim,spacedim,LAC>, LAC>
{

public:
  ~PoissonProblemNitsche () {};
  PoissonProblemNitsche ();

  virtual void declare_parameters(ParameterHandler &prm);



  virtual UpdateFlags get_face_update_flags() const
  {
    return (update_values             |
            update_gradients          | /* this is the new entry */
            update_quadrature_points  |
            update_normal_vectors     |
            update_JxW_values);
  }



  // interface with the PDESystemInterface :)


  template <typename EnergyType, typename ResidualType>
  void energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                              FEValuesCache<dim,spacedim> &scratch,
                              std::vector<EnergyType> &energies,
                              std::vector<std::vector<ResidualType> > &local_residuals,
                              bool compute_only_system_terms) const;


  void compute_system_operators(const std::vector<shared_ptr<LATrilinos::BlockMatrix> > &,
                                LinearOperator<LATrilinos::VectorType> &,
                                LinearOperator<LATrilinos::VectorType> &,
                                LinearOperator<LATrilinos::VectorType> &) const;

private:
  mutable shared_ptr<TrilinosWrappers::PreconditionJacobi> preconditioner;
  double gamma;
  mutable  ParsedMappedFunctions<spacedim> nitsche;


};

template <int dim, int spacedim, typename LAC>
PoissonProblemNitsche<dim,spacedim, LAC>::
PoissonProblemNitsche():
  PDESystemInterface<dim,spacedim,PoissonProblemNitsche<dim,spacedim,LAC>, LAC >("Poisson problem",
      1,1,
      "FESystem[FE_Q(1)]",
      "u","1"),
  nitsche("Nitsche boundary conditions",
          this->n_components,
          this->get_component_names(),
          "" /* do nothing by default */
         )
{}



template <int dim, int spacedim, typename LAC>
template <typename EnergyType, typename ResidualType>
void
PoissonProblemNitsche<dim,spacedim,LAC>::
energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                       FEValuesCache<dim,spacedim> &fe_cache,
                       std::vector<EnergyType> &,
                       std::vector<std::vector<ResidualType> > &local_residuals,
                       bool compute_only_system_terms) const
{

  const FEValuesExtractors::Scalar s(0);
  double h = cell->diameter();

  ResidualType rt = 0; // dummy number to define the type of variables
  this->reinit (rt, cell, fe_cache);
  auto &uts = fe_cache.get_values("solution_dot", "u", s, rt);
  auto &gradus = fe_cache.get_gradients("solution", "u", s, rt);

  const unsigned int n_q_points = uts.size();
  auto &JxW = fe_cache.get_JxW_values();

  auto &fev = fe_cache.get_current_fe_values();

  for (unsigned int q=0; q<n_q_points; ++q)
    {
      auto &ut = uts[q];
      auto &gradu = gradus[q];
      for (unsigned int i=0; i<local_residuals[0].size(); ++i)
        {
          auto v = fev[s].value(i,q);
          auto gradv = fev[s].gradient(i,q);
          local_residuals[0][i] += (
                                     ut*v
                                     +
                                     gradu*gradv
                                   )*JxW[q];
        }

      (void)compute_only_system_terms;

    }
  /// nitsche bcs

  ResidualType dummy;
  for (unsigned int face=0; face < GeometryInfo<dim>::faces_per_cell; ++face)
    {
      unsigned int face_id = cell->face(face)->boundary_id();
      if (cell->face(face)->at_boundary() && nitsche.acts_on_id(face_id))
        {
          this->reinit(dummy, cell, face, fe_cache);
          auto &gradusf = fe_cache.get_gradients("solution", "u", s, dummy);
          auto &usf = fe_cache.get_values("solution", "u", s, dummy);

          auto &fev = fe_cache.get_current_fe_values();
          auto &q_points = fe_cache.get_quadrature_points();
          auto &JxW = fe_cache.get_JxW_values();

          for (unsigned int q=0; q<q_points.size(); ++q)
            {
              auto &u = usf[q];
              const Tensor<1,spacedim> n = fev.normal_vector(q);
              auto &gradu = gradusf[q];

              // update time for nitsche_bcs
              nitsche.set_time(this->get_current_time());

              // get mapped function acting on this face_id
              Vector<double> u0(this->n_components);
              nitsche.get_mapped_function(face_id)->vector_value(q_points[q], u0);

              for (unsigned int i=0; i<local_residuals[0].size(); ++i)
                {
                  auto v = fev[s].value(i,q);
                  auto gradv = fev[s].gradient(i,q);

                  local_residuals[0][i] += (
                                             -SacadoTools::scalar_product(gradu,n)*v

                                             -(u-u0[0])*(gradv*n)

                                             +(1./(gamma*h))*(u-u0[0])*v

                                           )*JxW[q];


                }
            }// end loop over quadrature points


          break;

        } // endif face->at_boundary

    }// end loop over faces



}


template <int dim, int spacedim, typename LAC>
void
PoissonProblemNitsche<dim,spacedim,LAC>::compute_system_operators(const std::vector<shared_ptr<LATrilinos::BlockMatrix> > &matrices,
    LinearOperator<LATrilinos::VectorType> &system_op,
    LinearOperator<LATrilinos::VectorType> &prec_op,
    LinearOperator<LATrilinos::VectorType> &) const
{

  preconditioner.reset  (new TrilinosWrappers::PreconditionJacobi());
  preconditioner->initialize(matrices[0]->block(0,0));

  auto A  = linear_operator<LATrilinos::VectorType::BlockType>( matrices[0]->block(0,0) );

  LinearOperator<LATrilinos::VectorType::BlockType> P_inv;

  P_inv = linear_operator<LATrilinos::VectorType::BlockType>(matrices[0]->block(0,0), *preconditioner);

  auto P00 = P_inv;

  // ASSEMBLE THE PROBLEM:
  system_op  = block_operator<1, 1, LATrilinos::VectorType>({{
      {{ A }}
    }
  });

  prec_op = block_operator<1, 1, LATrilinos::VectorType>({{
      {{ P00}} ,
    }
  });
}


template <int dim, int spacedim, typename LAC>
void
PoissonProblemNitsche<dim,spacedim,LAC>::declare_parameters (ParameterHandler &prm)
{
  PDESystemInterface<dim,spacedim,PoissonProblemNitsche<dim,spacedim,LAC>,LAC>::declare_parameters(prm);
  this->add_parameter(prm, &gamma, "gamma", "0.1", Patterns::Double(0.0));
}

#endif
