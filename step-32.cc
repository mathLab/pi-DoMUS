/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2008 - 2014 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Authors: Martin Kronbichler, Uppsala University,
 *          Wolfgang Bangerth, Texas A&M University,
 *          Timo Heister, University of Goettingen, 2008-2011
 */



#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/trilinos_block_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <limits>
#include <locale>
#include <string>

#include <deal.II/distributed/solution_transfer.h>

#include <deal.II/base/index_set.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>


namespace Step32
{
  using namespace dealii;


  namespace EquationData
  {
    const double eta                   = 1e21;    /* Pa s       */
    const double kappa                 = 1e-6;    /* m^2 / s    */
    const double density     = 3300;    /* kg / m^3   */
    const double nu  = 1; // Added in review
    const double reference_temperature = 293;     /* K          */
    const double expansion_coefficient = 2e-5;    /* 1/K        */
    const double specific_heat         = 1250;    /* J / K / kg */
    const double radiogenic_heating    = 7.4e-12; /* W / kg     */

  
    const double R0      = 6371000.-2890000.;     /* m          */
    const double R1      = 6371000.-  35000.;     /* m          */

    const double T0      = 4000+273;              /* K          */
    const double T1      =  700+273;              /* K          */


    // double density (const double temperature)
    // {
    //   return (reference_density *
    //           (1 - expansion_coefficient * (temperature -
    //                                         reference_temperature)));
    // }


    template <int dim>
    Tensor<1,dim> gravity_vector (const Point<dim> &p)
    {
      const double r = p.norm();
      return -(1.245e-6 * r + 7.714e13/r/r) * p / r;
    }



    template <int dim>
    class TemperatureInitialValues : public Function<dim>
    {
    public:
      TemperatureInitialValues () : Function<dim>(1) {}

      virtual double value (const Point<dim>   &p,
                            const unsigned int  component = 0) const;

      virtual void vector_value (const Point<dim> &p,
                                 Vector<double>   &value) const;
    };



    template <int dim>
    double
    TemperatureInitialValues<dim>::value (const Point<dim>  &p,
                                          const unsigned int) const
    {
      const double r = p.norm();
      const double h = R1-R0;

      const double s = (r-R0)/h;
      const double q = (dim==3)?std::max(0.0,cos(numbers::PI*abs(p(2)/R1))):1.0;
      const double phi   = std::atan2(p(0),p(1));
      const double tau = s
                         +
                         0.2 * s * (1-s) * std::sin(6*phi) * q;

      return T0*(1.0-tau) + T1*tau;
    }


    template <int dim>
    void
    TemperatureInitialValues<dim>::vector_value (const Point<dim> &p,
                                                 Vector<double>   &values) const
    {
      for (unsigned int c=0; c<this->n_components; ++c)
        values(c) = TemperatureInitialValues<dim>::value (p, c);
    }


    const double pressure_scaling = eta / 10000;

    const double year_in_seconds  = 60*60*24*365.2425;

  }




  namespace LinearSolvers
  {
    template <class PreconditionerA, class PreconditionerMp>
    class BlockSchurPreconditioner : public Subscriptor
    {
    public:
      BlockSchurPreconditioner (const TrilinosWrappers::BlockSparseMatrix  &S,
                                const TrilinosWrappers::BlockSparseMatrix  &Spre,
                                const PreconditionerMp                     &Mppreconditioner,
                                const PreconditionerA                      &Apreconditioner,
                                const bool                                  do_solve_A)
        :
        stokes_matrix     (&S),
        stokes_preconditioner_matrix     (&Spre),
        mp_preconditioner (Mppreconditioner),
        a_preconditioner  (Apreconditioner),
        do_solve_A        (do_solve_A)
      {}

      void vmult (TrilinosWrappers::MPI::BlockVector       &dst,
                  const TrilinosWrappers::MPI::BlockVector &src) const
      {
        TrilinosWrappers::MPI::Vector utmp(src.block(0));

        {
          SolverControl solver_control(5000, 1e-6 * src.block(1).l2_norm());

          SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);

          solver.solve(stokes_preconditioner_matrix->block(1,1),
                       dst.block(1), src.block(1),
                       mp_preconditioner);

          dst.block(1) *= -1.0;
        }

        {
          stokes_matrix->block(0,1).vmult(utmp, dst.block(1));
          utmp*=-1.0;
          utmp.add(src.block(0));
        }

        if (do_solve_A == true)
          {
            SolverControl solver_control(5000, utmp.l2_norm()*1e-2);
            TrilinosWrappers::SolverCG solver(solver_control);
            solver.solve(stokes_matrix->block(0,0), dst.block(0), utmp,
                         a_preconditioner);
          }
        else
          a_preconditioner.vmult (dst.block(0), utmp);
      }

    private:
      const SmartPointer<const TrilinosWrappers::BlockSparseMatrix> stokes_matrix;
      const SmartPointer<const TrilinosWrappers::BlockSparseMatrix> stokes_preconditioner_matrix;
      const PreconditionerMp &mp_preconditioner;
      const PreconditionerA  &a_preconditioner;
      const bool do_solve_A;
    };
  }



  namespace Assembly
  {
    namespace Scratch
    {
      template <int dim>
      struct StokesPreconditioner
      {
        StokesPreconditioner (const FiniteElement<dim> &stokes_fe,
                              const Quadrature<dim>    &stokes_quadrature,
                              const Mapping<dim>       &mapping,
                              const UpdateFlags         update_flags);

        StokesPreconditioner (const StokesPreconditioner &data);


        FEValues<dim>               stokes_fe_values;

        std::vector<Tensor<2,dim> > grad_phi_u;
        std::vector<double>         phi_p;
      };

      template <int dim>
      StokesPreconditioner<dim>::
      StokesPreconditioner (const FiniteElement<dim> &stokes_fe,
                            const Quadrature<dim>    &stokes_quadrature,
                            const Mapping<dim>       &mapping,
                            const UpdateFlags         update_flags)
        :
        stokes_fe_values (mapping, stokes_fe, stokes_quadrature,
                          update_flags),
        grad_phi_u (stokes_fe.dofs_per_cell),
        phi_p (stokes_fe.dofs_per_cell)
      {}



      template <int dim>
      StokesPreconditioner<dim>::
      StokesPreconditioner (const StokesPreconditioner &scratch)
        :
        stokes_fe_values (scratch.stokes_fe_values.get_mapping(),
                          scratch.stokes_fe_values.get_fe(),
                          scratch.stokes_fe_values.get_quadrature(),
                          scratch.stokes_fe_values.get_update_flags()),
        grad_phi_u (scratch.grad_phi_u),
        phi_p (scratch.phi_p)
      {}



      template <int dim>
      struct StokesSystem : public StokesPreconditioner<dim>
      {
        StokesSystem (const FiniteElement<dim> &stokes_fe,
                      const Mapping<dim>       &mapping,
                      const Quadrature<dim>    &stokes_quadrature,
                      const UpdateFlags         stokes_update_flags,
                      const FiniteElement<dim> &temperature_fe,
                      const UpdateFlags         temperature_update_flags);

        StokesSystem (const StokesSystem<dim> &data);


        FEValues<dim>                        temperature_fe_values;

        std::vector<Tensor<1,dim> >          phi_u;
        std::vector<SymmetricTensor<2,dim> > grads_phi_u;
        std::vector<double>                  div_phi_u;

        std::vector<double>                  old_temperature_values;
      };


      template <int dim>
      StokesSystem<dim>::
      StokesSystem (const FiniteElement<dim> &stokes_fe,
                    const Mapping<dim>       &mapping,
                    const Quadrature<dim>    &stokes_quadrature,
                    const UpdateFlags         stokes_update_flags,
                    const FiniteElement<dim> &temperature_fe,
                    const UpdateFlags         temperature_update_flags)
        :
        StokesPreconditioner<dim> (stokes_fe, stokes_quadrature,
                                   mapping,
                                   stokes_update_flags),
        temperature_fe_values (mapping, temperature_fe, stokes_quadrature,
                               temperature_update_flags),
        phi_u (stokes_fe.dofs_per_cell),
        grads_phi_u (stokes_fe.dofs_per_cell),
        div_phi_u (stokes_fe.dofs_per_cell),
        old_temperature_values (stokes_quadrature.size())
      {}


      template <int dim>
      StokesSystem<dim>::
      StokesSystem (const StokesSystem<dim> &scratch)
        :
        StokesPreconditioner<dim> (scratch),
        temperature_fe_values (scratch.temperature_fe_values.get_mapping(),
                               scratch.temperature_fe_values.get_fe(),
                               scratch.temperature_fe_values.get_quadrature(),
                               scratch.temperature_fe_values.get_update_flags()),
        phi_u (scratch.phi_u),
        grads_phi_u (scratch.grads_phi_u),
        div_phi_u (scratch.div_phi_u),
        old_temperature_values (scratch.old_temperature_values)
      {}


      template <int dim>
      struct TemperatureMatrix
      {
        TemperatureMatrix (const FiniteElement<dim> &temperature_fe,
                           const Mapping<dim>       &mapping,
                           const Quadrature<dim>    &temperature_quadrature);

        TemperatureMatrix (const TemperatureMatrix &data);


        FEValues<dim>               temperature_fe_values;

        std::vector<double>         phi_T;
        std::vector<Tensor<1,dim> > grad_phi_T;
      };


      template <int dim>
      TemperatureMatrix<dim>::
      TemperatureMatrix (const FiniteElement<dim> &temperature_fe,
                         const Mapping<dim>       &mapping,
                         const Quadrature<dim>    &temperature_quadrature)
        :
        temperature_fe_values (mapping,
                               temperature_fe, temperature_quadrature,
                               update_values    | update_gradients |
                               update_JxW_values),
        phi_T (temperature_fe.dofs_per_cell),
        grad_phi_T (temperature_fe.dofs_per_cell)
      {}


      template <int dim>
      TemperatureMatrix<dim>::
      TemperatureMatrix (const TemperatureMatrix &scratch)
        :
        temperature_fe_values (scratch.temperature_fe_values.get_mapping(),
                               scratch.temperature_fe_values.get_fe(),
                               scratch.temperature_fe_values.get_quadrature(),
                               scratch.temperature_fe_values.get_update_flags()),
        phi_T (scratch.phi_T),
        grad_phi_T (scratch.grad_phi_T)
      {}


      template <int dim>
      struct TemperatureRHS
      {
        TemperatureRHS (const FiniteElement<dim> &temperature_fe,
                        const FiniteElement<dim> &stokes_fe,
                        const Mapping<dim>       &mapping,
                        const Quadrature<dim>    &quadrature);

        TemperatureRHS (const TemperatureRHS &data);


        FEValues<dim>                        temperature_fe_values;
        FEValues<dim>                        stokes_fe_values;

        std::vector<double>                  phi_T;
        std::vector<Tensor<1,dim> >          grad_phi_T;

        std::vector<Tensor<1,dim> >          old_velocity_values;
        std::vector<Tensor<1,dim> >          old_old_velocity_values;

        std::vector<SymmetricTensor<2,dim> > old_strain_rates;
        std::vector<SymmetricTensor<2,dim> > old_old_strain_rates;

        std::vector<double>                  old_temperature_values;
        std::vector<double>                  old_old_temperature_values;
        std::vector<Tensor<1,dim> >          old_temperature_grads;
        std::vector<Tensor<1,dim> >          old_old_temperature_grads;
        std::vector<double>                  old_temperature_laplacians;
        std::vector<double>                  old_old_temperature_laplacians;
      };


      template <int dim>
      TemperatureRHS<dim>::
      TemperatureRHS (const FiniteElement<dim> &temperature_fe,
                      const FiniteElement<dim> &stokes_fe,
                      const Mapping<dim>       &mapping,
                      const Quadrature<dim>    &quadrature)
        :
        temperature_fe_values (mapping,
                               temperature_fe, quadrature,
                               update_values    |
                               update_gradients |
                               update_hessians  |
                               update_quadrature_points |
                               update_JxW_values),
        stokes_fe_values (mapping,
                          stokes_fe, quadrature,
                          update_values | update_gradients),
        phi_T (temperature_fe.dofs_per_cell),
        grad_phi_T (temperature_fe.dofs_per_cell),

        old_velocity_values (quadrature.size()),
        old_old_velocity_values (quadrature.size()),
        old_strain_rates (quadrature.size()),
        old_old_strain_rates (quadrature.size()),

        old_temperature_values (quadrature.size()),
        old_old_temperature_values(quadrature.size()),
        old_temperature_grads(quadrature.size()),
        old_old_temperature_grads(quadrature.size()),
        old_temperature_laplacians(quadrature.size()),
        old_old_temperature_laplacians(quadrature.size())
      {}


      template <int dim>
      TemperatureRHS<dim>::
      TemperatureRHS (const TemperatureRHS &scratch)
        :
        temperature_fe_values (scratch.temperature_fe_values.get_mapping(),
                               scratch.temperature_fe_values.get_fe(),
                               scratch.temperature_fe_values.get_quadrature(),
                               scratch.temperature_fe_values.get_update_flags()),
        stokes_fe_values (scratch.stokes_fe_values.get_mapping(),
                          scratch.stokes_fe_values.get_fe(),
                          scratch.stokes_fe_values.get_quadrature(),
                          scratch.stokes_fe_values.get_update_flags()),
        phi_T (scratch.phi_T),
        grad_phi_T (scratch.grad_phi_T),

        old_velocity_values (scratch.old_velocity_values),
        old_old_velocity_values (scratch.old_old_velocity_values),
        old_strain_rates (scratch.old_strain_rates),
        old_old_strain_rates (scratch.old_old_strain_rates),

        old_temperature_values (scratch.old_temperature_values),
        old_old_temperature_values (scratch.old_old_temperature_values),
        old_temperature_grads (scratch.old_temperature_grads),
        old_old_temperature_grads (scratch.old_old_temperature_grads),
        old_temperature_laplacians (scratch.old_temperature_laplacians),
        old_old_temperature_laplacians (scratch.old_old_temperature_laplacians)
      {}
    }


    namespace CopyData
    {
      template <int dim>
      struct StokesPreconditioner
      {
        StokesPreconditioner (const FiniteElement<dim> &stokes_fe);
        StokesPreconditioner (const StokesPreconditioner &data);

        FullMatrix<double>          local_matrix;
        std::vector<types::global_dof_index> local_dof_indices;
      };

      template <int dim>
      StokesPreconditioner<dim>::
      StokesPreconditioner (const FiniteElement<dim> &stokes_fe)
        :
        local_matrix (stokes_fe.dofs_per_cell,
                      stokes_fe.dofs_per_cell),
        local_dof_indices (stokes_fe.dofs_per_cell)
      {}

      template <int dim>
      StokesPreconditioner<dim>::
      StokesPreconditioner (const StokesPreconditioner &data)
        :
        local_matrix (data.local_matrix),
        local_dof_indices (data.local_dof_indices)
      {}



      template <int dim>
      struct StokesSystem : public StokesPreconditioner<dim>
      {
        StokesSystem (const FiniteElement<dim> &stokes_fe);
        StokesSystem (const StokesSystem<dim> &data);

        Vector<double> local_rhs;
      };

      template <int dim>
      StokesSystem<dim>::
      StokesSystem (const FiniteElement<dim> &stokes_fe)
        :
        StokesPreconditioner<dim> (stokes_fe),
        local_rhs (stokes_fe.dofs_per_cell)
      {}

      template <int dim>
      StokesSystem<dim>::
      StokesSystem (const StokesSystem<dim> &data)
        :
        StokesPreconditioner<dim> (data),
        local_rhs (data.local_rhs)
      {}



      template <int dim>
      struct TemperatureMatrix
      {
        TemperatureMatrix (const FiniteElement<dim> &temperature_fe);
        TemperatureMatrix (const TemperatureMatrix &data);

        FullMatrix<double>          local_mass_matrix;
        FullMatrix<double>          local_stiffness_matrix;
        std::vector<types::global_dof_index>   local_dof_indices;
      };

      template <int dim>
      TemperatureMatrix<dim>::
      TemperatureMatrix (const FiniteElement<dim> &temperature_fe)
        :
        local_mass_matrix (temperature_fe.dofs_per_cell,
                           temperature_fe.dofs_per_cell),
        local_stiffness_matrix (temperature_fe.dofs_per_cell,
                                temperature_fe.dofs_per_cell),
        local_dof_indices (temperature_fe.dofs_per_cell)
      {}

      template <int dim>
      TemperatureMatrix<dim>::
      TemperatureMatrix (const TemperatureMatrix &data)
        :
        local_mass_matrix (data.local_mass_matrix),
        local_stiffness_matrix (data.local_stiffness_matrix),
        local_dof_indices (data.local_dof_indices)
      {}



      template <int dim>
      struct TemperatureRHS
      {
        TemperatureRHS (const FiniteElement<dim> &temperature_fe);
        TemperatureRHS (const TemperatureRHS &data);

        Vector<double>              local_rhs;
        std::vector<types::global_dof_index> local_dof_indices;
        FullMatrix<double>          matrix_for_bc;
      };

      template <int dim>
      TemperatureRHS<dim>::
      TemperatureRHS (const FiniteElement<dim> &temperature_fe)
        :
        local_rhs (temperature_fe.dofs_per_cell),
        local_dof_indices (temperature_fe.dofs_per_cell),
        matrix_for_bc (temperature_fe.dofs_per_cell,
                       temperature_fe.dofs_per_cell)
      {}

      template <int dim>
      TemperatureRHS<dim>::
      TemperatureRHS (const TemperatureRHS &data)
        :
        local_rhs (data.local_rhs),
        local_dof_indices (data.local_dof_indices),
        matrix_for_bc (data.matrix_for_bc)
      {}
    }
  }



  template <int dim>
  class BoussinesqFlowProblem
  {
  public:
    struct Parameters;
    BoussinesqFlowProblem (Parameters &parameters);
    void run ();

  private:
    void setup_dofs ();
    void assemble_stokes_preconditioner ();
    void build_stokes_preconditioner ();
    void assemble_stokes_system ();
    void assemble_temperature_matrix ();
    void assemble_temperature_system (const double maximal_velocity);
    void project_temperature_field ();
    double get_maximal_velocity () const;
    double get_cfl_number () const;
    double get_entropy_variation (const double average_temperature) const;
    std::pair<double,double> get_extrapolated_temperature_range () const;
    void solve ();
    void output_results ();
    void refine_mesh (const unsigned int max_grid_level);

    // double
    // compute_viscosity(const std::vector<double>          &old_temperature,
    //                   const std::vector<double>          &old_old_temperature,
    //                   const std::vector<Tensor<1,dim> >  &old_temperature_grads,
    //                   const std::vector<Tensor<1,dim> >  &old_old_temperature_grads,
    //                   const std::vector<double>          &old_temperature_laplacians,
    //                   const std::vector<double>          &old_old_temperature_laplacians,
    //                   const std::vector<Tensor<1,dim> >  &old_velocity_values,
    //                   const std::vector<Tensor<1,dim> >  &old_old_velocity_values,
    //                   const std::vector<SymmetricTensor<2,dim> >  &old_strain_rates,
    //                   const std::vector<SymmetricTensor<2,dim> >  &old_old_strain_rates,
    //                   const double                        global_u_infty,
    //                   const double                        global_T_variation,
    //                   const double                        average_temperature,
    //                   const double                        global_entropy_variation,
    //                   const double                        cell_diameter) const;

  public:

    struct Parameters
    {
      Parameters (const std::string &parameter_filename);

      static void declare_parameters (ParameterHandler &prm);
      void parse_parameters (ParameterHandler &prm);

      double       end_time;

      unsigned int initial_global_refinement;
      unsigned int initial_adaptive_refinement;

      bool         generate_graphical_output;
      unsigned int graphical_output_interval;

      unsigned int adaptive_refinement_interval;

      double       stabilization_alpha;
      double       stabilization_c_R;
      double       stabilization_beta;

      unsigned int stokes_velocity_degree;
      bool         use_locally_conservative_discretization;

      unsigned int temperature_degree;
    };

  private:
    Parameters                               &parameters;

    ConditionalOStream                        pcout;

    parallel::distributed::Triangulation<dim> triangulation;
    double                                    global_Omega_diameter;

    const MappingQ<dim>                       mapping;

    const FESystem<dim>                       stokes_fe;
    DoFHandler<dim>                           stokes_dof_handler;
    ConstraintMatrix                          stokes_constraints;

    TrilinosWrappers::BlockSparseMatrix       stokes_matrix;
    TrilinosWrappers::BlockSparseMatrix       stokes_preconditioner_matrix;

    TrilinosWrappers::MPI::BlockVector        stokes_solution;
    TrilinosWrappers::MPI::BlockVector        old_stokes_solution;
    TrilinosWrappers::MPI::BlockVector        stokes_rhs;


    FE_Q<dim>                                 temperature_fe;
    DoFHandler<dim>                           temperature_dof_handler;
    ConstraintMatrix                          temperature_constraints;

    TrilinosWrappers::SparseMatrix            temperature_mass_matrix;
    TrilinosWrappers::SparseMatrix            temperature_stiffness_matrix;
    TrilinosWrappers::SparseMatrix            temperature_matrix;

    TrilinosWrappers::MPI::Vector             temperature_solution;
    TrilinosWrappers::MPI::Vector             old_temperature_solution;
    TrilinosWrappers::MPI::Vector             old_old_temperature_solution;
    TrilinosWrappers::MPI::Vector             temperature_rhs;


    double                                    time_step;
    double                                    old_time_step;
    unsigned int                              timestep_number;

    std_cxx11::shared_ptr<TrilinosWrappers::PreconditionAMG>    Amg_preconditioner;
    std_cxx11::shared_ptr<TrilinosWrappers::PreconditionJacobi> Mp_preconditioner;
    std_cxx11::shared_ptr<TrilinosWrappers::PreconditionJacobi> T_preconditioner;

    bool                                      rebuild_stokes_matrix;
    bool                                      rebuild_stokes_preconditioner;
    bool                                      rebuild_temperature_matrices;
    bool                                      rebuild_temperature_preconditioner;

    TimerOutput                               computing_timer;

    void setup_stokes_matrix (const std::vector<IndexSet> &stokes_partitioning,
                              const std::vector<IndexSet> &stokes_relevant_partitioning);
    void setup_stokes_preconditioner (const std::vector<IndexSet> &stokes_partitioning,
                                      const std::vector<IndexSet> &stokes_relevant_partitioning);
    void setup_temperature_matrices (const IndexSet &temperature_partitioning,
                                     const IndexSet &temperature_relevant_partitioning);


    void
    local_assemble_stokes_preconditioner (const typename DoFHandler<dim>::active_cell_iterator &cell,
                                          Assembly::Scratch::StokesPreconditioner<dim> &scratch,
                                          Assembly::CopyData::StokesPreconditioner<dim> &data);

    void
    copy_local_to_global_stokes_preconditioner (const Assembly::CopyData::StokesPreconditioner<dim> &data);


    void
    local_assemble_stokes_system (const typename DoFHandler<dim>::active_cell_iterator &cell,
                                  Assembly::Scratch::StokesSystem<dim>  &scratch,
                                  Assembly::CopyData::StokesSystem<dim> &data);

    void
    copy_local_to_global_stokes_system (const Assembly::CopyData::StokesSystem<dim> &data);


    void
    local_assemble_temperature_matrix (const typename DoFHandler<dim>::active_cell_iterator &cell,
                                       Assembly::Scratch::TemperatureMatrix<dim>  &scratch,
                                       Assembly::CopyData::TemperatureMatrix<dim> &data);

    void
    copy_local_to_global_temperature_matrix (const Assembly::CopyData::TemperatureMatrix<dim> &data);



    void
    local_assemble_temperature_rhs (const std::pair<double,double> global_T_range,
                                    const double                   global_max_velocity,
                                    const double                   global_entropy_variation,
                                    const typename DoFHandler<dim>::active_cell_iterator &cell,
                                    Assembly::Scratch::TemperatureRHS<dim> &scratch,
                                    Assembly::CopyData::TemperatureRHS<dim> &data);

    void
    copy_local_to_global_temperature_rhs (const Assembly::CopyData::TemperatureRHS<dim> &data);

    class Postprocessor;
  };



  template <int dim>
  BoussinesqFlowProblem<dim>::Parameters::Parameters (const std::string &parameter_filename)
    :
    end_time (1e8),
    initial_global_refinement (2),
    initial_adaptive_refinement (2),
    adaptive_refinement_interval (10),
    stabilization_alpha (2),
    stabilization_c_R (0.11),
    stabilization_beta (0.078),
    stokes_velocity_degree (2),
    use_locally_conservative_discretization (true),
    temperature_degree (2)
  {
    ParameterHandler prm;
    BoussinesqFlowProblem<dim>::Parameters::declare_parameters (prm);

    std::ifstream parameter_file (parameter_filename.c_str());

    if (!parameter_file)
      {
        parameter_file.close ();

        std::ostringstream message;
        message << "Input parameter file <"
                << parameter_filename << "> not found. Creating a"
                << std::endl
                << "template file of the same name."
                << std::endl;

        std::ofstream parameter_out (parameter_filename.c_str());
        prm.print_parameters (parameter_out,
                              ParameterHandler::Text);

        AssertThrow (false, ExcMessage (message.str().c_str()));
      }

    const bool success = prm.read_input (parameter_file);
    AssertThrow (success, ExcMessage ("Invalid input parameter file."));

    parse_parameters (prm);
  }



  template <int dim>
  void
  BoussinesqFlowProblem<dim>::Parameters::
  declare_parameters (ParameterHandler &prm)
  {
    prm.declare_entry ("End time", "1e8",
                       Patterns::Double (0),
                       "The end time of the simulation in years.");
    prm.declare_entry ("Initial global refinement", "2",
                       Patterns::Integer (0),
                       "The number of global refinement steps performed on "
                       "the initial coarse mesh, before the problem is first "
                       "solved there.");
    prm.declare_entry ("Initial adaptive refinement", "2",
                       Patterns::Integer (0),
                       "The number of adaptive refinement steps performed after "
                       "initial global refinement.");
    prm.declare_entry ("Time steps between mesh refinement", "10",
                       Patterns::Integer (1),
                       "The number of time steps after which the mesh is to be "
                       "adapted based on computed error indicators.");
    prm.declare_entry ("Generate graphical output", "false",
                       Patterns::Bool (),
                       "Whether graphical output is to be generated or not. "
                       "You may not want to get graphical output if the number "
                       "of processors is large.");
    prm.declare_entry ("Time steps between graphical output", "50",
                       Patterns::Integer (1),
                       "The number of time steps between each generation of "
                       "graphical output files.");

    prm.enter_subsection ("Stabilization parameters");
    {
      prm.declare_entry ("alpha", "2",
                         Patterns::Double (1, 2),
                         "The exponent in the entropy viscosity stabilization.");
      prm.declare_entry ("c_R", "0.11",
                         Patterns::Double (0),
                         "The c_R factor in the entropy viscosity "
                         "stabilization.");
      prm.declare_entry ("beta", "0.078",
                         Patterns::Double (0),
                         "The beta factor in the artificial viscosity "
                         "stabilization. An appropriate value for 2d is 0.052 "
                         "and 0.078 for 3d.");
    }
    prm.leave_subsection ();

    prm.enter_subsection ("Discretization");
    {
      prm.declare_entry ("Stokes velocity polynomial degree", "2",
                         Patterns::Integer (1),
                         "The polynomial degree to use for the velocity variables "
                         "in the Stokes system.");
      prm.declare_entry ("Temperature polynomial degree", "2",
                         Patterns::Integer (1),
                         "The polynomial degree to use for the temperature variable.");
      prm.declare_entry ("Use locally conservative discretization", "true",
                         Patterns::Bool (),
                         "Whether to use a Stokes discretization that is locally "
                         "conservative at the expense of a larger number of degrees "
                         "of freedom, or to go with a cheaper discretization "
                         "that does not locally conserve mass (although it is "
                         "globally conservative.");
    }
    prm.leave_subsection ();
  }



  template <int dim>
  void
  BoussinesqFlowProblem<dim>::Parameters::
  parse_parameters (ParameterHandler &prm)
  {
    end_time                    = prm.get_double ("End time");
    initial_global_refinement   = prm.get_integer ("Initial global refinement");
    initial_adaptive_refinement = prm.get_integer ("Initial adaptive refinement");

    adaptive_refinement_interval= prm.get_integer ("Time steps between mesh refinement");

    generate_graphical_output   = prm.get_bool ("Generate graphical output");
    graphical_output_interval   = prm.get_integer ("Time steps between graphical output");

    prm.enter_subsection ("Stabilization parameters");
    {
      stabilization_alpha = prm.get_double ("alpha");
      stabilization_c_R   = prm.get_double ("c_R");
      stabilization_beta  = prm.get_double ("beta");
    }
    prm.leave_subsection ();

    prm.enter_subsection ("Discretization");
    {
      stokes_velocity_degree = prm.get_integer ("Stokes velocity polynomial degree");
      temperature_degree     = prm.get_integer ("Temperature polynomial degree");
      use_locally_conservative_discretization
        = prm.get_bool ("Use locally conservative discretization");
    }
    prm.leave_subsection ();
  }




  template <int dim>
  BoussinesqFlowProblem<dim>::BoussinesqFlowProblem (Parameters &parameters_)
    :
    parameters (parameters_),
    pcout (std::cout,
           (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
            == 0)),

    triangulation (MPI_COMM_WORLD,
                   typename Triangulation<dim>::MeshSmoothing
                   (Triangulation<dim>::smoothing_on_refinement |
                    Triangulation<dim>::smoothing_on_coarsening)),

    mapping (4),

    stokes_fe (FE_Q<dim>(parameters.stokes_velocity_degree),
               dim,
               (parameters.use_locally_conservative_discretization
                ?
                static_cast<const FiniteElement<dim> &>
                (FE_DGP<dim>(parameters.stokes_velocity_degree-1))
                :
                static_cast<const FiniteElement<dim> &>
                (FE_Q<dim>(parameters.stokes_velocity_degree-1))),
               1),

    stokes_dof_handler (triangulation),

    temperature_fe (parameters.temperature_degree),
    temperature_dof_handler (triangulation),

    time_step (0),
    old_time_step (0),
    timestep_number (0),
    rebuild_stokes_matrix (true),
    rebuild_stokes_preconditioner (true),
    rebuild_temperature_matrices (true),
    rebuild_temperature_preconditioner (true),

    computing_timer (MPI_COMM_WORLD,
                     pcout,
                     TimerOutput::summary,
                     TimerOutput::wall_times)
  {}




  template <int dim>
  double BoussinesqFlowProblem<dim>::get_maximal_velocity () const
  {
    // const QIterated<dim> quadrature_formula (QTrapez<1>(),
                                            //  parameters.stokes_velocity_degree);
    // const unsigned int n_q_points = quadrature_formula.size();

    // FEValues<dim> fe_values (mapping, stokes_fe, quadrature_formula, update_values);
    // std::vector<Tensor<1,dim> > velocity_values(n_q_points);

    // const FEValuesExtractors::Vector velocities (0);

    // double max_local_velocity = 0;

    // typename DoFHandler<dim>::active_cell_iterator
    // cell = stokes_dof_handler.begin_active(),
    // endc = stokes_dof_handler.end();
    // for (; cell!=endc; ++cell)
    //   if (cell->is_locally_owned())
    //     {
    //       fe_values.reinit (cell);
    //       fe_values[velocities].get_function_values (stokes_solution,
    //                                                  velocity_values);
    // 
    //       for (unsigned int q=0; q<n_q_points; ++q)
    //         max_local_velocity = std::max (max_local_velocity,
    //                                        velocity_values[q].norm());
        // }

    // return Utilities::MPI::max (max_local_velocity, MPI_COMM_WORLD);
    return 1.;
  }



  template <int dim>
  double BoussinesqFlowProblem<dim>::get_cfl_number () const
  {
    const QIterated<dim> quadrature_formula (QTrapez<1>(),
                                             parameters.stokes_velocity_degree);
    const unsigned int n_q_points = quadrature_formula.size();

    FEValues<dim> fe_values (mapping, stokes_fe, quadrature_formula, update_values);
    std::vector<Tensor<1,dim> > velocity_values(n_q_points);

    const FEValuesExtractors::Vector velocities (0);

    double max_local_cfl = 0;

    typename DoFHandler<dim>::active_cell_iterator
    cell = stokes_dof_handler.begin_active(),
    endc = stokes_dof_handler.end();
    for (; cell!=endc; ++cell)
      if (cell->is_locally_owned())
        {
          fe_values.reinit (cell);
          fe_values[velocities].get_function_values (stokes_solution,
                                                     velocity_values);

          double max_local_velocity = 1e-10;
          for (unsigned int q=0; q<n_q_points; ++q)
            max_local_velocity = std::max (max_local_velocity,
                                           velocity_values[q].norm());
          max_local_cfl = std::max(max_local_cfl,
                                   max_local_velocity / cell->diameter());
        }

    return Utilities::MPI::max (max_local_cfl, MPI_COMM_WORLD);
  }



  template <int dim>
  double
  BoussinesqFlowProblem<dim>::get_entropy_variation (const double average_temperature) const
  {
    if (parameters.stabilization_alpha != 2)
      return 1.;

    const QGauss<dim> quadrature_formula (parameters.temperature_degree+1);
    const unsigned int n_q_points = quadrature_formula.size();

    FEValues<dim> fe_values (temperature_fe, quadrature_formula,
                             update_values | update_JxW_values);
    std::vector<double> old_temperature_values(n_q_points);
    std::vector<double> old_old_temperature_values(n_q_points);

    double min_entropy = std::numeric_limits<double>::max(),
           max_entropy = -std::numeric_limits<double>::max(),
           area = 0,
           entropy_integrated = 0;

    typename DoFHandler<dim>::active_cell_iterator
    cell = temperature_dof_handler.begin_active(),
    endc = temperature_dof_handler.end();
    for (; cell!=endc; ++cell)
      if (cell->is_locally_owned())
        {
          fe_values.reinit (cell);
          fe_values.get_function_values (old_temperature_solution,
                                         old_temperature_values);
          fe_values.get_function_values (old_old_temperature_solution,
                                         old_old_temperature_values);
          for (unsigned int q=0; q<n_q_points; ++q)
            {
              const double T = (old_temperature_values[q] +
                                old_old_temperature_values[q]) / 2;
              const double entropy = ((T-average_temperature) *
                                      (T-average_temperature));

              min_entropy = std::min (min_entropy, entropy);
              max_entropy = std::max (max_entropy, entropy);
              area += fe_values.JxW(q);
              entropy_integrated += fe_values.JxW(q) * entropy;
            }
        }

    const double local_sums[2]   = { entropy_integrated, area },
                                   local_maxima[2] = { -min_entropy, max_entropy };
    double global_sums[2], global_maxima[2];

    Utilities::MPI::sum (local_sums,   MPI_COMM_WORLD, global_sums);
    Utilities::MPI::max (local_maxima, MPI_COMM_WORLD, global_maxima);

    const double average_entropy = global_sums[0] / global_sums[1];
    const double entropy_diff = std::max(global_maxima[1] - average_entropy,
                                         average_entropy - (-global_maxima[0]));
    return entropy_diff;
  }




  template <int dim>
  std::pair<double,double>
  BoussinesqFlowProblem<dim>::get_extrapolated_temperature_range () const
  {
    const QIterated<dim> quadrature_formula (QTrapez<1>(),
                                             parameters.temperature_degree);
    const unsigned int n_q_points = quadrature_formula.size();

    FEValues<dim> fe_values (mapping, temperature_fe, quadrature_formula,
                             update_values);
    std::vector<double> old_temperature_values(n_q_points);
    std::vector<double> old_old_temperature_values(n_q_points);

    double min_local_temperature = std::numeric_limits<double>::max(),
           max_local_temperature = -std::numeric_limits<double>::max();

    if (timestep_number != 0)
      {
        typename DoFHandler<dim>::active_cell_iterator
        cell = temperature_dof_handler.begin_active(),
        endc = temperature_dof_handler.end();
        for (; cell!=endc; ++cell)
          if (cell->is_locally_owned())
            {
              fe_values.reinit (cell);
              fe_values.get_function_values (old_temperature_solution,
                                             old_temperature_values);
              fe_values.get_function_values (old_old_temperature_solution,
                                             old_old_temperature_values);

              for (unsigned int q=0; q<n_q_points; ++q)
                {
                  const double temperature =
                    (1. + time_step/old_time_step) * old_temperature_values[q]-
                    time_step/old_time_step * old_old_temperature_values[q];

                  min_local_temperature = std::min (min_local_temperature,
                                                    temperature);
                  max_local_temperature = std::max (max_local_temperature,
                                                    temperature);
                }
            }
      }
    else
      {
        typename DoFHandler<dim>::active_cell_iterator
        cell = temperature_dof_handler.begin_active(),
        endc = temperature_dof_handler.end();
        for (; cell!=endc; ++cell)
          if (cell->is_locally_owned())
            {
              fe_values.reinit (cell);
              fe_values.get_function_values (old_temperature_solution,
                                             old_temperature_values);

              for (unsigned int q=0; q<n_q_points; ++q)
                {
                  const double temperature = old_temperature_values[q];

                  min_local_temperature = std::min (min_local_temperature,
                                                    temperature);
                  max_local_temperature = std::max (max_local_temperature,
                                                    temperature);
                }
            }
      }

    double local_extrema[2] = { -min_local_temperature,
                                max_local_temperature
                              };
    double global_extrema[2];
    Utilities::MPI::max (local_extrema, MPI_COMM_WORLD, global_extrema);

    return std::make_pair(-global_extrema[0], global_extrema[1]);
  }



  // template <int dim>
  // double
  // BoussinesqFlowProblem<dim>::
  // compute_viscosity (const std::vector<double>          &old_temperature,
  //                    const std::vector<double>          &old_old_temperature,
  //                    const std::vector<Tensor<1,dim> >  &old_temperature_grads,
  //                    const std::vector<Tensor<1,dim> >  &old_old_temperature_grads,
  //                    const std::vector<double>          &old_temperature_laplacians,
  //                    const std::vector<double>          &old_old_temperature_laplacians,
  //                    const std::vector<Tensor<1,dim> >  &old_velocity_values,
  //                    const std::vector<Tensor<1,dim> >  &old_old_velocity_values,
  //                    const std::vector<SymmetricTensor<2,dim> >  &old_strain_rates,
  //                    const std::vector<SymmetricTensor<2,dim> >  &old_old_strain_rates,
  //                    const double                        global_u_infty,
  //                    const double                        global_T_variation,
  //                    const double                        average_temperature,
  //                    const double                        global_entropy_variation,
  //                    const double                        cell_diameter) const
  // {
  //   if (global_u_infty == 0)
  //     return 5e-3 * cell_diameter;
  // 
  //   const unsigned int n_q_points = old_temperature.size();
  // 
  //   double max_residual = 0;
  //   double max_velocity = 0;
  // 
  //   for (unsigned int q=0; q < n_q_points; ++q)
  //     {
  //       const Tensor<1,dim> u = (old_velocity_values[q] +
  //                                old_old_velocity_values[q]) / 2;
  // 
  //       const SymmetricTensor<2,dim> strain_rate = (old_strain_rates[q] +
  //                                                   old_old_strain_rates[q]) / 2;
  // 
  //       const double T = (old_temperature[q] + old_old_temperature[q]) / 2;
  //       const double dT_dt = (old_temperature[q] - old_old_temperature[q])
  //                            / old_time_step;
  //       const double u_grad_T = u * (old_temperature_grads[q] +
  //                                    old_old_temperature_grads[q]) / 2;
  // 
  //       const double kappa_Delta_T = EquationData::kappa
  //                                    * (old_temperature_laplacians[q] +
  //                                       old_old_temperature_laplacians[q]) / 2;
  //       const double gamma
  //         = ((EquationData::radiogenic_heating * EquationData::density
  //             +
  //             2 * EquationData::eta * strain_rate * strain_rate) /
  //            (EquationData::density * EquationData::specific_heat));
  // 
  //       double residual
  //         = std::abs(dT_dt + u_grad_T - kappa_Delta_T - gamma);
  //       if (parameters.stabilization_alpha == 2)
  //         residual *= std::abs(T - average_temperature);
  // 
  //       max_residual = std::max (residual,        max_residual);
  //       max_velocity = std::max (std::sqrt (u*u), max_velocity);
  //     }
  // 
  //   const double max_viscosity = (parameters.stabilization_beta *
  //                                 max_velocity * cell_diameter);
  //   if (timestep_number == 0)
  //     return max_viscosity;
  //   else
  //     {
  //       Assert (old_time_step > 0, ExcInternalError());
  // 
  //       double entropy_viscosity;
  //       if (parameters.stabilization_alpha == 2)
  //         entropy_viscosity = (parameters.stabilization_c_R *
  //                              cell_diameter * cell_diameter *
  //                              max_residual /
  //                              global_entropy_variation);
  //       else
  //         entropy_viscosity = (parameters.stabilization_c_R *
  //                              cell_diameter * global_Omega_diameter *
  //                              max_velocity * max_residual /
  //                              (global_u_infty * global_T_variation));
  // 
  //       return std::min (max_viscosity, entropy_viscosity);
  //     }
  // }
  // 
  // 


  template <int dim>
  void BoussinesqFlowProblem<dim>::project_temperature_field ()
  {
    assemble_temperature_matrix ();

    QGauss<dim> quadrature(parameters.temperature_degree+2);
    UpdateFlags update_flags = UpdateFlags(update_values   |
                                           update_quadrature_points |
                                           update_JxW_values);
    FEValues<dim> fe_values (mapping, temperature_fe, quadrature, update_flags);

    const unsigned int dofs_per_cell = fe_values.dofs_per_cell,
                       n_q_points    = fe_values.n_quadrature_points;

    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
    Vector<double> cell_vector (dofs_per_cell);
    FullMatrix<double> matrix_for_bc (dofs_per_cell, dofs_per_cell);

    std::vector<double> rhs_values(n_q_points);

    TrilinosWrappers::MPI::Vector
    rhs (temperature_mass_matrix.row_partitioner()),
        solution (temperature_mass_matrix.row_partitioner());

    const EquationData::TemperatureInitialValues<dim> initial_temperature;

    typename DoFHandler<dim>::active_cell_iterator
    cell = temperature_dof_handler.begin_active(),
    endc = temperature_dof_handler.end();

    for (; cell!=endc; ++cell)
      if (cell->is_locally_owned())
        {
          cell->get_dof_indices (local_dof_indices);
          fe_values.reinit (cell);

          initial_temperature.value_list (fe_values.get_quadrature_points(),
                                          rhs_values);

          cell_vector = 0;
          matrix_for_bc = 0;
          for (unsigned int point=0; point<n_q_points; ++point)
            for (unsigned int i=0; i<dofs_per_cell; ++i)
              {
                cell_vector(i) += rhs_values[point] *
                                  fe_values.shape_value(i,point) *
                                  fe_values.JxW(point);
                if (temperature_constraints.is_inhomogeneously_constrained(local_dof_indices[i]))
                  {
                    for (unsigned int j=0; j<dofs_per_cell; ++j)
                      matrix_for_bc(j,i) += fe_values.shape_value(i,point) *
                                            fe_values.shape_value(j,point) *
                                            fe_values.JxW(point);
                  }
              }

          temperature_constraints.distribute_local_to_global (cell_vector,
                                                              local_dof_indices,
                                                              rhs,
                                                              matrix_for_bc);
        }

    rhs.compress (VectorOperation::add);

    SolverControl solver_control(5*rhs.size(), 1e-12*rhs.l2_norm());
    SolverCG<TrilinosWrappers::MPI::Vector> cg(solver_control);

    TrilinosWrappers::PreconditionJacobi preconditioner_mass;
    preconditioner_mass.initialize(temperature_mass_matrix, 1.3);

    cg.solve (temperature_mass_matrix, solution, rhs, preconditioner_mass);

    temperature_constraints.distribute (solution);

    temperature_solution = solution;
    old_temperature_solution = solution;
    old_old_temperature_solution = solution;
  }





  template <int dim>
  void BoussinesqFlowProblem<dim>::
  setup_stokes_matrix (const std::vector<IndexSet> &stokes_partitioning,
                       const std::vector<IndexSet> &stokes_relevant_partitioning)
  {
    stokes_matrix.clear ();

    TrilinosWrappers::BlockSparsityPattern sp(stokes_partitioning, stokes_partitioning,
                                              stokes_relevant_partitioning,
                                              MPI_COMM_WORLD);

    Table<2,DoFTools::Coupling> coupling (dim+1, dim+1);
    for (unsigned int c=0; c<dim+1; ++c)
      for (unsigned int d=0; d<dim+1; ++d)
        if (! ((c==dim) && (d==dim)))
          coupling[c][d] = DoFTools::always;
        else
          coupling[c][d] = DoFTools::none;

    DoFTools::make_sparsity_pattern (stokes_dof_handler,
                                     coupling, sp,
                                     stokes_constraints, false,
                                     Utilities::MPI::
                                     this_mpi_process(MPI_COMM_WORLD));
    sp.compress();

    stokes_matrix.reinit (sp);
  }



  template <int dim>
  void BoussinesqFlowProblem<dim>::
  setup_stokes_preconditioner (const std::vector<IndexSet> &stokes_partitioning,
                               const std::vector<IndexSet> &stokes_relevant_partitioning)
  {
    Amg_preconditioner.reset ();
    Mp_preconditioner.reset ();

    stokes_preconditioner_matrix.clear ();

    TrilinosWrappers::BlockSparsityPattern sp(stokes_partitioning, stokes_partitioning,
                                              stokes_relevant_partitioning,
                                              MPI_COMM_WORLD);

    Table<2,DoFTools::Coupling> coupling (dim+1, dim+1);
    for (unsigned int c=0; c<dim+1; ++c)
      for (unsigned int d=0; d<dim+1; ++d)
        if (c == d)
          coupling[c][d] = DoFTools::always;
        else
          coupling[c][d] = DoFTools::none;

    DoFTools::make_sparsity_pattern (stokes_dof_handler,
                                     coupling, sp,
                                     stokes_constraints, false,
                                     Utilities::MPI::
                                     this_mpi_process(MPI_COMM_WORLD));
    sp.compress();

    stokes_preconditioner_matrix.reinit (sp);
  }


  template <int dim>
  void BoussinesqFlowProblem<dim>::
  setup_temperature_matrices (const IndexSet &temperature_partitioner,
                              const IndexSet &temperature_relevant_partitioner)
  {
    T_preconditioner.reset ();
    temperature_mass_matrix.clear ();
    temperature_stiffness_matrix.clear ();
    temperature_matrix.clear ();

    TrilinosWrappers::SparsityPattern sp(temperature_partitioner,
                                         temperature_partitioner,
                                         temperature_relevant_partitioner,
                                         MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern (temperature_dof_handler, sp,
                                     temperature_constraints, false,
                                     Utilities::MPI::
                                     this_mpi_process(MPI_COMM_WORLD));
    sp.compress();

    temperature_matrix.reinit (sp);
    temperature_mass_matrix.reinit (sp);
    temperature_stiffness_matrix.reinit (sp);
  }



  template <int dim>
  void BoussinesqFlowProblem<dim>::setup_dofs ()
  {
    computing_timer.enter_section("Setup dof systems");

    std::vector<unsigned int> stokes_sub_blocks (dim+1,0);
    stokes_sub_blocks[dim] = 1;
    stokes_dof_handler.distribute_dofs (stokes_fe);
    DoFRenumbering::component_wise (stokes_dof_handler, stokes_sub_blocks);

    temperature_dof_handler.distribute_dofs (temperature_fe);

    std::vector<types::global_dof_index> stokes_dofs_per_block (2);
    DoFTools::count_dofs_per_block (stokes_dof_handler, stokes_dofs_per_block,
                                    stokes_sub_blocks);

    const unsigned int n_u = stokes_dofs_per_block[0],
                       n_p = stokes_dofs_per_block[1],
                       n_T = temperature_dof_handler.n_dofs();

    std::locale s = pcout.get_stream().getloc();
    pcout.get_stream().imbue(std::locale(""));
    pcout << "Number of active cells: "
          << triangulation.n_global_active_cells()
          << " (on "
          << triangulation.n_levels()
          << " levels)"
          << std::endl
          << "Number of degrees of freedom: "
          << n_u + n_p + n_T
          << " (" << n_u << '+' << n_p << '+'<< n_T <<')'
          << std::endl
          << std::endl;
    pcout.get_stream().imbue(s);


    std::vector<IndexSet> stokes_partitioning, stokes_relevant_partitioning;
    IndexSet temperature_partitioning (n_T), temperature_relevant_partitioning (n_T);
    IndexSet stokes_relevant_set;
    {
      IndexSet stokes_index_set = stokes_dof_handler.locally_owned_dofs();
      stokes_partitioning.push_back(stokes_index_set.get_view(0,n_u));
      stokes_partitioning.push_back(stokes_index_set.get_view(n_u,n_u+n_p));

      DoFTools::extract_locally_relevant_dofs (stokes_dof_handler,
                                               stokes_relevant_set);
      stokes_relevant_partitioning.push_back(stokes_relevant_set.get_view(0,n_u));
      stokes_relevant_partitioning.push_back(stokes_relevant_set.get_view(n_u,n_u+n_p));

      temperature_partitioning = temperature_dof_handler.locally_owned_dofs();
      DoFTools::extract_locally_relevant_dofs (temperature_dof_handler,
                                               temperature_relevant_partitioning);
    }

    {
      stokes_constraints.clear ();
      stokes_constraints.reinit (stokes_relevant_set);

      DoFTools::make_hanging_node_constraints (stokes_dof_handler,
                                               stokes_constraints);

      FEValuesExtractors::Vector velocity_components(0);
      VectorTools::interpolate_boundary_values (stokes_dof_handler,
                                                0,
                                                ZeroFunction<dim>(dim+1),
                                                stokes_constraints,
                                                stokes_fe.component_mask(velocity_components));

      std::set<types::boundary_id> no_normal_flux_boundaries;
      no_normal_flux_boundaries.insert (1);
      VectorTools::compute_no_normal_flux_constraints (stokes_dof_handler, 0,
                                                       no_normal_flux_boundaries,
                                                       stokes_constraints,
                                                       mapping);
      stokes_constraints.close ();
    }
    {
      temperature_constraints.clear ();
      temperature_constraints.reinit (temperature_relevant_partitioning);

      DoFTools::make_hanging_node_constraints (temperature_dof_handler,
                                               temperature_constraints);
      VectorTools::interpolate_boundary_values (temperature_dof_handler,
                                                0,
                                                EquationData::TemperatureInitialValues<dim>(),
                                                temperature_constraints);
      VectorTools::interpolate_boundary_values (temperature_dof_handler,
                                                1,
                                                EquationData::TemperatureInitialValues<dim>(),
                                                temperature_constraints);
      temperature_constraints.close ();
    }

    setup_stokes_matrix (stokes_partitioning, stokes_relevant_partitioning);
    setup_stokes_preconditioner (stokes_partitioning,
                                 stokes_relevant_partitioning);
    setup_temperature_matrices (temperature_partitioning,
                                temperature_relevant_partitioning);

    stokes_rhs.reinit (stokes_partitioning, stokes_relevant_partitioning,
                       MPI_COMM_WORLD, true);
    stokes_solution.reinit (stokes_relevant_partitioning, MPI_COMM_WORLD);
    old_stokes_solution.reinit (stokes_solution);

    temperature_rhs.reinit (temperature_partitioning,
                            temperature_relevant_partitioning,
                            MPI_COMM_WORLD, true);
    temperature_solution.reinit (temperature_relevant_partitioning, MPI_COMM_WORLD);
    old_temperature_solution.reinit (temperature_solution);
    old_old_temperature_solution.reinit (temperature_solution);

    rebuild_stokes_matrix              = true;
    rebuild_stokes_preconditioner      = true;
    rebuild_temperature_matrices       = true;
    rebuild_temperature_preconditioner = true;

    computing_timer.exit_section();
  }




  template <int dim>
  void
  BoussinesqFlowProblem<dim>::
  local_assemble_stokes_preconditioner (const typename DoFHandler<dim>::active_cell_iterator &cell,
                                        Assembly::Scratch::StokesPreconditioner<dim> &scratch,
                                        Assembly::CopyData::StokesPreconditioner<dim> &data)
  {
    const unsigned int   dofs_per_cell   = stokes_fe.dofs_per_cell;
    const unsigned int   n_q_points      = scratch.stokes_fe_values.n_quadrature_points;

    const FEValuesExtractors::Vector velocities (0);
    const FEValuesExtractors::Scalar pressure (dim);

    scratch.stokes_fe_values.reinit (cell);
    cell->get_dof_indices (data.local_dof_indices);

    data.local_matrix = 0;

    for (unsigned int q=0; q<n_q_points; ++q)
      {
        for (unsigned int k=0; k<dofs_per_cell; ++k)
          {
            scratch.grad_phi_u[k] = scratch.stokes_fe_values[velocities].gradient(k,q);
            scratch.phi_p[k]      = scratch.stokes_fe_values[pressure].value (k, q);
          }

        for (unsigned int i=0; i<dofs_per_cell; ++i)
          for (unsigned int j=0; j<dofs_per_cell; ++j)
            data.local_matrix(i,j) += (EquationData::eta *
                                       scalar_product (scratch.grad_phi_u[i],
                                                       scratch.grad_phi_u[j])
                                       +
                                       (1./EquationData::eta) *
                                       EquationData::pressure_scaling *
                                       EquationData::pressure_scaling *
                                       (scratch.phi_p[i] * scratch.phi_p[j]))
                                      * scratch.stokes_fe_values.JxW(q);
      }
  }



  template <int dim>
  void
  BoussinesqFlowProblem<dim>::
  copy_local_to_global_stokes_preconditioner (const Assembly::CopyData::StokesPreconditioner<dim> &data)
  {
    stokes_constraints.distribute_local_to_global (data.local_matrix,
                                                   data.local_dof_indices,
                                                   stokes_preconditioner_matrix);
  }


  template <int dim>
  void
  BoussinesqFlowProblem<dim>::assemble_stokes_preconditioner ()
  {
    stokes_preconditioner_matrix = 0;

    const QGauss<dim> quadrature_formula(parameters.stokes_velocity_degree+1);

    typedef
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
    CellFilter;

    WorkStream::
    run (CellFilter (IteratorFilters::LocallyOwnedCell(),
                     stokes_dof_handler.begin_active()),
         CellFilter (IteratorFilters::LocallyOwnedCell(),
                     stokes_dof_handler.end()),
         std_cxx11::bind (&BoussinesqFlowProblem<dim>::
                          local_assemble_stokes_preconditioner,
                          this,
                          std_cxx11::_1,
                          std_cxx11::_2,
                          std_cxx11::_3),
         std_cxx11::bind (&BoussinesqFlowProblem<dim>::
                          copy_local_to_global_stokes_preconditioner,
                          this,
                          std_cxx11::_1),
         Assembly::Scratch::
         StokesPreconditioner<dim> (stokes_fe, quadrature_formula,
                                    mapping,
                                    update_JxW_values |
                                    update_values |
                                    update_gradients),
         Assembly::CopyData::
         StokesPreconditioner<dim> (stokes_fe));

    stokes_preconditioner_matrix.compress(VectorOperation::add);
  }



  template <int dim>
  void
  BoussinesqFlowProblem<dim>::build_stokes_preconditioner ()
  {
    if (rebuild_stokes_preconditioner == false)
      return;

    computing_timer.enter_section ("   Build Stokes preconditioner");
    pcout << "   Rebuilding Stokes preconditioner..." << std::flush;

    assemble_stokes_preconditioner ();

    std::vector<std::vector<bool> > constant_modes;
    FEValuesExtractors::Vector velocity_components(0);
    DoFTools::extract_constant_modes (stokes_dof_handler,
                                      stokes_fe.component_mask(velocity_components),
                                      constant_modes);

    Mp_preconditioner.reset  (new TrilinosWrappers::PreconditionJacobi());
    Amg_preconditioner.reset (new TrilinosWrappers::PreconditionAMG());

    TrilinosWrappers::PreconditionAMG::AdditionalData Amg_data;
    Amg_data.constant_modes = constant_modes;
    Amg_data.elliptic = true;
    Amg_data.higher_order_elements = true;
    Amg_data.smoother_sweeps = 2;
    Amg_data.aggregation_threshold = 0.02;

    Mp_preconditioner->initialize (stokes_preconditioner_matrix.block(1,1));
    Amg_preconditioner->initialize (stokes_preconditioner_matrix.block(0,0),
                                    Amg_data);

    rebuild_stokes_preconditioner = false;

    pcout << std::endl;
    computing_timer.exit_section();
  }



  template <int dim>
  void
  BoussinesqFlowProblem<dim>::
  local_assemble_stokes_system (const typename DoFHandler<dim>::active_cell_iterator &cell,
                                Assembly::Scratch::StokesSystem<dim> &scratch,
                                Assembly::CopyData::StokesSystem<dim> &data)
  {
    const unsigned int dofs_per_cell = scratch.stokes_fe_values.get_fe().dofs_per_cell;
    const unsigned int n_q_points    = scratch.stokes_fe_values.n_quadrature_points;

    const FEValuesExtractors::Vector velocities (0);
    const FEValuesExtractors::Scalar pressure (dim);

    scratch.stokes_fe_values.reinit (cell);

    typename DoFHandler<dim>::active_cell_iterator
    temperature_cell (&triangulation,
                      cell->level(),
                      cell->index(),
                      &temperature_dof_handler);
    scratch.temperature_fe_values.reinit (temperature_cell);

    if (rebuild_stokes_matrix)
      data.local_matrix = 0;
    data.local_rhs = 0;

    scratch.temperature_fe_values.get_function_values (old_temperature_solution,
                                                       scratch.old_temperature_values);

    for (unsigned int q=0; q<n_q_points; ++q)
      {
        const double old_temperature = scratch.old_temperature_values[q];

        for (unsigned int k=0; k<dofs_per_cell; ++k)
          {
            scratch.phi_u[k] = scratch.stokes_fe_values[velocities].value (k,q);
            if (rebuild_stokes_matrix)
              {
                scratch.grads_phi_u[k] = scratch.stokes_fe_values[velocities].symmetric_gradient(k,q);
                scratch.div_phi_u[k]   = scratch.stokes_fe_values[velocities].divergence (k, q);
                scratch.phi_p[k]       = scratch.stokes_fe_values[pressure].value (k, q);
              }
          }

        if (rebuild_stokes_matrix == true)
          for (unsigned int i=0; i<dofs_per_cell; ++i)
            for (unsigned int j=0; j<dofs_per_cell; ++j)
              data.local_matrix(i,j) += (EquationData::eta * 2 *
                                         (scratch.grads_phi_u[i] * scratch.grads_phi_u[j])
                                         - (EquationData::pressure_scaling *
                                            scratch.div_phi_u[i] * scratch.phi_p[j])
                                         - (EquationData::pressure_scaling *
                                            scratch.phi_p[i] * scratch.div_phi_u[j]))
                                        * scratch.stokes_fe_values.JxW(q);

        const Tensor<1,dim>
        gravity = EquationData::gravity_vector (scratch.stokes_fe_values
                                                .quadrature_point(q));

        for (unsigned int i=0; i<dofs_per_cell; ++i)
          data.local_rhs(i) += (EquationData::density *
                                gravity  *
                                scratch.phi_u[i]) *
                               scratch.stokes_fe_values.JxW(q);
      }

    cell->get_dof_indices (data.local_dof_indices);
  }



  template <int dim>
  void
  BoussinesqFlowProblem<dim>::
  copy_local_to_global_stokes_system (const Assembly::CopyData::StokesSystem<dim> &data)
  {
    if (rebuild_stokes_matrix == true)
      stokes_constraints.distribute_local_to_global (data.local_matrix,
                                                     data.local_rhs,
                                                     data.local_dof_indices,
                                                     stokes_matrix,
                                                     stokes_rhs);
    else
      stokes_constraints.distribute_local_to_global (data.local_rhs,
                                                     data.local_dof_indices,
                                                     stokes_rhs);
  }



  template <int dim>
  void BoussinesqFlowProblem<dim>::assemble_stokes_system ()
  {
    computing_timer.enter_section ("   Assemble Stokes system");

    if (rebuild_stokes_matrix == true)
      stokes_matrix=0;

    stokes_rhs=0;

    const QGauss<dim> quadrature_formula(parameters.stokes_velocity_degree+1);

    typedef
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
    CellFilter;

    WorkStream::
    run (CellFilter (IteratorFilters::LocallyOwnedCell(),
                     stokes_dof_handler.begin_active()),
         CellFilter (IteratorFilters::LocallyOwnedCell(),
                     stokes_dof_handler.end()),
         std_cxx11::bind (&BoussinesqFlowProblem<dim>::
                          local_assemble_stokes_system,
                          this,
                          std_cxx11::_1,
                          std_cxx11::_2,
                          std_cxx11::_3),
         std_cxx11::bind (&BoussinesqFlowProblem<dim>::
                          copy_local_to_global_stokes_system,
                          this,
                          std_cxx11::_1),
         Assembly::Scratch::
         StokesSystem<dim> (stokes_fe, mapping, quadrature_formula,
                            (update_values    |
                             update_quadrature_points  |
                             update_JxW_values |
                             (rebuild_stokes_matrix == true
                              ?
                              update_gradients
                              :
                              UpdateFlags(0))),
                            temperature_fe,
                            update_values),
         Assembly::CopyData::
         StokesSystem<dim> (stokes_fe));

    if (rebuild_stokes_matrix == true)
      stokes_matrix.compress(VectorOperation::add);
    stokes_rhs.compress(VectorOperation::add);

    rebuild_stokes_matrix = false;

    pcout << std::endl;
    computing_timer.exit_section();
  }



  template <int dim>
  void BoussinesqFlowProblem<dim>::
  local_assemble_temperature_matrix (const typename DoFHandler<dim>::active_cell_iterator &cell,
                                     Assembly::Scratch::TemperatureMatrix<dim> &scratch,
                                     Assembly::CopyData::TemperatureMatrix<dim> &data)
  {
    const unsigned int dofs_per_cell = scratch.temperature_fe_values.get_fe().dofs_per_cell;
    const unsigned int n_q_points    = scratch.temperature_fe_values.n_quadrature_points;

    scratch.temperature_fe_values.reinit (cell);
    cell->get_dof_indices (data.local_dof_indices);

    data.local_mass_matrix = 0;
    data.local_stiffness_matrix = 0;

    for (unsigned int q=0; q<n_q_points; ++q)
      {
        for (unsigned int k=0; k<dofs_per_cell; ++k)
          {
            scratch.grad_phi_T[k] = scratch.temperature_fe_values.shape_grad (k,q);
            scratch.phi_T[k]      = scratch.temperature_fe_values.shape_value (k, q);
          }

        for (unsigned int i=0; i<dofs_per_cell; ++i)
          for (unsigned int j=0; j<dofs_per_cell; ++j)
            {
              data.local_mass_matrix(i,j)
              += (scratch.phi_T[i] * scratch.phi_T[j]
                  *
                  scratch.temperature_fe_values.JxW(q));
              data.local_stiffness_matrix(i,j)
              += (EquationData::kappa * scratch.grad_phi_T[i] * scratch.grad_phi_T[j]
                  *
                  scratch.temperature_fe_values.JxW(q));
            }
      }
  }



  template <int dim>
  void
  BoussinesqFlowProblem<dim>::
  copy_local_to_global_temperature_matrix (const Assembly::CopyData::TemperatureMatrix<dim> &data)
  {
    temperature_constraints.distribute_local_to_global (data.local_mass_matrix,
                                                        data.local_dof_indices,
                                                        temperature_mass_matrix);
    temperature_constraints.distribute_local_to_global (data.local_stiffness_matrix,
                                                        data.local_dof_indices,
                                                        temperature_stiffness_matrix);
  }


  template <int dim>
  void BoussinesqFlowProblem<dim>::assemble_temperature_matrix ()
  {
    if (rebuild_temperature_matrices == false)
      return;

    computing_timer.enter_section ("   Assemble temperature matrices");
    temperature_mass_matrix = 0;
    temperature_stiffness_matrix = 0;

    const QGauss<dim> quadrature_formula(parameters.temperature_degree+2);

    typedef
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
    CellFilter;

    WorkStream::
    run (CellFilter (IteratorFilters::LocallyOwnedCell(),
                     temperature_dof_handler.begin_active()),
         CellFilter (IteratorFilters::LocallyOwnedCell(),
                     temperature_dof_handler.end()),
         std_cxx11::bind (&BoussinesqFlowProblem<dim>::
                          local_assemble_temperature_matrix,
                          this,
                          std_cxx11::_1,
                          std_cxx11::_2,
                          std_cxx11::_3),
         std_cxx11::bind (&BoussinesqFlowProblem<dim>::
                          copy_local_to_global_temperature_matrix,
                          this,
                          std_cxx11::_1),
         Assembly::Scratch::
         TemperatureMatrix<dim> (temperature_fe, mapping, quadrature_formula),
         Assembly::CopyData::
         TemperatureMatrix<dim> (temperature_fe));

    temperature_mass_matrix.compress(VectorOperation::add);
    temperature_stiffness_matrix.compress(VectorOperation::add);

    rebuild_temperature_matrices = false;
    rebuild_temperature_preconditioner = true;

    computing_timer.exit_section();
  }



  template <int dim>
  void BoussinesqFlowProblem<dim>::
  local_assemble_temperature_rhs (const std::pair<double,double> global_T_range,
                                  const double                   global_max_velocity,
                                  const double                   global_entropy_variation,
                                  const typename DoFHandler<dim>::active_cell_iterator &cell,
                                  Assembly::Scratch::TemperatureRHS<dim> &scratch,
                                  Assembly::CopyData::TemperatureRHS<dim> &data)
  {
    const bool use_bdf2_scheme = (timestep_number != 0);

    const unsigned int dofs_per_cell = scratch.temperature_fe_values.get_fe().dofs_per_cell;
    const unsigned int n_q_points    = scratch.temperature_fe_values.n_quadrature_points;

    const FEValuesExtractors::Vector velocities (0);

    data.local_rhs = 0;
    data.matrix_for_bc = 0;
    cell->get_dof_indices (data.local_dof_indices);

    scratch.temperature_fe_values.reinit (cell);

    typename DoFHandler<dim>::active_cell_iterator
    stokes_cell (&triangulation,
                 cell->level(),
                 cell->index(),
                 &stokes_dof_handler);
    scratch.stokes_fe_values.reinit (stokes_cell);

    scratch.temperature_fe_values.get_function_values (old_temperature_solution,
                                                       scratch.old_temperature_values);
    scratch.temperature_fe_values.get_function_values (old_old_temperature_solution,
                                                       scratch.old_old_temperature_values);

    scratch.temperature_fe_values.get_function_gradients (old_temperature_solution,
                                                          scratch.old_temperature_grads);
    scratch.temperature_fe_values.get_function_gradients (old_old_temperature_solution,
                                                          scratch.old_old_temperature_grads);

    scratch.temperature_fe_values.get_function_laplacians (old_temperature_solution,
                                                           scratch.old_temperature_laplacians);
    scratch.temperature_fe_values.get_function_laplacians (old_old_temperature_solution,
                                                           scratch.old_old_temperature_laplacians);

    scratch.stokes_fe_values[velocities].get_function_values (stokes_solution,
                                                              scratch.old_velocity_values);
    scratch.stokes_fe_values[velocities].get_function_values (old_stokes_solution,
                                                              scratch.old_old_velocity_values);
    scratch.stokes_fe_values[velocities].get_function_symmetric_gradients (stokes_solution,
        scratch.old_strain_rates);
    scratch.stokes_fe_values[velocities].get_function_symmetric_gradients (old_stokes_solution,
        scratch.old_old_strain_rates);

    // const double nu
    //   = compute_viscosity (scratch.old_temperature_values,
    //                        scratch.old_old_temperature_values,
    //                        scratch.old_temperature_grads,
    //                        scratch.old_old_temperature_grads,
    //                        scratch.old_temperature_laplacians,
    //                        scratch.old_old_temperature_laplacians,
    //                        scratch.old_velocity_values,
    //                        scratch.old_old_velocity_values,
    //                        scratch.old_strain_rates,
    //                        scratch.old_old_strain_rates,
    //                        global_max_velocity,
    //                        global_T_range.second - global_T_range.first,
    //                        0.5 * (global_T_range.second + global_T_range.first),
    //                        global_entropy_variation,
    //                        cell->diameter());

    for (unsigned int q=0; q<n_q_points; ++q)
      {
        for (unsigned int k=0; k<dofs_per_cell; ++k)
          {
            scratch.phi_T[k]      = scratch.temperature_fe_values.shape_value (k, q);
            scratch.grad_phi_T[k] = scratch.temperature_fe_values.shape_grad (k,q);
          }


        const double T_term_for_rhs
          = (use_bdf2_scheme ?
             (scratch.old_temperature_values[q] *
              (1 + time_step/old_time_step)
              -
              scratch.old_old_temperature_values[q] *
              (time_step * time_step) /
              (old_time_step * (time_step + old_time_step)))
             :
             scratch.old_temperature_values[q]);

        const double ext_T
          = (use_bdf2_scheme ?
             (scratch.old_temperature_values[q] *
              (1 + time_step/old_time_step)
              -
              scratch.old_old_temperature_values[q] *
              time_step/old_time_step)
             :
             scratch.old_temperature_values[q]);

        const Tensor<1,dim> ext_grad_T
          = (use_bdf2_scheme ?
             (scratch.old_temperature_grads[q] *
              (1 + time_step/old_time_step)
              -
              scratch.old_old_temperature_grads[q] *
              time_step/old_time_step)
             :
             scratch.old_temperature_grads[q]);

        const Tensor<1,dim> extrapolated_u
          = (use_bdf2_scheme ?
             (scratch.old_velocity_values[q] *
              (1 + time_step/old_time_step)
              -
              scratch.old_old_velocity_values[q] *
              time_step/old_time_step)
             :
             scratch.old_velocity_values[q]);

        const SymmetricTensor<2,dim> extrapolated_strain_rate
          = (use_bdf2_scheme ?
             (scratch.old_strain_rates[q] *
              (1 + time_step/old_time_step)
              -
              scratch.old_old_strain_rates[q] *
              time_step/old_time_step)
             :
             scratch.old_strain_rates[q]);

        const double gamma
          = ((EquationData::radiogenic_heating * EquationData::density
              +
              2 * EquationData::eta * extrapolated_strain_rate * extrapolated_strain_rate) /
             (EquationData::density * EquationData::specific_heat));

        for (unsigned int i=0; i<dofs_per_cell; ++i)
          {
            data.local_rhs(i) += (T_term_for_rhs * scratch.phi_T[i]
                                  -
                                  time_step *
                                  extrapolated_u * ext_grad_T * scratch.phi_T[i]
                                  -
                                  time_step *
                                  EquationData::nu * ext_grad_T * scratch.grad_phi_T[i]
                                  +
                                  time_step *
                                  gamma * scratch.phi_T[i])
                                 *
                                 scratch.temperature_fe_values.JxW(q);

            if (temperature_constraints.is_inhomogeneously_constrained(data.local_dof_indices[i]))
              {
                for (unsigned int j=0; j<dofs_per_cell; ++j)
                  data.matrix_for_bc(j,i) += (scratch.phi_T[i] * scratch.phi_T[j] *
                                              (use_bdf2_scheme ?
                                               ((2*time_step + old_time_step) /
                                                (time_step + old_time_step)) : 1.)
                                              +
                                              scratch.grad_phi_T[i] *
                                              scratch.grad_phi_T[j] *
                                              EquationData::kappa *
                                              time_step)
                                             *
                                             scratch.temperature_fe_values.JxW(q);
              }
          }
      }
  }


  template <int dim>
  void
  BoussinesqFlowProblem<dim>::
  copy_local_to_global_temperature_rhs (const Assembly::CopyData::TemperatureRHS<dim> &data)
  {
    temperature_constraints.distribute_local_to_global (data.local_rhs,
                                                        data.local_dof_indices,
                                                        temperature_rhs,
                                                        data.matrix_for_bc);
  }



  template <int dim>
  void BoussinesqFlowProblem<dim>::assemble_temperature_system (const double maximal_velocity)
  {
    const bool use_bdf2_scheme = (timestep_number != 0);

    if (use_bdf2_scheme == true)
      {
        temperature_matrix.copy_from (temperature_mass_matrix);
        temperature_matrix *= (2*time_step + old_time_step) /
                              (time_step + old_time_step);
        temperature_matrix.add (time_step, temperature_stiffness_matrix);
      }
    else
      {
        temperature_matrix.copy_from (temperature_mass_matrix);
        temperature_matrix.add (time_step, temperature_stiffness_matrix);
      }

    if (rebuild_temperature_preconditioner == true)
      {
        T_preconditioner.reset (new TrilinosWrappers::PreconditionJacobi());
        T_preconditioner->initialize (temperature_matrix);
        rebuild_temperature_preconditioner = false;
      }

    temperature_rhs = 0;

    const QGauss<dim> quadrature_formula(parameters.temperature_degree+2);
    const std::pair<double,double>
    global_T_range = get_extrapolated_temperature_range();

    const double average_temperature = 0.5 * (global_T_range.first +
                                              global_T_range.second);
    const double global_entropy_variation =
      get_entropy_variation (average_temperature);

    typedef
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
    CellFilter;

    WorkStream::
    run (CellFilter (IteratorFilters::LocallyOwnedCell(),
                     temperature_dof_handler.begin_active()),
         CellFilter (IteratorFilters::LocallyOwnedCell(),
                     temperature_dof_handler.end()),
         std_cxx11::bind (&BoussinesqFlowProblem<dim>::
                          local_assemble_temperature_rhs,
                          this,
                          global_T_range,
                          maximal_velocity,
                          global_entropy_variation,
                          std_cxx11::_1,
                          std_cxx11::_2,
                          std_cxx11::_3),
         std_cxx11::bind (&BoussinesqFlowProblem<dim>::
                          copy_local_to_global_temperature_rhs,
                          this,
                          std_cxx11::_1),
         Assembly::Scratch::
         TemperatureRHS<dim> (temperature_fe, stokes_fe, mapping,
                              quadrature_formula),
         Assembly::CopyData::
         TemperatureRHS<dim> (temperature_fe));

    temperature_rhs.compress(VectorOperation::add);
  }





  template <int dim>
  void BoussinesqFlowProblem<dim>::solve ()
  {
    computing_timer.enter_section ("   Solve Stokes system");

    {
      pcout << "   Solving Stokes system... " << std::flush;

      TrilinosWrappers::MPI::BlockVector
      distributed_stokes_solution (stokes_rhs);
      distributed_stokes_solution = stokes_solution;

      distributed_stokes_solution.block(1) /= EquationData::pressure_scaling;

      const unsigned int
      start = (distributed_stokes_solution.block(0).size() +
               distributed_stokes_solution.block(1).local_range().first),
              end   = (distributed_stokes_solution.block(0).size() +
                       distributed_stokes_solution.block(1).local_range().second);
      for (unsigned int i=start; i<end; ++i)
        if (stokes_constraints.is_constrained (i))
          distributed_stokes_solution(i) = 0;


      PrimitiveVectorMemory<TrilinosWrappers::MPI::BlockVector> mem;

      unsigned int n_iterations = 0;
      const double solver_tolerance = 1e-8 * stokes_rhs.l2_norm();
      SolverControl solver_control (30, solver_tolerance);

      try
        {
          const LinearSolvers::BlockSchurPreconditioner<TrilinosWrappers::PreconditionAMG,
                TrilinosWrappers::PreconditionJacobi>
                preconditioner (stokes_matrix, stokes_preconditioner_matrix,
                                *Mp_preconditioner, *Amg_preconditioner,
                                false);

          SolverFGMRES<TrilinosWrappers::MPI::BlockVector>
          solver(solver_control, mem,
                 SolverFGMRES<TrilinosWrappers::MPI::BlockVector>::
                 AdditionalData(30, true));
          solver.solve(stokes_matrix, distributed_stokes_solution, stokes_rhs,
                       preconditioner);

          n_iterations = solver_control.last_step();
        }

      catch (SolverControl::NoConvergence)
        {
          const LinearSolvers::BlockSchurPreconditioner<TrilinosWrappers::PreconditionAMG,
                TrilinosWrappers::PreconditionJacobi>
                preconditioner (stokes_matrix, stokes_preconditioner_matrix,
                                *Mp_preconditioner, *Amg_preconditioner,
                                true);

          SolverControl solver_control_refined (stokes_matrix.m(), solver_tolerance);
          SolverFGMRES<TrilinosWrappers::MPI::BlockVector>
          solver(solver_control_refined, mem,
                 SolverFGMRES<TrilinosWrappers::MPI::BlockVector>::
                 AdditionalData(50, true));
          solver.solve(stokes_matrix, distributed_stokes_solution, stokes_rhs,
                       preconditioner);

          n_iterations = (solver_control.last_step() +
                          solver_control_refined.last_step());
        }


      stokes_constraints.distribute (distributed_stokes_solution);

      distributed_stokes_solution.block(1) *= EquationData::pressure_scaling;

      stokes_solution = distributed_stokes_solution;
      pcout << n_iterations  << " iterations."
            << std::endl;
    }
    computing_timer.exit_section();


    computing_timer.enter_section ("   Assemble temperature rhs");
    {
      old_time_step = time_step;

      const double scaling = (dim==3 ? 0.25 : 1.0);
      time_step = (scaling/(2.1*dim*std::sqrt(1.*dim)) /
                   (parameters.temperature_degree *
                    get_cfl_number()));

      const double maximal_velocity = get_maximal_velocity();
      pcout << "   Maximal velocity: "
            << maximal_velocity *EquationData::year_in_seconds * 100
            << " cm/year"
            << std::endl;
      pcout << "   " << "Time step: "
            << time_step/EquationData::year_in_seconds
            << " years"
            << std::endl;

      temperature_solution = old_temperature_solution;
      assemble_temperature_system (maximal_velocity);
    }
    computing_timer.exit_section ();

    computing_timer.enter_section ("   Solve temperature system");
    {
      SolverControl solver_control (temperature_matrix.m(),
                                    1e-12*temperature_rhs.l2_norm());
      SolverCG<TrilinosWrappers::MPI::Vector>   cg (solver_control);

      TrilinosWrappers::MPI::Vector
      distributed_temperature_solution (temperature_rhs);
      distributed_temperature_solution = temperature_solution;

      cg.solve (temperature_matrix, distributed_temperature_solution,
                temperature_rhs, *T_preconditioner);

      temperature_constraints.distribute (distributed_temperature_solution);
      temperature_solution = distributed_temperature_solution;

      pcout << "   "
            << solver_control.last_step()
            << " CG iterations for temperature" << std::endl;
      computing_timer.exit_section();

      double temperature[2] = { std::numeric_limits<double>::max(),
                                -std::numeric_limits<double>::max()
                              };
      double global_temperature[2];

      for (unsigned int i=distributed_temperature_solution.local_range().first;
           i < distributed_temperature_solution.local_range().second; ++i)
        {
          temperature[0] = std::min<double> (temperature[0],
                                             distributed_temperature_solution(i));
          temperature[1] = std::max<double> (temperature[1],
                                             distributed_temperature_solution(i));
        }

      temperature[0] *= -1.0;
      Utilities::MPI::max (temperature, MPI_COMM_WORLD, global_temperature);
      global_temperature[0] *= -1.0;

      pcout << "   Temperature range: "
            << global_temperature[0] << ' ' << global_temperature[1]
            << std::endl;
    }
  }



  template <int dim>
  class BoussinesqFlowProblem<dim>::Postprocessor : public DataPostprocessor<dim>
  {
  public:
    Postprocessor (const unsigned int partition,
                   const double       minimal_pressure);

    virtual
    void
    compute_derived_quantities_vector (const std::vector<Vector<double> >              &uh,
                                       const std::vector<std::vector<Tensor<1,dim> > > &duh,
                                       const std::vector<std::vector<Tensor<2,dim> > > &dduh,
                                       const std::vector<Point<dim> >                  &normals,
                                       const std::vector<Point<dim> >                  &evaluation_points,
                                       std::vector<Vector<double> >                    &computed_quantities) const;

    virtual std::vector<std::string> get_names () const;

    virtual
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    get_data_component_interpretation () const;

    virtual UpdateFlags get_needed_update_flags () const;

  private:
    const unsigned int partition;
    const double       minimal_pressure;
  };


  template <int dim>
  BoussinesqFlowProblem<dim>::Postprocessor::
  Postprocessor (const unsigned int partition,
                 const double       minimal_pressure)
    :
    partition (partition),
    minimal_pressure (minimal_pressure)
  {}


  template <int dim>
  std::vector<std::string>
  BoussinesqFlowProblem<dim>::Postprocessor::get_names() const
  {
    std::vector<std::string> solution_names (dim, "velocity");
    solution_names.push_back ("p");
    solution_names.push_back ("T");
    solution_names.push_back ("friction_heating");
    solution_names.push_back ("partition");

    return solution_names;
  }


  template <int dim>
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  BoussinesqFlowProblem<dim>::Postprocessor::
  get_data_component_interpretation () const
  {
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    interpretation (dim,
                    DataComponentInterpretation::component_is_part_of_vector);

    interpretation.push_back (DataComponentInterpretation::component_is_scalar);
    interpretation.push_back (DataComponentInterpretation::component_is_scalar);
    interpretation.push_back (DataComponentInterpretation::component_is_scalar);
    interpretation.push_back (DataComponentInterpretation::component_is_scalar);

    return interpretation;
  }


  template <int dim>
  UpdateFlags
  BoussinesqFlowProblem<dim>::Postprocessor::get_needed_update_flags() const
  {
    return update_values | update_gradients | update_q_points;
  }


  template <int dim>
  void
  BoussinesqFlowProblem<dim>::Postprocessor::
  compute_derived_quantities_vector (const std::vector<Vector<double> >              &uh,
                                     const std::vector<std::vector<Tensor<1,dim> > > &duh,
                                     const std::vector<std::vector<Tensor<2,dim> > > &/*dduh*/,
                                     const std::vector<Point<dim> >                  &/*normals*/,
                                     const std::vector<Point<dim> >                  &/*evaluation_points*/,
                                     std::vector<Vector<double> >                    &computed_quantities) const
  {
    const unsigned int n_quadrature_points = uh.size();
    Assert (duh.size() == n_quadrature_points,                  ExcInternalError());
    Assert (computed_quantities.size() == n_quadrature_points,  ExcInternalError());
    Assert (uh[0].size() == dim+2,                              ExcInternalError());

    for (unsigned int q=0; q<n_quadrature_points; ++q)
      {
        for (unsigned int d=0; d<dim; ++d)
          computed_quantities[q](d)
            = (uh[q](d) *  EquationData::year_in_seconds * 100);

        const double pressure = (uh[q](dim)-minimal_pressure);
        computed_quantities[q](dim) = pressure;

        const double temperature = uh[q](dim+1);
        computed_quantities[q](dim+1) = temperature;

        Tensor<2,dim> grad_u;
        for (unsigned int d=0; d<dim; ++d)
          grad_u[d] = duh[q][d];
        const SymmetricTensor<2,dim> strain_rate = symmetrize (grad_u);
        computed_quantities[q](dim+2) = 2 * EquationData::eta *
                                        strain_rate * strain_rate;

        computed_quantities[q](dim+3) = partition;
      }
  }


  template <int dim>
  void BoussinesqFlowProblem<dim>::output_results ()
  {
    computing_timer.enter_section ("Postprocessing");

    const FESystem<dim> joint_fe (stokes_fe, 1,
                                  temperature_fe, 1);

    DoFHandler<dim> joint_dof_handler (triangulation);
    joint_dof_handler.distribute_dofs (joint_fe);
    Assert (joint_dof_handler.n_dofs() ==
            stokes_dof_handler.n_dofs() + temperature_dof_handler.n_dofs(),
            ExcInternalError());

    TrilinosWrappers::MPI::Vector joint_solution;
    joint_solution.reinit (joint_dof_handler.locally_owned_dofs(), MPI_COMM_WORLD);

    {
      std::vector<types::global_dof_index> local_joint_dof_indices (joint_fe.dofs_per_cell);
      std::vector<types::global_dof_index> local_stokes_dof_indices (stokes_fe.dofs_per_cell);
      std::vector<types::global_dof_index> local_temperature_dof_indices (temperature_fe.dofs_per_cell);

      typename DoFHandler<dim>::active_cell_iterator
      joint_cell       = joint_dof_handler.begin_active(),
      joint_endc       = joint_dof_handler.end(),
      stokes_cell      = stokes_dof_handler.begin_active(),
      temperature_cell = temperature_dof_handler.begin_active();
      for (; joint_cell!=joint_endc;
           ++joint_cell, ++stokes_cell, ++temperature_cell)
        if (joint_cell->is_locally_owned())
          {
            joint_cell->get_dof_indices (local_joint_dof_indices);
            stokes_cell->get_dof_indices (local_stokes_dof_indices);
            temperature_cell->get_dof_indices (local_temperature_dof_indices);

            for (unsigned int i=0; i<joint_fe.dofs_per_cell; ++i)
              if (joint_fe.system_to_base_index(i).first.first == 0)
                {
                  Assert (joint_fe.system_to_base_index(i).second
                          <
                          local_stokes_dof_indices.size(),
                          ExcInternalError());

                  joint_solution(local_joint_dof_indices[i])
                    = stokes_solution(local_stokes_dof_indices
                                      [joint_fe.system_to_base_index(i).second]);
                }
              else
                {
                  Assert (joint_fe.system_to_base_index(i).first.first == 1,
                          ExcInternalError());
                  Assert (joint_fe.system_to_base_index(i).second
                          <
                          local_temperature_dof_indices.size(),
                          ExcInternalError());
                  joint_solution(local_joint_dof_indices[i])
                    = temperature_solution(local_temperature_dof_indices
                                           [joint_fe.system_to_base_index(i).second]);
                }
          }
    }

    joint_solution.compress(VectorOperation::insert);

    IndexSet locally_relevant_joint_dofs(joint_dof_handler.n_dofs());
    DoFTools::extract_locally_relevant_dofs (joint_dof_handler, locally_relevant_joint_dofs);
    TrilinosWrappers::MPI::Vector locally_relevant_joint_solution;
    locally_relevant_joint_solution.reinit (locally_relevant_joint_dofs, MPI_COMM_WORLD);
    locally_relevant_joint_solution = joint_solution;

    Postprocessor postprocessor (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD),
                                 stokes_solution.block(1).minimal_value());

    DataOut<dim> data_out;
    data_out.attach_dof_handler (joint_dof_handler);
    data_out.add_data_vector (locally_relevant_joint_solution, postprocessor);
    data_out.build_patches ();

    static int out_index=0;
    const std::string filename = ("solution-" +
                                  Utilities::int_to_string (out_index, 5) +
                                  "." +
                                  Utilities::int_to_string
                                  (triangulation.locally_owned_subdomain(), 4) +
                                  ".vtu");
    std::ofstream output (filename.c_str());
    data_out.write_vtu (output);


    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        std::vector<std::string> filenames;
        for (unsigned int i=0; i<Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD); ++i)
          filenames.push_back (std::string("solution-") +
                               Utilities::int_to_string (out_index, 5) +
                               "." +
                               Utilities::int_to_string(i, 4) +
                               ".vtu");
        const std::string
        pvtu_master_filename = ("solution-" +
                                Utilities::int_to_string (out_index, 5) +
                                ".pvtu");
        std::ofstream pvtu_master (pvtu_master_filename.c_str());
        data_out.write_pvtu_record (pvtu_master, filenames);

        const std::string
        visit_master_filename = ("solution-" +
                                 Utilities::int_to_string (out_index, 5) +
                                 ".visit");
        std::ofstream visit_master (visit_master_filename.c_str());
        data_out.write_visit_record (visit_master, filenames);
      }

    computing_timer.exit_section ();
    out_index++;
  }




  template <int dim>
  void BoussinesqFlowProblem<dim>::refine_mesh (const unsigned int max_grid_level)
  {
    computing_timer.enter_section ("Refine mesh structure, part 1");
    Vector<float> estimated_error_per_cell (triangulation.n_active_cells());

    KellyErrorEstimator<dim>::estimate (temperature_dof_handler,
                                        QGauss<dim-1>(parameters.temperature_degree+1),
                                        typename FunctionMap<dim>::type(),
                                        temperature_solution,
                                        estimated_error_per_cell,
                                        ComponentMask(),
                                        0,
                                        0,
                                        triangulation.locally_owned_subdomain());

    parallel::distributed::GridRefinement::
    refine_and_coarsen_fixed_fraction (triangulation,
                                       estimated_error_per_cell,
                                       0.3, 0.1);

    if (triangulation.n_levels() > max_grid_level)
      for (typename Triangulation<dim>::active_cell_iterator
           cell = triangulation.begin_active(max_grid_level);
           cell != triangulation.end(); ++cell)
        cell->clear_refine_flag ();

    std::vector<const TrilinosWrappers::MPI::Vector *> x_temperature (2);
    x_temperature[0] = &temperature_solution;
    x_temperature[1] = &old_temperature_solution;
    std::vector<const TrilinosWrappers::MPI::BlockVector *> x_stokes (2);
    x_stokes[0] = &stokes_solution;
    x_stokes[1] = &old_stokes_solution;

    parallel::distributed::SolutionTransfer<dim,TrilinosWrappers::MPI::Vector>
    temperature_trans(temperature_dof_handler);
    parallel::distributed::SolutionTransfer<dim,TrilinosWrappers::MPI::BlockVector>
    stokes_trans(stokes_dof_handler);

    triangulation.prepare_coarsening_and_refinement();
    temperature_trans.prepare_for_coarsening_and_refinement(x_temperature);
    stokes_trans.prepare_for_coarsening_and_refinement(x_stokes);

    triangulation.execute_coarsening_and_refinement ();
    computing_timer.exit_section();

    setup_dofs ();

    computing_timer.enter_section ("Refine mesh structure, part 2");

    {
      TrilinosWrappers::MPI::Vector distributed_temp1 (temperature_rhs);
      TrilinosWrappers::MPI::Vector distributed_temp2 (temperature_rhs);

      std::vector<TrilinosWrappers::MPI::Vector *> tmp (2);
      tmp[0] = &(distributed_temp1);
      tmp[1] = &(distributed_temp2);
      temperature_trans.interpolate(tmp);

      temperature_solution     = distributed_temp1;
      old_temperature_solution = distributed_temp2;
    }

    {
      TrilinosWrappers::MPI::BlockVector distributed_stokes (stokes_rhs);
      TrilinosWrappers::MPI::BlockVector old_distributed_stokes (stokes_rhs);

      std::vector<TrilinosWrappers::MPI::BlockVector *> stokes_tmp (2);
      stokes_tmp[0] = &(distributed_stokes);
      stokes_tmp[1] = &(old_distributed_stokes);

      stokes_trans.interpolate (stokes_tmp);
      stokes_solution     = distributed_stokes;
      old_stokes_solution = old_distributed_stokes;
    }

    computing_timer.exit_section();
  }




  template <int dim>
  void BoussinesqFlowProblem<dim>::run ()
  {
    GridGenerator::hyper_shell (triangulation,
                                Point<dim>(),
                                EquationData::R0,
                                EquationData::R1,
                                (dim==3) ? 96 : 12,
                                true);
    static HyperShellBoundary<dim> boundary;
    triangulation.set_boundary (0, boundary);
    triangulation.set_boundary (1, boundary);

    global_Omega_diameter = GridTools::diameter (triangulation);

    triangulation.refine_global (parameters.initial_global_refinement);

    setup_dofs();

    unsigned int pre_refinement_step = 0;

start_time_iteration:

    project_temperature_field ();

    timestep_number           = 0;
    time_step = old_time_step = 0;

    double time = 0;

    do
      {
        pcout << "Timestep " << timestep_number
              << ":  t=" << time/EquationData::year_in_seconds
              << " years"
              << std::endl;

        assemble_stokes_system ();
        build_stokes_preconditioner ();
        assemble_temperature_matrix ();

        solve ();

        pcout << std::endl;

        if ((timestep_number == 0) &&
            (pre_refinement_step < parameters.initial_adaptive_refinement))
          {
            refine_mesh (parameters.initial_global_refinement +
                         parameters.initial_adaptive_refinement);
            ++pre_refinement_step;
            goto start_time_iteration;
          }
        else if ((timestep_number > 0)
                 &&
                 (timestep_number % parameters.adaptive_refinement_interval == 0))
          refine_mesh (parameters.initial_global_refinement +
                       parameters.initial_adaptive_refinement);

        if ((parameters.generate_graphical_output == true)
            &&
            (timestep_number % parameters.graphical_output_interval == 0))
          output_results ();

        // if (time > parameters.end_time * EquationData::year_in_seconds)
        //   break;
        if (time > 3 * EquationData::year_in_seconds)
          break;

        TrilinosWrappers::MPI::BlockVector old_old_stokes_solution;
        old_old_stokes_solution      = old_stokes_solution;
        old_stokes_solution          = stokes_solution;
        old_old_temperature_solution = old_temperature_solution;
        old_temperature_solution     = temperature_solution;
        if (old_time_step > 0)
          {
            {
              TrilinosWrappers::MPI::BlockVector distr_solution (stokes_rhs);
              distr_solution = stokes_solution;
              TrilinosWrappers::MPI::BlockVector distr_old_solution (stokes_rhs);
              distr_old_solution = old_old_stokes_solution;
              distr_solution .sadd (1.+time_step/old_time_step, -time_step/old_time_step,
                                    distr_old_solution);
              stokes_solution = distr_solution;
            }
            {
              TrilinosWrappers::MPI::Vector distr_solution (temperature_rhs);
              distr_solution = temperature_solution;
              TrilinosWrappers::MPI::Vector distr_old_solution (temperature_rhs);
              distr_old_solution = old_old_temperature_solution;
              distr_solution .sadd (1.+time_step/old_time_step, -time_step/old_time_step,
                                    distr_old_solution);
              temperature_solution = distr_solution;
            }
          }

        if ((timestep_number > 0) && (timestep_number % 100 == 0))
          computing_timer.print_summary ();

        time += time_step;
        ++timestep_number;
      }
    while (true);

    if ((parameters.generate_graphical_output == true)
        &&
        !((timestep_number-1) % parameters.graphical_output_interval == 0))
      output_results ();
  }
}




int main (int argc, char *argv[])
{
  using namespace Step32;
  using namespace dealii;

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                      numbers::invalid_unsigned_int);

  try
    {
      deallog.depth_console (0);

      std::string parameter_filename;
      if (argc>=2)
        parameter_filename = argv[1];
      else
        parameter_filename = "step-32.prm";

      const int dim = 2;
      BoussinesqFlowProblem<dim>::Parameters  parameters(parameter_filename);
      BoussinesqFlowProblem<dim> flow_problem (parameters);
      flow_problem.run ();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
