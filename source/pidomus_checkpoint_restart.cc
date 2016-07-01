#include "pidomus.h"
#include "pidomus_macros.h"

#include <deal.II/numerics/solution_transfer.h>

#include <deal.II/distributed/solution_transfer.h>

#include <deal2lkit/utilities.h>

#ifdef DEAL_II_WITH_ZLIB
#include <zlib.h>
#endif

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

using namespace dealii;
using namespace deal2lkit;

template <int dim, int spacedim, typename LAC>
void piDoMUS<dim,spacedim,LAC>::create_snapshot() const
{
  auto _timer = computing_timer.scoped_timer ("Create snapshot");
  unsigned int my_id = Utilities::MPI::this_mpi_process (comm);

  if (my_id == 0)
    {
      // if we have previously written a snapshot, then keep the last
      // snapshot in case this one fails to save. Note: static variables
      // will only be initialied once per model run.
      static bool previous_snapshot_exists =
        (resume_computation == true && file_exists(snap_prefix+"restart.mesh"));

      if (previous_snapshot_exists == true)
        {
          copy_file (snap_prefix + "restart.mesh",
                     snap_prefix + "restart.mesh.old");
          copy_file (snap_prefix + "restart.mesh.info",
                     snap_prefix + "restart.mesh.info.old");
          copy_file (snap_prefix + "restart.resume.z",
                     snap_prefix + "restart.resume.z.old");
        }
      // from now on, we know that if we get into this
      // function again that a snapshot has previously
      // been written
      previous_snapshot_exists = true;
    }


  typename LAC::VectorType tmp(solution);
  tmp = locally_relevant_explicit_solution;

  // save triangulation and solution vectors
  save_solutions_and_triangulation(solution,
                                   solution_dot,
                                   locally_relevant_explicit_solution,
                                   locally_relevant_solution,
                                   locally_relevant_solution_dot);

// save general information. This calls serialize() on all
// processes but only writes to the restart file on process 0
  {
    std::ostringstream oss;

    // serialize into a stringstream
    boost::archive::binary_oarchive oa (oss);
    oa << (*this);

    // compress with zlib and write to file on the root processor
#ifdef DEAL_II_WITH_ZLIB
    if (my_id == 0)
      {
        uLongf compressed_data_length = compressBound (oss.str().length());
        std::vector<char *> compressed_data (compressed_data_length);
        int err = compress2 ((Bytef *) &compressed_data[0],
                             &compressed_data_length,
                             (const Bytef *) oss.str().data(),
                             oss.str().length(),
                             Z_BEST_COMPRESSION);
        (void)err;
        Assert (err == Z_OK, ExcInternalError());

        // build compression header
        const uint32_t compression_header[4]
          = { 1,                                   /* number of blocks */
              (uint32_t)oss.str().length(), /* size of block */
              (uint32_t)oss.str().length(), /* size of last block */
              (uint32_t)compressed_data_length
            }; /* list of compressed sizes of blocks */

        std::ofstream f ((snap_prefix + "restart.resume.z").c_str());
        f.write((const char *)compression_header, 4 * sizeof(compression_header[0]));
        f.write((char *)&compressed_data[0], compressed_data_length);
      }
#else
    AssertThrow (false,
                 ExcMessage ("You need to have deal.II configured with the 'libz' "
                             "option to support checkpoint/restart, but deal.II "
                             "did not detect its presence when you called 'cmake'."));
#endif

  }
  pcout << "*** Snapshot created!" << std::endl << std::endl;
}


template <int dim, int spacedim, typename LAC>
void
piDoMUS<dim,spacedim,LAC>::
save_solutions_and_triangulation(const LADealII::VectorType &y,
                                 const LADealII::VectorType &y_dot,
                                 const LADealII::VectorType &y_expl,
                                 const LADealII::VectorType &,
                                 const LADealII::VectorType &) const
{
  std::string sy = snap_prefix+"y.bin";
  std::string sdot = snap_prefix+"y_dot.bin";
  std::string sexpl = snap_prefix+"y_expl.bin";
  const char *file_y = sy.c_str();
  const char *file_y_dot = sdot.c_str();
  const char *file_y_expl = sexpl.c_str();

  if (file_exists(file_y))
    copy_file(file_y,snap_prefix+"y.bin.old");
  if (file_exists(file_y_dot))
    copy_file(file_y_dot,snap_prefix+"y_dot.bin.old");
  if (file_exists(file_y_expl))
    copy_file(file_y_expl,snap_prefix+"y_expl.bin.old");

  std::ofstream out_y (file_y);
  std::ofstream out_y_dot (file_y_dot);
  std::ofstream out_y_expl (file_y_expl);

  y.block_write(out_y);
  out_y.close();

  y_dot.block_write(out_y_dot);
  out_y_dot.close();

  y_expl.block_write(out_y_expl);
  out_y_expl.close();

  triangulation->save ((snap_prefix + "restart.mesh").c_str());
}

template <int dim, int spacedim, typename LAC>
void
piDoMUS<dim,spacedim,LAC>::
save_solutions_and_triangulation(const LATrilinos::VectorType &,
                                 const LATrilinos::VectorType &,
                                 const LATrilinos::VectorType &y_expl,
                                 const LATrilinos::VectorType &y,
                                 const LATrilinos::VectorType &y_dot) const
{
  std::vector<const LATrilinos::VectorType *> x_system (3);
  x_system[0] = &y;
  x_system[1] = &y_dot;
  x_system[2] = &y_expl;


  parallel::distributed::SolutionTransfer<dim, LATrilinos::VectorType, DoFHandler<dim,spacedim> >
  system_trans (*dof_handler);

  system_trans.prepare_serialization (x_system);

  triangulation->save ((snap_prefix + "restart.mesh").c_str());
}


template <int dim, int spacedim, typename LAC>
void piDoMUS<dim,spacedim,LAC>::resume_from_snapshot()
{
  // first check existence of the two restart files
  {
    const std::string filename = snap_prefix + "restart.mesh";
    std::ifstream in (filename.c_str());
    if (!in)
      AssertThrow (false,
                   ExcMessage (std::string("You are trying to restart a previous computation, "
                                           "but the restart file <")
                               +
                               filename
                               +
                               "> does not appear to exist!"));
  }
  {
    const std::string filename = snap_prefix + "restart.resume.z";
    std::ifstream in (filename.c_str());
    if (!in)
      AssertThrow (false,
                   ExcMessage (std::string("You are trying to restart a previous computation, "
                                           "but the restart file <")
                               +
                               filename
                               +
                               "> does not appear to exist!"));
  }

  pcout << "*** Resuming from snapshot!" << std::endl << std::endl;

  // first try to load from the most recent snapshot (i.e., without "old" suffix)
  triangulation = SP(pgg.distributed(comm));
  try
    {
      triangulation->load ((snap_prefix + "restart.mesh").c_str());
    }
  catch (...)
    {
      try
        {
          copy_file (snap_prefix + "restart.mesh.old",
                     snap_prefix + "restart.mesh");
          copy_file (snap_prefix + "restart.mesh.info.old",
                     snap_prefix + "restart.mesh.info");
          copy_file (snap_prefix + "restart.resume.z.old",
                     snap_prefix + "restart.resume.z");
          triangulation->load ((snap_prefix + "restart.mesh").c_str());
        }
      catch (...)
        {
          AssertThrow(false, ExcMessage("Cannot open snapshot mesh file or read the triangulation stored there."));
        }
    }
  dof_handler = SP(new DoFHandler<dim, spacedim>(*triangulation));
  fe = SP(interface.pfe());
  setup_dofs(false);

  load_solutions(locally_relevant_solution,
                 locally_relevant_solution_dot,
                 locally_relevant_explicit_solution);
  solution= locally_relevant_solution;
  solution_dot = locally_relevant_solution_dot;
  // read zlib compressed resume.z
  try
    {
#ifdef DEAL_II_WITH_ZLIB
      std::ifstream ifs ((snap_prefix + "restart.resume.z").c_str());

      AssertThrow(ifs.is_open(),
                  ExcMessage("Cannot open snapshot resume file."));

      uint32_t compression_header[4];
      ifs.read((char *)compression_header, 4 * sizeof(compression_header[0]));
      Assert(compression_header[0]==1, ExcInternalError());

      std::vector<char> compressed(compression_header[3]);
      std::vector<char> uncompressed(compression_header[1]);
      ifs.read(&compressed[0],compression_header[3]);
      uLongf uncompressed_size = compression_header[1];

      const int err = uncompress((Bytef *)&uncompressed[0], &uncompressed_size,
                                 (Bytef *)&compressed[0], compression_header[3]);
      AssertThrow (err == Z_OK,
                   ExcMessage (std::string("Uncompressing the data buffer resulted in an error with code <")
                               +
                               Utilities::int_to_string(err)));

      {
        std::istringstream ss;
        ss.str(std::string (&uncompressed[0], uncompressed_size));
        boost::archive::binary_iarchive ia (ss);
        ia >> (*this);
      }
#else
      AssertThrow (false,
                   ExcMessage ("You need to have deal.II configured with the 'libz' "
                               "option to support checkpoint/restart, but deal.II "
                               "did not detect its presence when you called 'cmake'."));
#endif
    }
  catch (std::exception &e)
    {
      AssertThrow (false,
                   ExcMessage (std::string("Cannot seem to deserialize the data previously stored!\n")
                               +
                               "Some part of the machinery generated an exception that says <"
                               +
                               e.what()
                               +
                               ">"));
    }
}


template <int dim, int spacedim, typename LAC>
void
piDoMUS<dim,spacedim,LAC>::load_solutions(LATrilinos::VectorType &y,
                                          LATrilinos::VectorType &y_dot,
                                          LATrilinos::VectorType &y_expl)
{
  LATrilinos::VectorType distributed_system;
  LATrilinos::VectorType expl_distributed_system;
  LATrilinos::VectorType distributed_system_dot;

  distributed_system.reinit(partitioning,comm);
  expl_distributed_system.reinit(partitioning,comm);
  distributed_system_dot.reinit(partitioning,comm);


  std::vector<LATrilinos::VectorType *> x_system (3);
  x_system[0] = & (distributed_system);
  x_system[1] = & (distributed_system_dot);
  x_system[2] = & (expl_distributed_system);

  parallel::distributed::SolutionTransfer<dim, LATrilinos::VectorType, DoFHandler<dim,spacedim> >
  system_trans (*dof_handler);

  system_trans.deserialize (x_system);

  y = distributed_system;
  y_expl = expl_distributed_system;
  y_dot = distributed_system_dot;

}


template <int dim, int spacedim, typename LAC>
void
piDoMUS<dim,spacedim,LAC>::load_solutions(LADealII::VectorType &y,
                                          LADealII::VectorType &y_dot,
                                          LADealII::VectorType &y_expl)
{
  try
    {
      std::string sy = snap_prefix+"y.bin";
      std::string sdot = snap_prefix+"y_dot.bin";
      std::string sexpl = snap_prefix+"y_expl.bin";
      const char *file_y = sy.c_str();
      const char *file_y_dot = sdot.c_str();
      const char *file_y_expl = sexpl.c_str();

      std::ifstream in_y (file_y);
      std::ifstream in_y_dot (file_y_dot);
      std::ifstream in_y_expl (file_y_expl);

      y.block_read(in_y);
      in_y.close();

      y_dot.block_read(in_y_dot);
      in_y_dot.close();

      y_expl.block_read(in_y_expl);
      in_y_expl.close();
    }
  catch (...)
    {
      std::string sy = snap_prefix+"y.bin.old";
      std::string sdot = snap_prefix+"y_dot.bin.old";
      std::string sexpl = snap_prefix+"y_expl.bin.old";
      const char *file_y = sy.c_str();
      const char *file_y_dot = sdot.c_str();
      const char *file_y_expl = sexpl.c_str();

      std::ifstream in_y (file_y);
      std::ifstream in_y_dot (file_y_dot);
      std::ifstream in_y_expl (file_y_expl);

      y.block_read(in_y);
      in_y.close();

      y_dot.block_read(in_y_dot);
      in_y_dot.close();

      y_expl.block_read(in_y_expl);
      in_y_expl.close();
    }

}

// BOOST_CLASS_TRACKING (aspect::Simulator<2>, boost::serialization::track_never)
// BOOST_CLASS_TRACKING (aspect::Simulator<3>, boost::serialization::track_never)

template <int dim, int spacedim, typename LAC>
template <class Archive>
void
piDoMUS<dim,spacedim,LAC>::serialize (Archive &ar, const unsigned int /*version*/)
{
  ar &current_time;
  ar &current_alpha;
  ar &current_dt;
  ar &step_number;
  ar &current_cycle;
}


// instantiate all but serialize()
#define INSTANTIATE(dim,spacedim,LAC) \
  template void piDoMUS<dim,spacedim,LAC>::resume_from_snapshot(); \
  template void piDoMUS<dim,spacedim,LAC>::create_snapshot() const; \
  template void piDoMUS<dim,spacedim,LAC>::save_solutions_and_triangulation(const LADealII::VectorType &y, \
      const LADealII::VectorType &y_dot, \
      const LADealII::VectorType &y_expl, \
      const LADealII::VectorType &, \
      const LADealII::VectorType &) const; \
  template void piDoMUS<dim,spacedim,LAC>::save_solutions_and_triangulation(const LATrilinos::VectorType &, \
      const LATrilinos::VectorType &, \
      const LATrilinos::VectorType &y_expl, \
      const LATrilinos::VectorType &y, \
      const LATrilinos::VectorType &y_dot) const; \
  template void piDoMUS<dim,spacedim,LAC>::load_solutions(LADealII::VectorType &y, \
                                                          LADealII::VectorType &y_expl, \
                                                          LADealII::VectorType &y_dot); \
  template void piDoMUS<dim,spacedim,LAC>::load_solutions(LATrilinos::VectorType &y, \
                                                          LATrilinos::VectorType &y_expl, \
                                                          LATrilinos::VectorType &y_dot);





PIDOMUS_INSTANTIATE(INSTANTIATE)
