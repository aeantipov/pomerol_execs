//
// This file is a part of pomerol - a scientific ED code for obtaining 
// properties of a Hubbard model on a finite-size lattice 
//
// Copyright (C) 2010-2014 Andrey Antipov <Andrey.E.Antipov@gmail.com>
// Copyright (C) 2010-2014 Igor Krivenko <Igor.S.Krivenko@gmail.com>
//
// pomerol is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// pomerol is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with pomerol.  If not, see <http://www.gnu.org/licenses/>.


/** \file prog/anderson.cpp
** \brief Diagonalization of the Anderson impurity model (1 impurity coupled to a set of non-interacting bath sites)
**
** \author Andrey Antipov (Andrey.E.Antipov@gmail.com)
*/

#include <boost/serialization/complex.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/lexical_cast.hpp>

#include <string>
#include <iostream>
#include <algorithm>
#include <tclap/CmdLine.h>

#include<cstdlib>
#include <fstream>

#include <pomerol.h>
#include "mpi_dispatcher/mpi_dispatcher.hpp"

#include <alps/params.hpp>
#undef DEBUG
#include <gftools.hpp>
#include <gftools/hdf5.hpp>

void print_section (const std::string& str, boost::mpi::communicator comm = boost::mpi::communicator());

//boost::mpi::environment env;
using namespace Pomerol;
using gftools::grid_object;
using gftools::tools::is_float_equal;
using gftools::fmatsubara_grid;
using gftools::bmatsubara_grid;
using gftools::FMatsubara;
using gftools::BMatsubara;


#define mpi_cout if(!comm.rank()) std::cout 

// cmdline parser
alps::params cmdline_params(int argc, char* argv[]) 
{
    alps::params p(argc, (const char**)argv);
    p.description("Full-ED of the Anderson model");
    p.define <std::vector<double> > ("levels", "energy levels of the bath sites");
    p.define <std::vector<double> > ("hoppings", "hopping to the bath sites");

    p.define<double> ( "beta", 1, "Value of inverse temperature");
    p.define<std::string>("2pgf.terms_res", "File with resonant terms");
    p.define<std::string>("2pgf.terms_nonres", "File with non-resonant terms");

    p.define<int>("2pgf.reduce", 1, "Run a new reduction of terms");
    p.define<double>("2pgf.reduce_tol", 1e-5, "Energy resonance resolution in 2pgf");
    p.define<double>("2pgf.multiterm_tol", 1e-6, "terms with |nominator| < (multiterm_tol / nterms) are thrown away");

    p.define<int>("2pgf.wf_max", 12, "Number of fermionic freqs");
    p.define<int>("2pgf.wb_max", 1,  "Number of bosonic freqs");

    p.define<int>("verbosity", 0, "How verbose should output be");
    p.define<int>("plaintext", 0, "Save plaintext output");
    p.define<std::string>("2pgf.label", "", "Label output 2pgf (in hdf5 + plaintext files)");

    p.define<std::string>("output", "pomerol.h5", "HDF5 archive with 2pgf");

    return p;
}

int main(int argc, char* argv[])
{
    boost::mpi::environment env(argc,argv);
    boost::mpi::communicator comm;
    int rank = comm.rank();
    const int ROOT = 0;

    print_section("Convert terms to 2pgf");
    //double eta, hbw, step; // for evaluation of GF on real axis 

    alps::params p = cmdline_params(argc, argv);
    if (p.help_requested(std::cerr)) { MPI_Finalize(); exit(1); };
    double beta = p["beta"];
    int verbosity = p["verbosity"];
    int plaintext = p["plaintext"];
    std::string label_2pgf=p["2pgf.label"];

    //int wf_max, wb_max;
    
    std::vector<TwoParticleGFPart::ResonantTerm> terms_res;
    std::vector<TwoParticleGFPart::NonResonantTerm> terms_nonres;

    if (rank == ROOT) { 
        {
            std::ifstream term_res_stream(p["2pgf.terms_res"].as<std::string>().c_str());
            boost::archive::text_iarchive oa_res(term_res_stream);
            oa_res & terms_res;
        }
        mpi_cout << terms_res.size() << " resonant terms" << std::endl;

        {
            std::ifstream term_nonres_stream(p["2pgf.terms_nonres"].as<std::string>().c_str());
            boost::archive::text_iarchive oa_nonres(term_nonres_stream);
            oa_nonres & terms_nonres;
        }
        mpi_cout << terms_nonres.size() << " non-resonant terms" << std::endl;

        if (p["2pgf.reduce"].as<int>()) { 
            double nonres_tol = p["2pgf.multiterm_tol"].as<double>() / (terms_nonres.size() + 1.);
            double res_tol = p["2pgf.multiterm_tol"].as<double>() / (terms_res.size() + 1.);
            mpi_cout << "Reducing " << terms_nonres.size() << " + " << terms_res.size() << " terms ..." << std::flush;
            reduceTerms(p["2pgf.reduce_tol"], nonres_tol, res_tol, terms_nonres, terms_res); 
            mpi_cout << "done. Reduced to " << terms_nonres.size() << " + " << terms_res.size() << " terms" << std::endl;

            std::ofstream res_str(("terms_res"+label_2pgf+".pom").c_str());
            std::ofstream nonres_str(("terms_nonres"+label_2pgf+".pom").c_str());
            boost::archive::text_oarchive oa_res(res_str); oa_res << terms_res;
            boost::archive::text_oarchive oa_nonres(nonres_str); oa_nonres << terms_nonres;
            }
        }
    comm.barrier();
    mpi_cout << "Broadcasting terms ..." << std::flush;
    boost::mpi::broadcast(comm, terms_nonres, ROOT);
    boost::mpi::broadcast(comm, terms_res, ROOT);
    comm.barrier();
    mpi_cout << "done" << std::endl;

    struct element_2pgf { 
        typedef std::function<std::complex<double>(std::complex<double>, std::complex<double>, std::complex<double>)> f_type;
        void run() { output_val_ = f_(W_, w1_, w2_); }
        element_2pgf(f_type const& f, std::complex<double> W, std::complex<double> w1, std::complex<double> w2):f_(f), W_(W), w1_(w1), w2_(w2) {}
        std::complex<double> value() { return output_val_; }
        
        f_type const& f_;
        std::complex<double> W_;
        std::complex<double> w1_;
        std::complex<double> w2_;
        std::complex<double> output_val_ = 0.0;
        int complexity = 1;
    };

    int wf_max = p["2pgf.wf_max"];
    int wb_max = p["2pgf.wb_max"];
    fmatsubara_grid fgrid(-wf_max, wf_max, beta);
    bmatsubara_grid bgrid(-wb_max + 1, wb_max, beta);
    typedef grid_object<std::complex<double>, bmatsubara_grid, fmatsubara_grid, fmatsubara_grid> two_pgf_t; 

    two_pgf_t output(std::make_tuple(bgrid, fgrid, fgrid));

    std::vector<std::tuple<bmatsubara_grid::point, fmatsubara_grid::point, fmatsubara_grid::point> > job_freqs;
    size_t njobs = fgrid.size() * fgrid.size() * bgrid.size();
    job_freqs.reserve(njobs);


    pMPI::mpi_skel<element_2pgf> skel;
    skel.parts.reserve(njobs + 1);

    mpi_cout << bgrid.size() << " * " << fgrid.size() << " * " << fgrid.size() << " freqs = " << njobs << " jobs." << std::endl;

    // convert terms to value of the two-particle gf
    two_pgf_t::function_type f = [&](std::complex<double> W, std::complex<double> w1, std::complex<double> w2) -> std::complex<double> {
        std::complex<double> w1p = W + w1;
        std::complex<double> out = 0.0;
        for (auto const& x : terms_res) out+=x(w1p, w2, w1);
        for (auto const& x : terms_nonres) out+=x(w1p, w2, w1);
        return out;
        };

    for (auto W : bgrid.points()) {  
        for (auto w1 : fgrid.points()) {  
            for (auto w2 : fgrid.points()) {  
                job_freqs.push_back( std::make_tuple(W, w1, w2) );
                skel.parts.emplace_back(f, W, w1, w2);
            }
        }
    }

    njobs = job_freqs.size();
    if (output.size() != njobs) throw std::logic_error("Mismatch between njobs and output");

    // do the calculation
    auto job_map = skel.run(comm, bool(verbosity > 0)); 
    comm.barrier();

    // now store the data in the right places of the 2pgf
    for (size_t j = 0; j < njobs; ++j) { 
        output(job_freqs[j]) = (rank == job_map[j])?skel.parts[j].value():0.0;
        }
    comm.barrier();

    boost::mpi::reduce(comm, output.data().data(), njobs, output.data().data(), std::plus<std::complex<double> >(), ROOT); 

    if (rank == ROOT) { 
        alps::hdf5::archive ar(p["output"].as<std::string>(), "w");
        gftools::save_grid_object(ar, "/chi" + label_2pgf, output, plaintext > 0);
    }
}

void print_section (const std::string& str, boost::mpi::communicator comm)
{
    mpi_cout << std::string(str.size(),'=') << std::endl;
    mpi_cout << str << std::endl;
    mpi_cout << std::string(str.size(),'=') << std::endl;
}

