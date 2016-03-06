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

void print_section (const std::string& str, boost::mpi::communicator comm = boost::mpi::communicator());

//boost::mpi::environment env;
using namespace Pomerol;
using gftools::grid_object;
using gftools::tools::is_float_equal;
using gftools::enum_grid;
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

    p.define<double> ( "U", 10.0, "Value of U");
    p.define<double> ( "beta", 1, "Value of inverse temperature");
    p.define<double> ( "ed", 0.0, "Value of energy level of the impurity");

    p.define<int>("calc_gf", false, "Calculate Green's functions");
    p.define<int>("calc_2pgf", false, "Calculate 2-particle Green's functions");

    //p.define<double>("eta", 0.05, "Offset from the real axis for Green's function calculation");

    p.define<double>("2pgf.reduce_tol", 1e-5, "Energy resonance resolution in 2pgf");
    p.define<double>("2pgf.coeff_tol",  1e-12, "Tolerance on nominators in 2pgf");
    p.define<size_t>("2pgf.reduce_freq", 1e5, "How often to reduce terms in 2pgf");
    p.define<double>("2pgf.multiterm_tol", 1e-6, "How often to reduce terms in 2pgf");
    p.define<std::vector<size_t> >("2pgf.indices", "2pgf index combination");

        //cmd.xorAdd(beta_arg,T_arg);
        //TCLAP::ValueArg<size_t> wn_arg("","wf","Number of positive fermionic Matsubara Freqs",false,64,"int",cmd);
        //TCLAP::ValueArg<size_t> wb_arg("","wb","Number of positive bosonic Matsubara Freqs",false,1,"int",cmd);
        //TCLAP::ValueArg<RealType> eta_arg("","eta","Offset from the real axis for Green's function calculation",false,0.05,"RealType",cmd);
        //TCLAP::ValueArg<RealType> hbw_arg("D","hbw","Half-bandwidth. Default = U",false,0.0,"RealType",cmd);
        //TCLAP::ValueArg<RealType> step_arg("","step","Step on a real axis. Default : 0.01",false,0.01,"RealType",cmd);

    return p;
}

int main(int argc, char* argv[])
{
    boost::mpi::environment env(argc,argv);
    boost::mpi::communicator comm;

    print_section("Anderson model ED");
    
    RealType e0, U, beta;
    bool calc_gf, calc_2pgf;
    size_t L;

    double eta, hbw, step; // for evaluation of GF on real axis 

    std::vector<double> levels;
    std::vector<double> hoppings;

    alps::params p = cmdline_params(argc, argv);
    if (p.help_requested(std::cerr)) { MPI_Finalize(); exit(1); };

    U = p["U"];
    e0 = p["ed"];
    boost::tie(beta, calc_gf, calc_2pgf) = boost::make_tuple(  
        p["beta"], p["calc_gf"].as<int>(), p["calc_2pgf"].as<int>());
    calc_gf = calc_gf || calc_2pgf;

    if (p.exists("levels")) { 
        levels = p["levels"].as<std::vector<double>>();
        hoppings = p["hoppings"].as<std::vector<double>>();
    }

    if (levels.size() != hoppings.size()) throw (std::logic_error("number of levels != number of hoppings"));

    L = levels.size();
    INFO("Diagonalization of 1+" << L << " sites");

    /* Add sites */
    Lattice Lat;
    Lat.addSite(new Lattice::Site("A",1,2));
    LatticePresets::addCoulombS(&Lat, "A", U, e0);

    std::vector<std::string> names(L);
    for (size_t i=0; i<L; i++)
        {
            std::stringstream s; s << i; 
            names[i]="b"+s.str();
            Lat.addSite(new Lattice::Site(names[i],1,2));
            LatticePresets::addHopping(&Lat, "A", names[i], hoppings[i]);
            LatticePresets::addLevel(&Lat, names[i], levels[i]);
        };
    
    INFO("Sites");
    Lat.printSites();

    int rank = comm.rank();
    if (!rank) {
        mpi_cout << "Terms with 2 operators" << std::endl;
        Lat.printTerms(2);

        mpi_cout << "Terms with 4 operators" << std::endl;
        Lat.printTerms(4);
        };

    IndexClassification IndexInfo(Lat.getSiteMap());
    // Create index space
    IndexInfo.prepare(false); 
    if (!rank) { print_section("Indices"); IndexInfo.printIndices(); };
    int index_size = IndexInfo.getIndexSize();

    print_section("Matrix element storage");
    IndexHamiltonian Storage(&Lat,IndexInfo); 
    // Write down the Hamiltonian as a symbolic formula
    Storage.prepare(); 
    print_section("Terms");
    if (!rank) INFO(Storage);

    Symmetrizer Symm(IndexInfo, Storage);
    // Find symmetries of the problem
    Symm.compute(); 

    // Introduce Fock space and classify states to blocks
    StatesClassification S(IndexInfo,Symm); 
    S.compute();

    // Hamiltonian in the basis of Fock Space
    Hamiltonian H(IndexInfo, Storage, S); 
    // enter the Hamiltonian matrices
    H.prepare(); 
    // compute eigenvalues and eigenvectors
    H.compute(); 

    // Save spectrum
    if (!rank) {
        grid_object<double, enum_grid> evals1(enum_grid(0, S.getNumberOfStates()));
        RealVectorType evals (H.getEigenValues());
        std::sort(evals.data(), evals.data() + H.getEigenValues().size());
        std::copy(evals.data(), evals.data() + S.getNumberOfStates(), evals1.data().data());
        evals1.savetxt("spectrum.dat");
    }
    //savetxt("spectrum.dat", evals); // dump eigenvalues

    DensityMatrix rho(S,H,beta); // create Density Matrix
    rho.prepare();
    rho.compute(); // evaluate thermal weights with respect to ground energy, i.e exp(-beta(e-e_0))/Z 

    ParticleIndex d0 = IndexInfo.getIndex("A",0,down); // find the indices of the impurity, i.e. spin up index
    ParticleIndex u0 = IndexInfo.getIndex("A",0,up);

    if (!rank) { 
        // get average total particle number
        mpi_cout << "<N> = " << rho.getAverageOccupancy() << std::endl; 
        // get average energy
        mpi_cout << "<H> = " << rho.getAverageEnergy() << std::endl; 
        // get double occupancy
        mpi_cout << "<N_{" << IndexInfo.getInfo(u0) << "}N_{"<< IndexInfo.getInfo(u0) << "}> = " << rho.getAverageDoubleOccupancy(u0,d0) << std::endl; 
        // get average total particle number per index
        for (ParticleIndex i=0; i<IndexInfo.getIndexSize(); i++) {  
            std::cout << "<N_{" << IndexInfo.getInfo(i) << "[" << i <<"]}> = " << rho.getAverageOccupancy(i) << std::endl; 
            }
        double n_av = rho.getAverageOccupancy();
        gftools::num_io<double>(n_av).savetxt("N_T.dat");
        }

    // Green's function calculation starts here
    
    FieldOperatorContainer Operators(IndexInfo, S, H); // Create a container for c and c^+ in the eigenstate basis

    if (calc_gf) {
        INFO("1-particle Green's functions calc");
        std::set<ParticleIndex> f; // a set of indices to evaluate c and c^+
        std::set<IndexCombination2> indices2; // a set of pairs of indices to evaluate Green's function

        // Take only impurity spin up and spin down indices
        f.insert(u0); 
        f.insert(d0);
        indices2.insert(IndexCombination2(d0,d0)); // evaluate only G_{\down \down}

        Operators.prepareAll(f); 
        Operators.computeAll(); // evaluate c, c^+ for chosen indices 

        GFContainer G(IndexInfo,S,H,rho,Operators);

        G.prepareAll(indices2); // identify all non-vanishing block connections in the Green's function
        G.computeAll(); // Evaluate all GF terms, i.e. resonances and weights of expressions in Lehmans representation of the Green's function
/*
        if (!comm.rank()) // dump gf into a file
        // loops over all components (pairs of indices) of the Green's function 
        for (std::set<IndexCombination2>::const_iterator it = indices2.begin(); it != indices2.end(); ++it) { 
            IndexCombination2 ind2 = *it;
            // Save Matsubara GF from pi/beta to pi/beta*(4*wf_max + 1)
            mpi_cout << "Saving imfreq G" << ind2 << " on "<< 4*wf_max << " Matsubara freqs. " << std::endl;
            std::ofstream gw_im(("gw_imag"+ boost::lexical_cast<std::string>(ind2.Index1)+ boost::lexical_cast<std::string>(ind2.Index2)+".dat").c_str());
            const GreensFunction & GF = G(ind2);
            for (int wn = 0; wn < wf_max*4; wn++) {
                ComplexType val = GF(I*FMatsubara(wn,beta)); // this comes from Pomerol - see GreensFunction::operator()
                gw_im << std::scientific << std::setprecision(12) << FMatsubara(wn,beta) << "   " << real(val) << " " << imag(val) << std::endl;
            };
            gw_im.close();
            // Save Retarded GF on the real axis
            std::ofstream gw_re(("gw_real"+boost::lexical_cast<std::string>(ind2.Index1)+boost::lexical_cast<std::string>(ind2.Index2)+".dat").c_str()); 
            mpi_cout << "Saving real-freq GF " << ind2 << " in energy space [" << e0-hbw << ":" << e0+hbw << ":" << step << "] + I*" << eta << "." << std::endl;
            for (double w = e0-hbw; w < e0+hbw; w+=step) {
                ComplexType val = GF(ComplexType(w) + I*eta);
                gw_re << std::scientific << std::setprecision(12) << w << "   " << real(val) << " " << imag(val) << std::endl;
            };
            gw_re.close();
        }
*/

        // Start Two-particle GF calculation

        if (calc_2pgf) {   
            print_section("2-Particle Green's function calc");

            std::vector<size_t> indices_2pgf = p["2pgf.indices"].as<std::vector<size_t> >();
            if (indices_2pgf.size() != 4) throw std::logic_error("Need 4 indices for 2pgf");
    
            // a set of four indices to evaluate the 2pgf
            IndexCombination4 index_comb(indices_2pgf[0], indices_2pgf[1], indices_2pgf[2], indices_2pgf[3]);

            std::set<IndexCombination4> indices4; 
            // 2PGF = <T c c c^+ c^+>
            indices4.insert(index_comb);
            std::string ind_str = boost::lexical_cast<std::string>(index_comb.Index1) 
                                + boost::lexical_cast<std::string>(index_comb.Index2) 
                                + boost::lexical_cast<std::string>(index_comb.Index3) 
                                + boost::lexical_cast<std::string>(index_comb.Index4);

            TwoParticleGFContainer Chi4(IndexInfo,S,H,rho,Operators);
            /* Some knobs to make calc faster - the larger the values of tolerances, the faster is calc, but rounding errors may show up. */
            /** A difference in energies with magnitude less than this value is treated as zero - resolution of energy resonances. */
            Chi4.ReduceResonanceTolerance = p["2pgf.reduce_tol"].as<double>();
            /** Minimal magnitude of the coefficient of a term to take it into account - resolution of thermal weight. */
            Chi4.CoefficientTolerance = p["2pgf.coeff_tol"].as<double>();
            /** Knob that controls the caching frequency. */
            Chi4.ReduceInvocationThreshold = p["2pgf.reduce_freq"].as<size_t>();
            /** Minimal magnitude of the coefficient of a term to take it into account with respect to amount of terms. */
            Chi4.MultiTermCoefficientTolerance = p["2pgf.multiterm_tol"].as<double>();
            
            Chi4.prepareAll(indices4); // find all non-vanishing block connections inside 2pgf
            comm.barrier(); // MPI::BARRIER

            // ! The most important routine - actually calculate the 2PGF
            Chi4.computeAll(comm, true); 

            // dump 2PGF into files - loop through 2pgf components
            if (!comm.rank()) mpi_cout << "Saving 2PGF " << index_comb << std::endl;
            const TwoParticleGF &chi = Chi4(index_comb);

            // Save terms of two particle GF
            std::ofstream term_res_stream(("terms_res"+ind_str+".pom").c_str());
            std::ofstream term_nonres_stream(("terms_nonres"+ind_str+".pom").c_str());
            boost::archive::text_oarchive oa_res(term_res_stream);
            boost::archive::text_oarchive oa_nonres(term_nonres_stream);
            for(std::vector<TwoParticleGFPart*>::const_iterator iter = chi.parts.begin(); iter != chi.parts.end(); iter++) {
                oa_nonres << ((*iter)->getNonResonantTerms());
                oa_res << ((*iter)->getResonantTerms());
                };
            }
        }
}

void print_section (const std::string& str, boost::mpi::communicator comm)
{
    mpi_cout << std::string(str.size(),'=') << std::endl;
    mpi_cout << str << std::endl;
    mpi_cout << std::string(str.size(),'=') << std::endl;
}

