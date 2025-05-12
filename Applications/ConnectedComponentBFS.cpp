/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.6 -------------------------------------------------*/
/* date: 6/15/2017 ---------------------------------------------*/
/* authors: Ariful Azad, Aydin Buluc  --------------------------*/
/****************************************************************/
/*
 Copyright (c) 2010-2017, The Regents of the University of California
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 */

#include <cstdint>
#define DETERMINISTIC
#include "CombBLAS/CombBLAS.h"
#include <mpi.h>
#include <sys/time.h> 
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>

#include <kagen.h>

int cblas_splits;

double cblas_alltoalltime;
double cblas_allgathertime;
double cblas_mergeconttime;
double cblas_transvectime;
double cblas_localspmvtime;

#define ITERS 1
#define EDGEFACTOR 16
using namespace std;
using namespace combblas;

// 64-bit floor(log2(x)) function 
// note: least significant bit is the "zeroth" bit
// pre: v > 0
unsigned int highestbitset(uint64_t v)
{
	// b in binary is {10,1100, 11110000, 1111111100000000 ...}  
	const uint64_t b[] = {0x2ULL, 0xCULL, 0xF0ULL, 0xFF00ULL, 0xFFFF0000ULL, 0xFFFFFFFF00000000ULL};
	const unsigned int S[] = {1, 2, 4, 8, 16, 32};
	int i;

	unsigned int r = 0; // result of log2(v) will go here
	for (i = 5; i >= 0; i--) 
	{
		if (v & b[i])	// highestbitset is on the left half (i.e. v > S[i] for sure)
		{
			v >>= S[i];
			r |= S[i];
		} 
	}
	return r;
}

template <class T>
bool from_string(T & t, const string& s, std::ios_base& (*f)(std::ios_base&))
{
        istringstream iss(s);
        return !(iss >> f >> t).fail();
}


template <typename PARMAT>
void Symmetricize(PARMAT & A)
{
	// boolean addition is practically a "logical or"
	// therefore this doesn't destruct any links
	PARMAT AT = A;
	AT.Transpose();
	A += AT;
}

/**
 * Binary function to prune the previously discovered vertices from the current frontier 
 * When used with EWiseApply(SparseVec V, DenseVec W,...) we get the 'exclude = false' effect of EWiseMult
**/
struct prunediscovered: public std::binary_function<int64_t, int64_t, int64_t >
{
  	int64_t operator()(int64_t x, const int64_t & y) const
	{
		return ( y == -1 ) ? x: -1;
	}
};

int main(int argc, char* argv[])
{
	int nprocs, myrank;
#ifdef _OPENMP
	int cblas_splits = omp_get_max_threads();
    	int provided, flag, claimed;
    	MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided );
    	MPI_Is_thread_main( &flag );
    	if (!flag)
        	SpParHelper::Print("This thread called init_thread but Is_thread_main gave false\n");
    	MPI_Query_thread( &claimed );
    	if (claimed != provided)
        	SpParHelper::Print("Query thread gave different thread level than requested\n");
#else
	MPI_Init(&argc, &argv);
	int cblas_splits = 1;	
#endif
    
	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
	if(argc < 3)
	{
		if(myrank == 0)
		{
			cout << "Usage: ./run_cc <Force,Input> <Scale Forced | Input Name> {FastGen}" << endl;
			cout << "Example: ./run_cc Force 25 FastGen" << endl;
		}
		MPI_Finalize();
		return -1;
	}		
	{
		typedef SelectMaxSRing<bool, int32_t> SR;
		typedef SpParMat < int64_t, bool, SpDCCols<int64_t,bool> > PSpMat_Bool;
		typedef SpParMat < int64_t, bool, SpDCCols<int32_t,bool> > PSpMat_s32p64;	// sequentially use 32-bits for local matrices, but parallel semantics are 64-bits
		typedef SpParMat < int64_t, int, SpDCCols<int32_t,int> > PSpMat_s32p64_Int;	// similarly mixed, but holds integers as upposed to booleans
		typedef SpParMat < int64_t, int64_t, SpDCCols<int64_t,int64_t> > PSpMat_Int64;
		shared_ptr<CommGrid> fullWorld;
		fullWorld.reset( new CommGrid(MPI_COMM_WORLD, 0, 0) );

		// Declare objects
		PSpMat_Bool A(fullWorld);
		PSpMat_s32p64 Aeff(fullWorld);
		FullyDistVec<int64_t, int64_t> degrees(fullWorld);	// degrees of vertices (including multi-edges and self-loops)
		FullyDistVec<int64_t, int64_t> nonisov(fullWorld);	// id's of non-isolated (connected) vertices
		unsigned scale;
		OptBuf<int32_t, int64_t> optbuf;	// let indices be 32-bits
		bool scramble = false;

		if(string(argv[1]) == string("Input")) // input option
		{
			// A.ReadDistribute(string(argv[2]), 0);	// read it from file
			auto mapping = A.ReadGeneralizedTuples(string(argv[2]), maximum<bool>());
			SpParHelper::Print("Read input\n");


			A.PrintInfo();
			
			PSpMat_Int64 * G = new PSpMat_Int64(A); 
			G->Reduce(degrees, Row, plus<int64_t>(), static_cast<int64_t>(0));	// identity is 0 
			delete G;

			std::cout << "degrees in input\n";
			degrees.DebugPrint();

			Symmetricize(A);	// A += A';
			FullyDistVec<int64_t, int64_t> * ColSums = new FullyDistVec<int64_t, int64_t>(A.getcommgrid());
			A.Reduce(*ColSums, Column, plus<int64_t>(), static_cast<int64_t>(0)); 	// plus<int64_t> matches the type of the output vector
			nonisov = ColSums->FindInds(bind2nd(greater<int64_t>(), 0));	// only the indices of non-isolated vertices
			bool zero_vertex_present = nonisov.Count([](int64_t x) { return x == 0;}) > 0 ? true : false;
			if (zero_vertex_present) {
				std::cout << "Vertex 0 present!\n";
			} else {
				std::cout << "Vertex 0 not present!\n";
				throw std::domain_error("Vertex 0 is isolated!");
			}
			delete ColSums;
			// A.PrintInfo();
			A = A(nonisov, nonisov);
			// A.PrintInfo();
			Aeff = PSpMat_s32p64(A);
			A.FreeMemory();
			SpParHelper::Print("Symmetricized and pruned\n");
			std::cout << "Aeff\n";
			Aeff.PrintInfo();
                        Aeff.OptimizeForGraph500(optbuf);               // Should be called before threading is activated
                #ifdef THREADED
                        ostringstream tinfo;
                        tinfo << "Threading activated with " << cblas_splits << " threads" << endl;
                        SpParHelper::Print(tinfo.str());
                        Aeff.ActivateThreading(cblas_splits);
                #endif
		}
		else if(string(argv[1]) == string("KaGen"))
		{
			if(argc < 3) {
				if(myrank == 0) {
					cout << "Usage for KaGen: ./run_cc KaGen <options>" << endl;
					cout << "Example: ./run_cc KaGen 'rmat;N=12;M=15'" << endl;
				}
				MPI_Finalize();
				return -1;
			}

			string options = string(argv[2]);
			ostringstream outs;
			outs << "Generating graph with KaGen using options: " << options << endl;
			SpParHelper::Print(outs.str());
			// Initialize KaGen
			kagen::KaGen gen(MPI_COMM_WORLD);
			gen.SetSeed(1); // Use deterministic seed for reproducibility
			gen.EnableUndirectedGraphVerification();
			gen.UseEdgeListRepresentation();

			// Generate graph using option string
			kagen::Graph graph = gen.GenerateFromOptionString(options);

			// Convert KaGen graph to CombBLAS format

			SpParHelper::Print("Generated graph with KaGen\n");

			// Convert to sparse matrix format

			auto globaln = graph.NumberOfGlobalVertices();
			auto local_edges = graph.TakeEdges<int64_t>();
			
			DistEdgeList<int64_t> *DEL = new DistEdgeList<int64_t>(fullWorld->GetWorld(), globaln, local_edges);
			auto del_edges = DEL->getEdges();
			for (int i = 0; i < DEL->getNumLocalEdges(); i++) {
				std::cout << "Edge " << i << ": " << del_edges[2 * i] << " " << del_edges[2 * i + 1] << std::endl;
			}
			A = PSpMat_Bool(*DEL, false);
			delete DEL;
			
			PSpMat_Int64 * G = new PSpMat_Int64(A); 
			G->Reduce(degrees, Row, plus<int64_t>(), static_cast<int64_t>(0));	// identity is 0 
			delete G;

			FullyDistVec<int64_t, int64_t> * ColSums = new FullyDistVec<int64_t, int64_t>(A.getcommgrid());
			A.Reduce(*ColSums, Column, plus<int64_t>(), static_cast<int64_t>(0)); 	// plus<int64_t> matches the type of the output vector
			nonisov = ColSums->FindInds(bind2nd(greater<int64_t>(), 0));

			std::cout << "degrees in kagen\n";
			degrees.DebugPrint();
			ostringstream graphinfo;
			graphinfo << "Converted to Boolean matrix" << endl;
			SpParHelper::Print(graphinfo.str());
			A.PrintInfo();

			Aeff = PSpMat_s32p64(A);
			A.FreeMemory();
			std::cout << "Aeff\n";
			Aeff.PrintInfo();
			SpParHelper::Print("Symmetricized\n");

			Aeff.OptimizeForGraph500(optbuf);		// Should be called before threading is activated
#ifdef THREADED
			ostringstream tinfo;
			tinfo << "Threading activated with " << cblas_splits << " threads" << endl;
			SpParHelper::Print(tinfo.str());
			Aeff.ActivateThreading(cblas_splits);
#endif
		}
		else if(string(argv[1]) == string("Binary"))
		{
			uint64_t n, m;
			from_string(n,string(argv[3]),std::dec);
			from_string(m,string(argv[4]),std::dec);
			
			ostringstream outs;
			outs << "Reading " << argv[2] << " with " << n << " vertices and " << m << " edges" << endl;
			SpParHelper::Print(outs.str());
			DistEdgeList<int64_t> * DEL = new DistEdgeList<int64_t>(argv[2], n, m);
			SpParHelper::Print("Read binary input to distributed edge list\n");

			PermEdges(*DEL);
			SpParHelper::Print("Permuted Edges\n");

			RenameVertices(*DEL);	
			//DEL->Dump32bit("graph_permuted");
			SpParHelper::Print("Renamed Vertices\n");

			// conversion from distributed edge list, keeps self-loops, sums duplicates
			PSpMat_Int64 * G = new PSpMat_Int64(*DEL, false); 
			delete DEL;	// free memory before symmetricizing
			SpParHelper::Print("Created Int64 Sparse Matrix\n");

			G->Reduce(degrees, Row, plus<int64_t>(), static_cast<int64_t>(0));	// Identity is 0 

			A =  PSpMat_Bool(*G);			// Convert to Boolean
			delete G;
			int64_t removed  = A.RemoveLoops();

			ostringstream loopinfo;
			loopinfo << "Converted to Boolean and removed " << removed << " loops" << endl;
			SpParHelper::Print(loopinfo.str());
			A.PrintInfo();

			FullyDistVec<int64_t, int64_t> * ColSums = new FullyDistVec<int64_t, int64_t>(A.getcommgrid());
			FullyDistVec<int64_t, int64_t> * RowSums = new FullyDistVec<int64_t, int64_t>(A.getcommgrid());
			A.Reduce(*ColSums, Column, plus<int64_t>(), static_cast<int64_t>(0)); 	
			A.Reduce(*RowSums, Row, plus<int64_t>(), static_cast<int64_t>(0)); 	
			ColSums->EWiseApply(*RowSums, plus<int64_t>());
			delete RowSums;

			nonisov = ColSums->FindInds(bind2nd(greater<int64_t>(), 0));	// only the indices of non-isolated vertices
			delete ColSums;

			SpParHelper::Print("Found (and permuted) non-isolated vertices\n");	
			nonisov.RandPerm();	// so that A(v,v) is load-balanced (both memory and time wise)
			A.PrintInfo();
		#ifndef NOPERMUTE
			A(nonisov, nonisov, true);	// in-place permute to save memory
			SpParHelper::Print("Dropped isolated vertices from input\n");	
			A.PrintInfo();
		#endif
			Aeff = PSpMat_s32p64(A);	// Convert to 32-bit local integers
			A.FreeMemory();

			Symmetricize(Aeff);	// A += A';
			SpParHelper::Print("Symmetricized\n");	
			//A.Dump("graph_symmetric");

	                Aeff.OptimizeForGraph500(optbuf);		// Should be called before threading is activated
		#ifdef THREADED	
			ostringstream tinfo;
			tinfo << "Threading activated with " << cblas_splits << " threads" << endl;
			SpParHelper::Print(tinfo.str());
			Aeff.ActivateThreading(cblas_splits);	
		#endif
		}
		else 
		{	
			if(string(argv[1]) == string("Force"))	
			{
				scale = static_cast<unsigned>(atoi(argv[2]));
				ostringstream outs;
				outs << "Forcing scale to : " << scale << endl;
				SpParHelper::Print(outs.str());

				if(argc > 3 && string(argv[3]) == string("FastGen"))
				{
					SpParHelper::Print("Using fast vertex permutations; skipping edge permutations (like v2.1)\n");	
					scramble = true;
				}
			}
			else
			{
				SpParHelper::Print("Unknown option\n");
				MPI_Finalize();
				return -1;	
			}
			// this is an undirected graph, so A*x does indeed BFS
 			double initiator[4] = {.57, .19, .19, .05};

			double t01 = MPI_Wtime();
			double t02;
			DistEdgeList<int64_t> * DEL = new DistEdgeList<int64_t>();
			if(!scramble)
			{
				DEL->GenGraph500Data(initiator, scale, EDGEFACTOR);
				SpParHelper::Print("Generated edge lists\n");
				t02 = MPI_Wtime();
				ostringstream tinfo;
				tinfo << "Generation took " << t02-t01 << " seconds" << endl;
				SpParHelper::Print(tinfo.str());
		
				PermEdges(*DEL);
				SpParHelper::Print("Permuted Edges\n");
				//DEL->Dump64bit("edges_permuted");
				//SpParHelper::Print("Dumped\n");

				RenameVertices(*DEL);	// intermediate: generates RandPerm vector, using MemoryEfficientPSort
				SpParHelper::Print("Renamed Vertices\n");
			}
			else	// fast generation
			{
				DEL->GenGraph500Data(initiator, scale, EDGEFACTOR, true, true );	// generate packed edges
				SpParHelper::Print("Generated renamed edge lists\n");
				t02 = MPI_Wtime();
				ostringstream tinfo;
				tinfo << "Generation took " << t02-t01 << " seconds" << endl;
				SpParHelper::Print(tinfo.str());
			}

			// Start Kernel #1
			MPI_Barrier(MPI_COMM_WORLD);
			double t1 = MPI_Wtime();

			// conversion from distributed edge list, keeps self-loops, sums duplicates
			PSpMat_s32p64_Int * G = new PSpMat_s32p64_Int(*DEL, false); 
			delete DEL;	// free memory before symmetricizing
			SpParHelper::Print("Created Sparse Matrix (with int32 local indices and values)\n");

			MPI_Barrier(MPI_COMM_WORLD);
			double redts = MPI_Wtime();
			G->Reduce(degrees, Row, plus<int64_t>(), static_cast<int64_t>(0));	// Identity is 0 
			MPI_Barrier(MPI_COMM_WORLD);
			double redtf = MPI_Wtime();

			ostringstream redtimeinfo;
			redtimeinfo << "Calculated degrees in " << redtf-redts << " seconds" << endl;
			SpParHelper::Print(redtimeinfo.str());
			A =  PSpMat_Bool(*G);			// Convert to Boolean
			delete G;
			int64_t removed  = A.RemoveLoops();

			ostringstream loopinfo;
			loopinfo << "Converted to Boolean and removed " << removed << " loops" << endl;
			SpParHelper::Print(loopinfo.str());
			A.PrintInfo();

			FullyDistVec<int64_t, int64_t> * ColSums = new FullyDistVec<int64_t, int64_t>(A.getcommgrid());
			FullyDistVec<int64_t, int64_t> * RowSums = new FullyDistVec<int64_t, int64_t>(A.getcommgrid());
			A.Reduce(*ColSums, Column, plus<int64_t>(), static_cast<int64_t>(0)); 	
			A.Reduce(*RowSums, Row, plus<int64_t>(), static_cast<int64_t>(0)); 	
			SpParHelper::Print("Reductions done\n");
			ColSums->EWiseApply(*RowSums, plus<int64_t>());
			delete RowSums;
			SpParHelper::Print("Intersection of colsums and rowsums found\n");

			nonisov = ColSums->FindInds(bind2nd(greater<int64_t>(), 0));	// only the indices of non-isolated vertices
			delete ColSums;

			SpParHelper::Print("Found (and permuted) non-isolated vertices\n");	
			nonisov.RandPerm();	// so that A(v,v) is load-balanced (both memory and time wise)
			A.PrintInfo();
		#ifndef NOPERMUTE
			A(nonisov, nonisov, true);	// in-place permute to save memory	
			SpParHelper::Print("Dropped isolated vertices from input\n");	
			A.PrintInfo();
		#endif
		
			Aeff = PSpMat_s32p64(A);	// Convert to 32-bit local integers
			A.FreeMemory();
			Symmetricize(Aeff);	// A += A';
			SpParHelper::Print("Symmetricized\n");	

	                Aeff.OptimizeForGraph500(optbuf);		// Should be called before threading is activated
		#ifdef THREADED	
			ostringstream tinfo;
			tinfo << "Threading activated with " << cblas_splits << " threads" << endl;
			SpParHelper::Print(tinfo.str());
			Aeff.ActivateThreading(cblas_splits);	
		#endif
			Aeff.PrintInfo();
			
			MPI_Barrier(MPI_COMM_WORLD);
			double t2=MPI_Wtime();
			
			ostringstream k1timeinfo;
			k1timeinfo << (t2-t1) - (redtf-redts) << " seconds elapsed for Kernel #1" << endl;
			SpParHelper::Print(k1timeinfo.str());
		}
		Aeff.PrintInfo();
		float balance = Aeff.LoadImbalance();
		ostringstream outs;
		outs << "Load balance: " << balance << endl;
		SpParHelper::Print(outs.str());

		MPI_Barrier(MPI_COMM_WORLD);
		double t1 = MPI_Wtime();

		// Now that every remaining vertex is non-isolated, randomly pick ITERS many of them as starting vertices
		degrees.DebugPrint();
		FullyDistVec<int64_t, int64_t> Cands(ITERS, 0);
		double nver = (double) degrees.TotalLength();

#ifdef DETERMINISTIC
		MTRand M(1);
#else
		MTRand M;	// generate random numbers with Mersenne Twister 
#endif
		vector<double> loccands(ITERS);
		vector<int64_t> loccandints(ITERS);
		if(myrank == 0)
		{
			for(int i=0; i<ITERS; ++i)
				loccands[i] = M.rand();
			loccands[0] = 0;
			copy(loccands.begin(), loccands.end(), ostream_iterator<double>(cout," ")); cout << endl;
			transform(loccands.begin(), loccands.end(), loccands.begin(), bind2nd( multiplies<double>(), nver ));
			
			for(int i=0; i<ITERS; ++i)
				loccandints[i] = static_cast<int64_t>(loccands[i]);
			copy(loccandints.begin(), loccandints.end(), ostream_iterator<double>(cout," ")); cout << endl;
		}
		MPI_Bcast(&(loccandints[0]), ITERS, MPIType<int64_t>(),0,MPI_COMM_WORLD);
		for(int i=0; i<ITERS; ++i)
		{
			Cands.SetElement(i,loccandints[i]);
		}

		MPI_Pcontrol(1,"BFSCC");
        double MTEPS[ITERS]; double INVMTEPS[ITERS]; double TIMES[ITERS]; double EDGES[ITERS];
        
		FullyDistVec<int64_t, int64_t> parents ( Aeff.getcommgrid(), Aeff.getncol(), (int64_t) -1);	// identity is -1
		uint64_t num_components = 0;
		double cc_start = MPI_Wtime();
		std::cout << "nver: " << nver << "\n";
		std::cout << "nonisov: ";
		nonisov.DebugPrint();
        for(int vertex = 0; vertex < nver;) 
        {
			cblas_allgathertime = 0;
			cblas_alltoalltime = 0;
			cblas_mergeconttime = 0;
			cblas_transvectime  = 0;
			cblas_localspmvtime = 0;
			MPI_Barrier(MPI_COMM_WORLD);
			double bfs_iteration_start = MPI_Wtime();
			++num_components;

			FullyDistSpVec<int64_t, int64_t> fringe(Aeff.getcommgrid(), Aeff.getncol());	// numerical values are stored 0-based
			fringe.SetElement(static_cast<int64_t>(vertex), static_cast<int64_t>(vertex));
			// std::cout << "Frontier before running BFS\n";
			// fringe.DebugPrint(); 
			while(fringe.getnnz() > 0)
			{
				std::cout << "fringe before running BFS\n";
				fringe.DebugPrint();
				fringe.setNumToInd();
				fringe = SpMV(Aeff, fringe,optbuf);	// SpMV with sparse vector (with indexisvalue flag preset), optimization enabled
				fringe = EWiseMult(fringe, parents, true, (int64_t) -1);	// clean-up vertices that already has parents 
				std::cout << "fringe after running BFS\n";
				fringe.DebugPrint();
				std::cout << "parents before running BFS\n";
				parents.DebugPrint();
				parents.Set(fringe);
				std::cout << "parents after running BFS\n";
				parents.DebugPrint();
			}
			MPI_Barrier(MPI_COMM_WORLD);
			double bfs_iteration_end = MPI_Wtime();
			// std::cout << "Frontier after BFS\n";
			// fringe.DebugPrint();
			// FullyDistSpVec<int64_t, int64_t> parentsp = parents.Find(bind2nd(greater<int64_t>(), -1));
			parents.DebugPrint();
			auto [next_vertex, parent] = parents.MinElement();
			while (next_vertex == vertex) {
				parents.SetElement(vertex, vertex);
				std::tie(next_vertex, parent) = parents.MinElement();
				if (parent != -1) {
					break;
				}
			}
			if (parent != -1) {
				break;
			}
			vertex = next_vertex;
			std::cout << "next_vertex: " << next_vertex << "\n";

			ostringstream outnew;
			outnew << "BFS iteration time: " << bfs_iteration_end - bfs_iteration_start << " seconds" << endl;
			outnew << "Communication per iteration: " << (cblas_allgathertime + cblas_alltoalltime) << endl;
			SpParHelper::Print(outnew.str());
        }
		double cc_end = MPI_Wtime();
        SpParHelper::Print("Finished\n");
        SpParHelper::Print("Number of components: " + to_string(num_components) + "\n");
        ostringstream os;
        MPI_Pcontrol(-1,"BFSCC");

		os << "CC runtime: " << (cc_end - cc_start) << "\n";
        SpParHelper::Print(os.str());
	}
	MPI_Finalize();
	return 0;
}

