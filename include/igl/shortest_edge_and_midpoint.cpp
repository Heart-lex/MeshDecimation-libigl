// This file is part of libigl, a simple c++ geometry processing library.
// 
// Copyright (C) 2016 Alec Jacobson <alecjacobson@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.
#include "shortest_edge_and_midpoint.h"

// Global to influence cost 
static Eigen::VectorXd g_weights;

namespace igl
{
	void set_weights(Eigen::VectorXd& W) {
		g_weights = W;
	}
}


IGL_INLINE void igl::shortest_edge_and_midpoint(
	const int e,
	const Eigen::MatrixXd& V,
	const Eigen::MatrixXi& /*F*/,
	const Eigen::MatrixXi& E,
	const Eigen::VectorXi& /*EMAP*/,
	const Eigen::MatrixXi& EF,
	const Eigen::MatrixXi& /*EI*/,
	double& cost,
	Eigen::RowVectorXd& p)
{

	// Original code:
	//cost = (V.row(E(e,0))-V.row(E(e,1))).norm();
	//p = 0.5*(V.row(E(e,0))+V.row(E(e,1)));

	// Compute the cost of collapsing edge e to its midpoint
	// and return the midpoint in p.
	// The cost is defined as the length of the edge multiplied by a factor
	// that prioritizes higher points for collapse.

	// Initialize average weight accumulator
	double avg_weight = 0.0;
	int count = 0;

	// EF(e,0) and EF(e,1) are the indices of the two faces adjacent to edge 'e'
	int f0 = EF(e, 0);
	int f1 = EF(e, 1);

	// Accumulate the weight of each adjacent face if it exists (not -1)
	if (f0 != -1) {
		avg_weight += g_weights(f0);  // Add the weight of face f0
		count++;                       // Keep track of how many faces contributed
	}
	if (f1 != -1) {
		avg_weight += g_weights(f1);  // Add the weight of face f1
		count++;
	}

	// Compute the average weight over the contributing faces
	if (count > 0)
		avg_weight /= count;

	// Invert the height to prioritize higher points for collapse
	cost = 1.0 + avg_weight;

	// Compute the midpoint as usual
	p = 0.5 * (V.row(E(e, 0)) + V.row(E(e, 1)));
}