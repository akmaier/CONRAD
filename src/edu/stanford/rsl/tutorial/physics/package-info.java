/*
 * Copyright (C) 2010-2014 Andreas Maier, Tobias Miksch, Fabian R�ckert
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.tutorial.physics;
/**
 * This package contains different algorithms for the calculation of X-Ray propagation. 
 * The focus lies thereby on the propagation of scattered rays throughout all media.
 * 
 * The main method is located in the file XRayTracer.java where different setting are already set but can be adjusted to fit the need of the user:
 *  - number of rays traced (main component for the calculation time and the precision of the result)
 *  - energy of the photon-emitter (responsible for the physical effects)
 *  - maximal length of a light ray (or maximal number of interactions before a photon is dismissed)
 *
 * On execution the user will always be queried if the results should only be visualized or an additional file should be created (and the location of the file)
 * And which algorithm should be used. The implemented methods are ray tracing, bidirectional path tracing, virtual point lights and virtual ray lights.
 *
 * Files created by this method can than be compared and analyzed with the XRayComperator. It can compare the values, the absolute values or use the 
 *  Root-Mean-Square Error or structural similarity index(SSIM) methods. 
 * 
 * @author Tobias Miksch based on the work of Fabian R�ckert
 */