/**
 * 
 */
package edu.stanford.rsl.conrad.phantom;

import edu.stanford.rsl.conrad.data.numeric.Grid3D;

/**
 * @author febmeng
 *
 */
public class NumericalSheppLogan3D {

	boolean init = false;
	int nx = 0;
	int ny = 0;
	int nz = 0;
	double wx, wy, wz;
	double dx, dy, dz;
	Grid3D image;
	SheppLogan3D phan;

	public NumericalSheppLogan3D( int maxI, int maxJ, int maxK ){
		
		initSheppLogan3D(  maxI, maxJ, maxK);
		
		if (nx == 0 || ny == 0 || nz == 0 ){
			System.out.println("Errpr: Wrong volume image size!");			
		}
		this.image = new Grid3D(nx, ny, nz);
		
	}
	
	
	public NumericalSheppLogan3D( Grid3D grid ){
		this.image = grid;
		initSheppLogan3D( image.getSize()[0], image.getSize()[1], image.getSize()[2] );
	}
	
	
	void initSheppLogan3D( int maxI, int maxJ, int maxK){

		this.nx = maxI;
		this.ny = maxJ;
		this.nz = maxK;
		wx = ( nx - 1 ) / 2.0 ;
		wy = ( ny - 1 ) / 2.0 ;
		wz = ( nz - 1 ) / 2.0 ;
		dx = 2.0 / ( nx - 1 );
		dy = 2.0 / ( ny - 1 );
		dz = 2.0 / ( nz - 1 );
		
		phan = new SheppLogan3D( ellipsoidsModifid );
		init = true;
	}

	void computeNumericalPhantom( ){
		
		//System.out.print("nx = " + nx + ", ny = " + ny + ", nz = " + nz + " .\n");
		//System.out.print("dx = " + dx + ", dy = " + dy + ", dz = " + dz + " .\n");
		//System.out.print("wx = " + wx + ", wy = " + wy + ", wz = " + wz + " .\n");
		
		for( int ix = 0; ix < nx; ix++ ){
			
			double x = ( ix - wx ) * dx ;
			
			for( int iy = 0; iy < ny; iy++ ){
				
				double y = ( iy - wy ) * dy ;
				
				for( int iz = 0; iz < nz; iz++ ){
					
					double z = ( iz - wz ) * dz ;
					
					image.setAtIndex(ix, iy, iz, (float)phan.ImageDomainSignal(x, y, z));
					
				}
			}
		}
		
	}
	
	public Grid3D getNumericalSheppLoganPhantom(){
		if( init) 
			computeNumericalPhantom();
		return image;
	}
	
	
	private static double[][] ellipsoidsModifid =
	// { delta_x, delta_y, delta_z,        a,       b,       c,            phi,  theta,  psi,     rho }
	 {  {       0,       0,       0,     0.69,    0.92,    0.81,              0,      0,    0,     1 },
		{       0, -0.0184,       0,   0.6624,   0.874,    0.78,              0,      0,    0,   -0.8 },
		{    0.22,       0,       0,     0.11,    0.31,    0.22, -(Math.PI)/10.,      0,    0,   -0.2 },
		{   -0.22,       0,       0,     0.16,    0.41,    0.28,  (Math.PI)/10.,      0,    0,   -0.2 },
		{       0,    0.35,   -0.15,     0.21,    0.25,    0.41,              0,      0,    0,    0.1 },
		{       0,     0.1,    0.25,    0.046,   0.046,    0.05,              0,      0,    0,    0.1 },
		{       0,    -0.1,    0.25,    0.046,   0.046,    0.05,              0,      0,    0,    0.1 },
		{   -0.08,  -0.605,       0,    0.046,   0.023,    0.05,              0,      0,    0,    0.1 },
		{       0,  -0.605,       0,    0.023,   0.023,    0.02,              0,      0,    0,    0.1 },
		{    0.06,  -0.605,       0,    0.023,   0.046,    0.02,              0,      0,    0,    0.1 }};

	
	private static double[][] sphere =
	// { delta_x, delta_y, delta_z,        a,       b,       c,            phi,  theta,  psi,     rho }
	 {  {       0,       0,       0,     0.69,    0.69,    0.69,              0,      0,    0,     1 }};
	
	private static double[][] point =
	// { delta_x, delta_y, delta_z,        a,       b,       c,            phi,  theta,  psi,     rho }
	 {  {       0.6,      0.4,       0,     0.19,    0.19,    0.19,              0,      0,    0,     1 }};
	
	private static double[][] cylinder =
			// { delta_x, delta_y, delta_z,        a,       b,       c,            phi,  theta,  psi,     rho }
	{  {       0,      0,       0,     0.69,    0.69,      10,        0,      0,    0,     1 }};
			

}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/