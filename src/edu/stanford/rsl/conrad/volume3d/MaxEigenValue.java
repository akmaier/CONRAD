package edu.stanford.rsl.conrad.volume3d;

import java.util.Arrays;


public class MaxEigenValue implements Runnable {

	private int x = -1, y, z;
	private float [][][] T;
	private int dimensions;
	private float [] eigmax;
	private boolean finished = true;
	private boolean data = false;
	private boolean terminate = false;

	public MaxEigenValue(int dimensions){
		this.dimensions = dimensions;
		data = false;
		finished = true;
	}

	public void terminate (){
		terminate = true;
	}

	public void setData(int x, int y, int z, float [][][] T){
		this.x = x;
		this.y = y;
		this.z = z;
		this.T = T;
		data = true;
		finished = false;
		eigmax = new float[z];
	}

	public boolean done(){
		return finished;
	}

	public void getData(float [][][] array){
		if (x >= 0){
			for (int i = 0; i< z; i++){
				array[x][y][i] = eigmax[i];
			}
		}
	}

	private void nrerror(Object s){
		//TODO: nothing
		//System.out.println("Jacobi " +x + " " + y + " "+ z + ": " +s.toString());
	}

	private static void ROTATE(float [][] a, int i, int j, int k, int l, float s, float tau) {
		float g = a[i][j];
		float h = a[k][l];
		a[i][j]=g-s*(h+g*tau);
		a[k][l]=h+s*(g-h*tau);
	}

	private static float fabs(float in){
		return Math.abs(in);
	}

	private static float sqrt(double in){
		return (float) Math.sqrt(in);
	}

	public static float [] vector(int i, int n){
		float [] revan = new float [n];
		Arrays.fill(revan, i);
		return revan;
	}

	public static void jacobi(float [][] a, int n, float []d, float [][]v, Integer nrot){
		int j,iq,ip,i;
		float tresh,theta,tau,t,sm,s,h,g,c;
		float [] b;
		float [] z;

		b=vector(1,n);
		z=vector(1,n);
		for (ip=0;ip<n;ip++) {
			for (iq=0;iq<n;iq++) v[ip][iq]=0.0f;
			v[ip][ip]=1.0f;
		}
		for (ip=0;ip<n;ip++) {
			b[ip]=d[ip]=a[ip][ip];
			z[ip]=0.0f;
		}
		nrot=0;
		for (i=0;i<50;i++) {
			sm=0.0f;
			for (ip=0;ip<n-1;ip++) {
				for (iq=ip+1;iq<n;iq++)
					sm += Math.abs(a[ip][iq]);
			}
			if (sm == 0.0) {
				z=null;
				b=null;
				return;
			}
			if (i < 4)
				tresh= (0.2f*sm/(n*n));
			else
				tresh=0.0f;
			for (ip=0;ip<n-1;ip++) {
				for (iq=ip+1;iq<n;iq++) {
					g=100.0f*Math.abs(a[ip][iq]);
					if (i > 4 && fabs(d[ip])+g == fabs(d[ip])
							&& fabs(d[iq])+g == fabs(d[iq]))
						a[ip][iq]=0.0f;
					else if (fabs(a[ip][iq]) > tresh) {
						h=d[iq]-d[ip];
						if (fabs(h)+g == fabs(h))
							t=(a[ip][iq])/h;
						else {
							theta=0.5f*h/(a[ip][iq]);
							t=1.0f/(fabs(theta)+sqrt(1.0+theta*theta));
							if (theta < 0.0) t = -t;
						}
						c=1.0f/sqrt(1+t*t);
						s=t*c;
						tau=s/(1.0f+c);
						h=t*a[ip][iq];
						z[ip] -= h;
						z[iq] += h;
						d[ip] -= h;
						d[iq] += h;
						a[ip][iq]=0.0f;
						for (j=0;j<=ip-1;j++) {
							ROTATE(a,j,ip,j,iq,s,tau);
							g = a[j][ip];
							h = a[j][iq];
						}
						for (j=ip+1;j<=iq-1;j++) {
							ROTATE(a,ip,j,j,iq,s,tau);							
							g = a[ip][j];
							h = a[j][iq];
						}
						for (j=iq+1;j<n;j++) {
							ROTATE(a,ip,j,iq,j,s,tau);
							g = a[ip][j];
							h = a[iq][j];
						}
						for (j=0;j<n;j++) {
							ROTATE(v,j,ip,j,iq,s,tau);
							g = v[j][ip];
							h = v[j][iq];
						}
						++(nrot);
					}
				}
			}
			for (ip=0;ip<n;ip++) {
				b[ip] += z[ip];
				d[ip]=b[ip];
				z[ip]=0.0f;
			}
		}
		//nrerror("Too many iterations in routine JACOBI");
	}

	public void run() {
		while (!terminate){
			if(data){
				for (int i = 0; i < z; i++){
					float [][] v = new float[Volume3D.MAX_DIM][Volume3D.MAX_DIM];
					float [] eig = new float [Volume3D.MAX_DIM];
					int dim_loop;
					Integer nrot = new Integer(0);
					// TODO Auto-generated method stub
					jacobi(T[i], dimensions, eig, v, nrot);

					eigmax[i]=eig[0];
					for (dim_loop=1; dim_loop<dimensions; dim_loop++)
						if (eig[dim_loop]>eigmax[i]) eigmax[i]=eig[dim_loop];
					nrerror("Maximal Value:" + eigmax[i]);
				}
				finished = true;
				data = false;
			} else {
				try {
					Thread.sleep(0);
				} catch (InterruptedException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		}
	}

}
/*
 * Copyright (C) 2010-2014  Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/