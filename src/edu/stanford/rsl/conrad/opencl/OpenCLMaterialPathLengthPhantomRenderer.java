/*
 * Copyright (C) 2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.conrad.opencl;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.HashMap;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLMemory.Mem;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.MultiChannelGrid2D;
import edu.stanford.rsl.conrad.phantom.AnalyticPhantom;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.physics.materials.Material;

public class OpenCLMaterialPathLengthPhantomRenderer extends OpenCLProjectionPhantomRenderer {

	ArrayList<Material> materials;
	String [] channelNames;

	protected HashMap<Material, CLBuffer<FloatBuffer>> mus;

	private CLBuffer<FloatBuffer> generateMuMap(CLContext context, CLDevice device, Material mat) {
		// absorption model
		// setting only the present material to 10 (due to the normalization in the drawing code.
		// this means we draw the path length in [mm].
		CLBuffer<FloatBuffer> mu = context.createFloatBuffer(phantom.size() +1, Mem.READ_ONLY); 
		mu.getBuffer().put((float) phantom.getBackgroundMaterial().getDensity());
		for (PhysicalObject o: phantom){
			float density = 0.0f;
			if (o.getMaterial().equals(mat)) density = 10.0f;
			mu.getBuffer().put(density);
		}
		mu.getBuffer().rewind();
		device.createCommandQueue().putWriteBuffer(mu, false).finish().release();
		return mu;
	}

	@Override
	protected void evaluateAbsorptionModel(int k) {

		MultiChannelGrid2D multiChannelGrid2D = new MultiChannelGrid2D(dimx, dimy, materials.size());
		multiChannelGrid2D.setChannelNames(channelNames);

		for (int c = 0; c < materials.size(); c++){
			// evaluate absorption model
			long time = System.nanoTime();
			renderer.drawScreenMonochromatic(screenBuffer, mus.get(materials.get(c)), priorities);
			//render.drawScreen(screenBuffer);
			time = System.nanoTime() - time;
			System.out.println("path length screen buffer drawing took: "+(time/1000000)+"ms for " +materials.get(c));


			CLCommandQueue clc = renderer.device.createCommandQueue();
			clc.putReadBuffer(screenBuffer, true).finish();
			clc.release();

			float[] array = new float [dimx*dimy]; 
			for (int j = 0; j < dimy; j++){
				for (int i = 0; i < dimx; i++){
					array[(j*dimx)+i] = screenBuffer.getBuffer().get();
				}
			}
			screenBuffer.getBuffer().rewind();

			// Save data to buffer
			Grid2D image = new Grid2D(array, dimx, dimy);
			multiChannelGrid2D.setChannel(c, image);

			screenBuffer.getBuffer().clear(); 	// its not the screenbuffer, copied TestOpenCL code
		}

		buffer.add(multiChannelGrid2D, k); // its not the buffer, deleted it
	}

	@Override
	public void release(){
		super.release();
		for (int i=mus.size()-1; i > -1; i--){
			CLBuffer<FloatBuffer> buffer = mus.get(i);
			if (buffer != null){
				mus.remove(i);
				buffer.release();
			}
		}
	}

	@Override
	public void configure() throws Exception{
		AnalyticPhantom phantom = AnalyticPhantom.getCurrentPhantom();
		CLContext context = OpenCLUtil.createContext();
		CLDevice device = context.getMaxFlopsDevice();

		configure(phantom, context, device, false);

		materials = new ArrayList<Material>();
		for (PhysicalObject o: phantom){
			if (! materials.contains(o.getMaterial())) materials.add(o.getMaterial());
		}

		mus = new HashMap<Material, CLBuffer<FloatBuffer>>();
		for (int i = 0; i< materials.size(); i++){
			mus.put(materials.get(i), generateMuMap(context, device, materials.get(i)));
		}

		channelNames = new String[materials.size()];
		for (int i=0; i < materials.size(); i ++){
			channelNames[i]=materials.get(i).getName(); 
		}

		configured = true;
	}

	@Override
	public String getBibtexCitation() {
		String bibtex = "@Article{Maier12-FSO,\n" +
		"  author = {Maier, A. and Hofmann, H.G. and Schwemmer, C. and Hornegger, J. and Keil, A. and Fahrig, R.},\n" +
		"  title = {Fast simulation of x-ray projections of spline-based surfaces using an append buffer},\n" +
		"  journal = {Physics in Medicine and Biology},\n" +
		"  volume = {57},\n" +
		"  number = {19},\n" +
		"  pages = {6193-6210},\n" +
		"  year = {2012}\n" +
		"}";
		return bibtex;
	}

	@Override
	public String getMedlineCitation() {
		String medline = "Maier A, Hofmann HG, Schwemmer C, Hornegger J, Keil A, Fahrig R. Fast simulation of x-ray projections of spline-based surfaces using an append buffer. Phys Med Biol. 57(19):6193-210. 2012";
		return medline;
	}

	@Override
	public String toString() {
		return "OpenCL Material Path Length Renderer";
	}

}
