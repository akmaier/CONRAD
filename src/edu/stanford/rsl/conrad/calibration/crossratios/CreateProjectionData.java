package edu.stanford.rsl.conrad.calibration.crossratios;

import ij.IJ;
import ij.ImagePlus;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;

import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Sphere;
import edu.stanford.rsl.conrad.opencl.OpenCLProjectionPhantomRenderer;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;
import edu.stanford.rsl.conrad.phantom.AnalyticPhantom;
import edu.stanford.rsl.conrad.phantom.renderer.PhantomRenderer;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.ImageUtil;

public class CreateProjectionData {

	public static void main(String[] args) {
		// read phantom definition
		
		AnalyticPhantom objects = new AnalyticPhantom() {
			
			/**
			 * 
			 */
			private static final long serialVersionUID = 3622828209730135558L;

			@Override
			public String getMedlineCitation() {
				return CONRAD.CONRADMedline;
			}
			
			@Override
			public String getBibtexCitation() {
				// TODO Auto-generated method stub
				return CONRAD.CONRADBibtex;
			}
			
			@Override
			public String getName() {
				return "Cross Ratio Phantom";
			}
		};
		
		try {
			@SuppressWarnings("resource")
			BufferedReader reader = new BufferedReader(new FileReader(args[0]));
			String line = reader.readLine();
			while (line != null){
				String [] numbers = line.split(";");
				double x = Double.parseDouble(numbers[0]);
				double y = Double.parseDouble(numbers[1]);
				double z = Double.parseDouble(numbers[2]);
				double radius = Double.parseDouble(numbers[3]);
				Sphere sp = new Sphere(radius, new PointND(x,y,z));
				PhysicalObject po = new PhysicalObject();
				po.setNameString("Bead " + objects.size());
				po.setShape(sp);
				po.setMaterial(MaterialsDB.getMaterial("tungsten"));
				objects.add(po);
				line = reader.readLine();
			};
			
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		// create projection data
		Configuration.loadConfiguration();
		long time = System.currentTimeMillis();
		OpenCLProjectionPhantomRenderer render = new OpenCLProjectionPhantomRenderer();
		ImagePlus image = null;
		try {
			CLContext context = OpenCLUtil.createContext();
			CLDevice device = context.getMaxFlopsDevice();

			render.configure(objects, context, device, true);

			Grid3D result = PhantomRenderer.generateProjections(render);
			image = ImageUtil.wrapGrid3D(result, objects.toString());
			time = System.currentTimeMillis() - time;
			System.out.println("Runtime: " + time/1000.0 + " s");
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		// write to disk

		if (image != null){
			IJ.save(image, args[1]);
		}
		
	}

}
