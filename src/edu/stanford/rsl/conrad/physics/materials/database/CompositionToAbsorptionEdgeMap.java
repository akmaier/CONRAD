package edu.stanford.rsl.conrad.physics.materials.database;

import java.io.File;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Scanner;
import java.util.TreeMap;

import edu.stanford.rsl.conrad.physics.materials.utils.WeightedAtomicComposition;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.XmlUtils;


/**
 * Retrieves or calculate all the absorption edges  of a material based on its atomic composition.
 * @author Rotimi X Ojo *
 */
public class CompositionToAbsorptionEdgeMap {
	public static final File file = new File(MaterialsDB.getDatabaseLocation() + "configfiles/maps/compositionToAbsorptionEdge.xml");
	@SuppressWarnings("unchecked")
	private static TreeMap<String, String> map = (TreeMap<String, String>) XmlUtils.deserializeObject(file);

	/**
	 * Retrieve a set of unique absorption edges given an atomic composition
	 * @param comp is atomic composition of material
	 * @return an empty arraylist if there are no edges
	 */
	public static  ArrayList<Double> getAbsorptionEdges(WeightedAtomicComposition comp) {
		Iterator<String> it = comp.keysIterator();
		ArrayList<Double> edges = new ArrayList<Double>();
		while(it.hasNext()){
			String buf = map.get(it.next().trim());
			Scanner sc = new Scanner(buf);
			while(sc.hasNext()){
				double val = Double.parseDouble(sc.next());
				if(!edges.contains(val) && val > CONRAD.SMALL_VALUE){
					edges.add(val);
				}
			}
		}

		return edges;
	}

}
/*
 * Copyright (C) 2010-2014 Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/