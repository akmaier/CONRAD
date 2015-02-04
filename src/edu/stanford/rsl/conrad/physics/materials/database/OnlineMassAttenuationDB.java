package edu.stanford.rsl.conrad.physics.materials.database;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.UnsupportedEncodingException;
import java.net.URL;
import java.net.URLConnection;
import java.net.URLEncoder;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Scanner;
import java.util.Set;
import java.util.TreeMap;

import edu.stanford.rsl.conrad.numerics.mathexpressions.Evaluator;
import edu.stanford.rsl.conrad.physics.materials.utils.AttenuationType;
import edu.stanford.rsl.conrad.physics.materials.utils.WeightedAtomicComposition;
import edu.stanford.rsl.conrad.utils.interpolation.NumberInterpolatingTreeMap;


/**
 * <p>This class provides access to NIST XCOM Database. 
 * <br/>This can be used to retrieve the energy dependent mass attenuation of elements, and arbitrary combination of elements.
 * <br/>The NIST database supports energies between 1 keV - 100 GeV;</p>
 * @author Rotimi X Ojo
 */
public class OnlineMassAttenuationDB {

	private static String nistDBUrl = "http://physics.nist.gov";

	public static double getMassAttenuationData(WeightedAtomicComposition comp,
			double energy, AttenuationType attType) {
		ArrayList<AttenuationType> ty = new ArrayList<AttenuationType>();
		ty.add(attType);
		return getMassAttenuationData(comp, energy + ";", ty, false).get(attType).firstEntry().getValue().doubleValue();
	}

	/**
	 * Retrieves energy dependent mass attenuation data from NIST XCOM database
	 * @param comp is atomic composition by weight of material to be retrieved
	 * @param energies is semi-colon seperated list  of energies(MEV) of interest.
	 * @return energy dependent mass attenuation of supplied energies including K-Edges
	 */
	public static TreeMap<AttenuationType, NumberInterpolatingTreeMap> getMassAttenuationData(
			WeightedAtomicComposition comp, String energies, ArrayList<AttenuationType> att, boolean useDefaultEnergies) {		
		return getMassAttenuationDataFromNIST(getCompositionAsString(comp),energies, att,useDefaultEnergies);	
	}

	private static String buildURLLiteral(String composition, String energies, ArrayList<AttenuationType> att, boolean useDefaultEnergies) {
		String data = "";
		composition= composition.trim();

		try {
			data  = URLEncoder.encode("WindowXmin", "UTF-8") + "=" + URLEncoder.encode(0.001 + "", "UTF-8");
			data += "&" + URLEncoder.encode("WindowXmax", "UTF-8") + "=" + URLEncoder.encode(100000 + "", "UTF-8"); 
			data += "&" + URLEncoder.encode("Formulae", "UTF-8") + "=" + URLEncoder.encode(composition, "UTF-8");	
			data += "&" + URLEncoder.encode("Method", "UTF-8") + "=" + URLEncoder.encode("3", "UTF-8");
			data += "&" + URLEncoder.encode("NumAdd", "UTF-8") + "=" + URLEncoder.encode("1", "UTF-8");
			data += "&" + URLEncoder.encode("Energies", "UTF-8") + "=" + URLEncoder.encode(energies, "UTF-8");
			if(useDefaultEnergies){
				data += "&" + URLEncoder.encode("Output", "UTF-8") + "=" + URLEncoder.encode("on", "UTF-8");	
			}else{
				data += "&" + URLEncoder.encode("Output", "UTF-8") + "=" + URLEncoder.encode("", "UTF-8");	
			}

			for(AttenuationType type:att){
				data += "&" + URLEncoder.encode(type.getName(), "UTF-8") + "=" + URLEncoder.encode("on", "UTF-8");
			}

		} catch (UnsupportedEncodingException e1) {
			e1.printStackTrace();
		} 	
		return data;
	}

	/**
	 * Retrieves Mass attenuation data from the NIST XCOM Database
	 * @param composition is space separated list of atomic composition of material of interest
	 * @param energies is semi-colon separated list of energies
	 * @param att is array list containing the types of attenuation the developer is interested in retrieving
	 * @return Mass attenuation data from the NIST XCOM Database
	 */
	private static TreeMap<AttenuationType, NumberInterpolatingTreeMap>getMassAttenuationDataFromNIST(String composition, String energies, ArrayList<AttenuationType> att, boolean useDefaultEnergies){
		String urlSpec = buildURLLiteral(composition, energies, att, useDefaultEnergies);
		TreeMap<AttenuationType, NumberInterpolatingTreeMap> table = new TreeMap<AttenuationType, NumberInterpolatingTreeMap>();
		table.put(AttenuationType.COHERENT_ATTENUATION, new NumberInterpolatingTreeMap());
		table.put(AttenuationType.INCOHERENT_ATTENUATION, new NumberInterpolatingTreeMap());
		table.put(AttenuationType.PHOTOELECTRIC_ABSORPTION, new NumberInterpolatingTreeMap());
		table.put(AttenuationType.ELECTRON_FIELD_PAIRPRODUCTION, new NumberInterpolatingTreeMap());
		table.put(AttenuationType.NUCLEAR_FIELD_PAIRPRODUCTION, new NumberInterpolatingTreeMap());
		table.put(AttenuationType.TOTAL_WITH_COHERENT_ATTENUATION, new NumberInterpolatingTreeMap());
		table.put(AttenuationType.TOTAL_WITHOUT_COHERENT_ATTENUATION, new NumberInterpolatingTreeMap());

		try {
			URL url = new URL(nistDBUrl + "/cgi-bin/Xcom/data.pl"); 
			URLConnection conn = url.openConnection(); 
			conn.setDoOutput(true); 
			OutputStreamWriter wr = new OutputStreamWriter(conn.getOutputStream()); 
			wr.write(urlSpec); 
			wr.flush(); 
			BufferedReader rd = new BufferedReader(new InputStreamReader(conn.getInputStream())); 
			String line;

			while(true){
				line = rd.readLine();
				if (line != null) {
					if (line.length()> 0) {
						String substr = line.substring(0,1);
						if (substr != null){
							if(line.length()> 0 && line.substring(0,1).matches("\\p{Digit}+")){
								break;
							}

						}
					}
				} else {
					System.out.println("No attenuation coefficients found for "+ composition);
					return null;
				}
			}			
			do{
				addEntryToMap(table,line, att);
			}while ((line = rd.readLine()) != null); 

		} catch (Exception e) {
			e.printStackTrace();
			return null;
		} 
		return table;
	}



	private static String getCompositionAsString(
			WeightedAtomicComposition comp) {
		TreeMap<String, Double> table = comp.getCompositionTable();
		String composition = "";
		Set<String> keys = table.keySet();
		Iterator<String> it =keys.iterator();	

		NumberFormat f = NumberFormat.getInstance();  
		f.setGroupingUsed(false);  
		while(it.hasNext()){
			String text = it.next();
			text += ("  " + f.format(table.get(text))+ " ");
			composition+=text;
		}	
		return composition;
	}



	private static void addEntryToMap(
			TreeMap<AttenuationType, NumberInterpolatingTreeMap> table, String line, ArrayList<AttenuationType> att) {
		Scanner s = new Scanner(line);
		double energy = Evaluator.getValue(s.next().trim());
		NumberInterpolatingTreeMap mp;
		if(att.contains(AttenuationType.COHERENT_ATTENUATION)){
			mp = table.get(AttenuationType.COHERENT_ATTENUATION);
			mp.put(energy, Evaluator.getValue(s.next().trim()));
		}
		if(att.contains(AttenuationType.INCOHERENT_ATTENUATION)){
			mp = table.get(AttenuationType.INCOHERENT_ATTENUATION);
			mp.put(energy, Evaluator.getValue(s.next().trim()));
		}
		if(att.contains(AttenuationType.PHOTOELECTRIC_ABSORPTION)){
			mp = table.get(AttenuationType.PHOTOELECTRIC_ABSORPTION);
			mp.put(energy, Evaluator.getValue(s.next().trim()));
		}
		if(att.contains(AttenuationType.NUCLEAR_FIELD_PAIRPRODUCTION)){
			mp = table.get(AttenuationType.NUCLEAR_FIELD_PAIRPRODUCTION);
			mp.put(energy, Evaluator.getValue(s.next().trim()));
		}
		if(att.contains(AttenuationType.ELECTRON_FIELD_PAIRPRODUCTION)){
			mp = table.get(AttenuationType.ELECTRON_FIELD_PAIRPRODUCTION);
			mp.put(energy, Evaluator.getValue(s.next().trim()));
		}
		if(att.contains(AttenuationType.TOTAL_WITH_COHERENT_ATTENUATION)){
			mp = table.get(AttenuationType.TOTAL_WITH_COHERENT_ATTENUATION);
			mp.put(energy, Evaluator.getValue(s.next().trim()));
		}
		if(att.contains(AttenuationType.TOTAL_WITHOUT_COHERENT_ATTENUATION)){
			mp = table.get(AttenuationType.TOTAL_WITHOUT_COHERENT_ATTENUATION);
			mp.put(energy, Evaluator.getValue(s.next().trim()));
		}

	}






}
/*
 * Copyright (C) 2010-2014 Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/