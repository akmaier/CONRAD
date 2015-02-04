package edu.stanford.rsl.conrad.physics.materials.database.xmldatagenerators;

import java.io.File;
import java.util.ArrayList;
import java.util.Scanner;
import java.util.TreeMap;

import edu.stanford.rsl.conrad.physics.materials.Compound;
import edu.stanford.rsl.conrad.physics.materials.Element;
import edu.stanford.rsl.conrad.physics.materials.Mixture;
import edu.stanford.rsl.conrad.physics.materials.database.CompositionToAbsorptionEdgeMap;
import edu.stanford.rsl.conrad.physics.materials.database.ElementalMassAttenuationData;
import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;
import edu.stanford.rsl.conrad.physics.materials.database.OnlineMassAttenuationDB;
import edu.stanford.rsl.conrad.physics.materials.utils.AttenuationType;
import edu.stanford.rsl.conrad.physics.materials.utils.WeightedAtomicComposition;
import edu.stanford.rsl.conrad.utils.interpolation.NumberInterpolatingTreeMap;


/**
 * Class to build a xml database of defined materials (Elements, Compounds, Mixtures) and Elemental Mass Attenuation Database.
 * To be used incase of data corruption
 * @author Rotimi X Ojo
 *
 */
public class RebuildMaterialDatabase {
	
	private static File elements;
	private static File compounds;
	private static File mixtures;	
	private static double DB_MIN_ENERGY = 0.001;

	
	public static void main(String[] args) throws Exception{
		elements = new File(MaterialsDB.getDatabaseLocation() + "configfiles/materials/elements.txt");
		compounds = new File(MaterialsDB.getDatabaseLocation() + "configfiles/materials/compounds.txt");
		mixtures = new File(MaterialsDB.getDatabaseLocation() + "configfiles/materials/mixtures.txt");		
		rebuildMaterialFromConfigFiles();
	}

	public static void rebuildMaterialFromConfigFiles() throws Exception{
		buildMassAttenuationDB();
		buildElementDB();
		buildCompoundDB();
		buildMixtureDB();
	}	

	private static void buildMassAttenuationDB() throws Exception {
		Scanner scanner = new Scanner(elements);
		while(scanner.hasNext()){
			RetrieveElementalDataFromNist(scanner.nextLine());
		}		
	}

	public static void buildElementDB() throws Exception{
		Scanner scanner = new Scanner(elements);
		while(scanner.hasNext()){
			processElementDefinition(scanner.nextLine());
		}
	}
	
	public static void buildCompoundDB() throws Exception{
		Scanner scanner = new Scanner(compounds);
		while(scanner.hasNext()){
			processCompoundDefinition(scanner.nextLine());
		}
	}
	
	public static void buildMixtureDB() throws Exception{
		Scanner scanner = new Scanner(mixtures);
		while(scanner.hasNext()){
			processMixtureDefinition(scanner.nextLine());
		}
	}	
	
	private static void RetrieveElementalDataFromNist(String line) {
		String sym = line.substring(line.indexOf('=')+ 1, line.indexOf(';')).trim();
		Scanner s = new Scanner(line.substring(line.indexOf(':')+1).trim());
		s.useDelimiter(";");		
		WeightedAtomicComposition comp = new WeightedAtomicComposition();
		comp.addUniqueElement(sym, 1);		

		String energies = getArbsoptionEdges(comp);
		ArrayList<AttenuationType> att = new ArrayList<AttenuationType>();
		att.add(AttenuationType.COHERENT_ATTENUATION);
		att.add(AttenuationType.INCOHERENT_ATTENUATION);
		att.add(AttenuationType.ELECTRON_FIELD_PAIRPRODUCTION);
		att.add(AttenuationType.NUCLEAR_FIELD_PAIRPRODUCTION);
		att.add(AttenuationType.PHOTOELECTRIC_ABSORPTION);
		att.add(AttenuationType.TOTAL_WITH_COHERENT_ATTENUATION);
		att.add(AttenuationType.TOTAL_WITHOUT_COHERENT_ATTENUATION);
		String name = line.substring(0,line.indexOf(':')).trim().toLowerCase();
		TreeMap<AttenuationType, NumberInterpolatingTreeMap> map = OnlineMassAttenuationDB.getMassAttenuationData(comp ,energies,att, true);
		if (map != null) ElementalMassAttenuationData.put(name, map);	
	}


	private static void processElementDefinition(String line) {
		String sym = line.substring(line.indexOf('=')+ 1, line.indexOf(';')).trim();
		Scanner s = new Scanner(line.substring(line.indexOf(':')+1).trim());
		s.useDelimiter(";");
		Element element = new Element();
		element.setName(line.substring(0,line.indexOf(':')).trim().toLowerCase());
		String token = s.next();
		element.setSymbol(token.substring(token.indexOf('=')+1).trim());
		token = s.next();
		element.setAtomicNumber(Double.valueOf(token.substring(token.indexOf('=')+1).trim()));
		token = s.next();
		element.setAtomicWeight(Double.valueOf(token.substring(token.indexOf('=')+1).trim()));
		token = s.next();
		element.setDensity(Double.valueOf(token.substring(token.indexOf('=')+1).trim()));
		WeightedAtomicComposition comp = new WeightedAtomicComposition();
		comp.addUniqueElement(sym, 1);
		element.setWeightedAtomicComposition(comp);	
		System.out.println(element.getName());
		MaterialsDB.put(element);	
	}
	


	private static  void processCompoundDefinition(String line) {
		if(line.equals("")){
			return;
		}
		String name = line.substring(0,line.indexOf(':')).trim().toLowerCase();
		System.out.println(name);
		double density = Double.valueOf(line.substring(line.indexOf("=")+1,line.indexOf(";")));
		String formula = line.substring(line.indexOf("F=")+2).trim();		
		Compound props = new Compound();
		props.setDensity(density);
		props.setFormula(formula);
		props.setName(name);
		WeightedAtomicComposition comp = new WeightedAtomicComposition(formula);
		props.setWeightedAtomicComposition(comp);
		MaterialsDB.put(props);
	}
	
	private static void processMixtureDefinition(String line) {
		if(line.trim().length() == 0){
			return;
		}
		String name = line.substring(0,line.indexOf(':')).trim().toLowerCase();
		System.out.println(name);
		double density = Double.valueOf(line.substring(line.indexOf("=")+1,line.indexOf(";")));
		Mixture mix = new Mixture();
		mix.setName(name);
		mix.setDensity(density);		
		mix.setWeightedAtomicComposition(getMixturesAtomicComposition(line.substring(line.indexOf(';'))));
		MaterialsDB.put(mix);
	}

	private static WeightedAtomicComposition getMixturesAtomicComposition(
			String line) {
		WeightedAtomicComposition comp = new WeightedAtomicComposition();
		Scanner sc = new Scanner(line);
		sc.useDelimiter(";");
		while(sc.hasNext()){
			String next = sc.next().trim();
			if(next.length() < 1){
				continue;
			}
			comp.add(next.substring(0,next.indexOf("[")).trim(), Double.valueOf(next.substring(next.indexOf("[")+1,next.indexOf("]"))));
		}
		return comp;
	}
	
	private static String getArbsoptionEdges(WeightedAtomicComposition comp) {	
		String vals = "";
		ArrayList<Double> energies = CompositionToAbsorptionEdgeMap.getAbsorptionEdges(comp);
		if (energies.size() == 0) {
			return vals;
		}
		

		for (double val : energies) {
			if(val < DB_MIN_ENERGY ){
				continue;
			}
			vals += val + ";";
		}
		return vals;
	}
	



}
/*
 * Copyright (C) 2010-2014 Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/