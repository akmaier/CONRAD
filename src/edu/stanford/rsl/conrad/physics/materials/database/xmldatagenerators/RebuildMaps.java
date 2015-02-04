package edu.stanford.rsl.conrad.physics.materials.database.xmldatagenerators;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
import java.util.TreeMap;

import edu.stanford.rsl.conrad.physics.materials.database.CompositionToAbsorptionEdgeMap;
import edu.stanford.rsl.conrad.physics.materials.database.FormulaToNameMap;
import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;
import edu.stanford.rsl.conrad.physics.materials.database.NameToFormulaMap;
import edu.stanford.rsl.conrad.utils.XmlUtils;

/**
 * 
 * @author Rotimi X Ojo
 *
 */
public class RebuildMaps {
	
	/**
	 * Entry point of map database rebuilding program.
	 */
	public static void main(String[] args) {
		rebuildFormulaToNameMap();
		rebuildNameToFormulaMap();
		rebuildCompositionToAbsorptionEdgeMap();		
	}
	
	/**
	 * Routine to rebuild persistent atomic composition to absorption edge map.
	 * @see CompositionToAbsorptionEdgeMap
	 */
	private static void rebuildCompositionToAbsorptionEdgeMap() {
		TreeMap<String, String> table = new TreeMap<String, String>();
		try {
			Scanner sc = new Scanner( new File(MaterialsDB.getDatabaseLocation() +"configfiles/materials/AbsorptionEdgeEnergies.txt"));
			while(sc.hasNextLine()){
				String line = sc.nextLine();
				String key = line.substring(0, line.indexOf(":")).trim();
				String data = line.substring(line.indexOf(":")+1).trim();	
				table.put(key, data);
			}				
			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}		
		XmlUtils.serializeObject(CompositionToAbsorptionEdgeMap.file, table);
	}
	
	/**
	 * Routine to rebuild material name to material formula map.
	 * @see NameToFormulaMap
	 */
	private static void rebuildNameToFormulaMap() {
		TreeMap<String, String> table = new TreeMap<String, String>();
		table.put("scandium","Sc");
		table.put("antimony","Sb");
		table.put("ytterbium","Yb");
		table.put("francium","Fr");
		table.put("CZT","Cd9ZnTe10");
		table.put("lutetium","Lu");
		table.put("ruthenium","Ru");
		table.put("Plastic","C5H8");
		table.put("iron","Fe");
		table.put("lithium","Li");
		table.put("radon","Rn");
		table.put("rhodium","Rh");
		table.put("lanthanum","La");
		table.put("rhenium","Re");
		table.put("BGO","B4Ge3O12");
		table.put("rubidium","Rb");
		table.put("radium","Ra");
		table.put("xenon","Xe");
		table.put("europium","Eu");
		table.put("erbium","Er");
		table.put("Water","H2O");
		table.put("krypton","Kr");
		table.put("PTFE","CF2");
		table.put("Polyethylene","H2C");
		table.put("LuYAP-70","Lu8Y2Al10O30");
		table.put("Dysprosium","Dy");
		table.put("LSO","Lu2SO5");
		table.put("Scinti-C9H10","C9H10");
		table.put("platinum","Pt");
		table.put("praseodymium","Pr");
		table.put("NaI ","NaI");
		table.put("polonium","Po");
		table.put("PVC","H3C2Cl");
		table.put("YAP","YAlO3");
		table.put("promethium","Pm");
		table.put("palladium","Pd");
		table.put("lead","Pb");
		table.put("copper","Cu");
		table.put("cesium","Cs");
		table.put("chromium","Cr");
		table.put("yttrium","Y");
		table.put("tungsten","W");
		table.put("cobalt","Co");
		table.put("vanadium","V");
		table.put("uranium","U");
		table.put("iridium","Ir");
		table.put("chlorine","Cl");
		table.put("sulfur","S");
		table.put("indium","In");
		table.put("osmium","Os");
		table.put("phosphorus","P");
		table.put("oxygen","O");
		table.put("nitrogen","N");
		table.put("cerium","Ce");
		table.put("cadmium","Cd");
		table.put("potassium","K");
		table.put("iodine","I");
		table.put("calcium","Ca");
		table.put("hydrogen","H");
		table.put("fluorine","F");
		table.put("carbon","C");
		table.put("boron","B");
		table.put("PWO","PbWO4");
		table.put("bromine","Br");
		table.put("holmium","Ho");
		table.put("bismuth","Bi");
		table.put("beryllium","Be");
		table.put("mercury","Hg");
		table.put("barium","Ba");
		table.put("helium","He");
		table.put("Quartz","SiO2");
		table.put("nickel","Ni");
		table.put("zirconium","Zr");
		table.put("thulium","Tm");
		table.put("thallium","Tl");
		table.put("neon","Ne");
		table.put("zinc","Zn");
		table.put("titanium","Ti");
		table.put("neodymium","Nd");
		table.put("thorium","Th");
		table.put("niobium","Nb");
		table.put("sodium","Na");
		table.put("tellurium","Te");
		table.put("gold","Au");
		table.put("astatine","At");
		table.put("technetium","Tc");
		table.put("arsenic","As");
		table.put("terbium","Tb");
		table.put("argon","Ar");
		table.put("tantalum","Ta");
		table.put("aluminium","Al");
		table.put("GSO","Gd2SO5");
		table.put("silver","Ag");
		table.put("molybdenum","Mo");
		table.put("manganese","Mn");
		table.put("actinium","Ac");
		table.put("germanium","Ge");
		table.put("tin","Sn");
		table.put("gadolinium","Gd");
		table.put("samarium","Sm");
		table.put("magnesium","Mg");
		table.put("gallium","Ga");
		table.put("silicon","Si");
		table.put("LuAP","LuAlO3");
		table.put("selenium","Se");
		
		XmlUtils.serializeObject(NameToFormulaMap.file, table);
		
	}
	
	/**
	 * Routine to rebuild formula to name map
	 * @see FormulaToNameMap
	 */
	private static void rebuildFormulaToNameMap() {
		TreeMap<String, String> table = new TreeMap<String, String>();
		table.put("Ac","actinium");
		table.put("Al","aluminium");
		table.put("Sb","antimony");
		table.put("Ar","argon");
		table.put("As","arsenic");
		table.put("Ba","barium");
		table.put("Ag","silver");
		table.put("Be","beryllium");
		table.put("Bi","bismuth");
		table.put("B","boron");
		table.put("Br","bromine");
		table.put("Cd","cadmium");
		table.put("Ca","calcium");
		table.put("C","carbon");
		table.put("Ce","cerium");
		table.put("Cs","cesium");
		table.put("Cl","chlorine");
		table.put("Cr","chromium");
		table.put("Co","cobalt");
		table.put("Cu","copper");
		table.put("Dy","Dysprosium");
		table.put("Er","erbium");
		table.put("Eu","europium");
		table.put("F","fluorine");
		table.put("Fr","francium");
		table.put("Gd","gadolinium");
		table.put("Ga","gallium");
		table.put("Ge","germanium");
		table.put("Au","gold");
		table.put("He","helium");
		table.put("Ho","holmium");
		table.put("H","hydrogen");
		table.put("In","indium");
		table.put("I","iodine");
		table.put("Ir","iridium");
		table.put("Fe","iron");
		table.put("Kr","krypton");
		table.put("La","lanthanum");
		table.put("Pb","lead");
		table.put("Li","lithium");
		table.put("Lu","lutetium");
		table.put("Mg","magnesium");
		table.put("Mn","manganese");
		table.put("Hg","mercury");
		table.put("Mo","molybdenum");
		table.put("Nd","neodymium");
		table.put("At","astatine");
		table.put("Ne","neon");
		table.put("Ni","nickel");
		table.put("Nb","niobium");
		table.put("N","nitrogen");
		table.put("Os","osmium");
		table.put("O","oxygen");
		table.put("Pd","palladium");
		table.put("P","phosphorus");
		table.put("Pt","platinum");
		table.put("Po","polonium");
		table.put("K","potassium");
		table.put("Pr","praseodymium");
		table.put("Pm","promethium");
		table.put("Ra","radium");
		table.put("Rn","radon");
		table.put("Re","rhenium");
		table.put("Rh","rhodium");
		table.put("Rb","rubidium");
		table.put("Ru","ruthenium");
		table.put("Sm","samarium");
		table.put("Sc","scandium");
		table.put("Se","selenium");
		table.put("Si","silicon");
		table.put("Na","sodium");
		table.put("S","sulfur");
		table.put("Ta","tantalum");
		table.put("Tc","technetium");
		table.put("Te","tellurium");
		table.put("Tb","terbium");
		table.put("Tl","thallium");
		table.put("Th","thorium");
		table.put("Tm","thulium");
		table.put("Sn","tin");
		table.put("Ti","titanium");
		table.put("W","tungsten");
		table.put("U","uranium");
		table.put("V","vanadium");
		table.put("Xe","xenon");
		table.put("Yb","ytterbium");
		table.put("Y","yttrium");
		table.put("Zn","zinc");
		table.put("Zr","zirconium");
		//
		table.put("NaI", "NaI ");
		table.put("PbWO4", "PWO");
		table.put("B4Ge3O12", "BGO");
		table.put("Lu2SO5", "LSO");
		table.put("H2O", "Water");
		table.put("Gd2SO5", "GSO");
		table.put("LuAlO3", "LuAP");
		table.put("YAlO3", "YAP");
		table.put("SiO2", "Quartz");
		table.put("C9H10", "Scinti-C9H10");
		table.put("Lu8Y2Al10O30", "LuYAP-70");
		table.put("C5H8", "Plastic");
		table.put("Cd9ZnTe10", "CZT");
		table.put("H2C", "Polyethylene");
		table.put("H3C2Cl", "PVC");
		table.put("CF2", "PTFE");	
		
		XmlUtils.serializeObject(FormulaToNameMap.file, table);
		
	}

	

}
/*
 * Copyright (C) 2010-2014 Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/