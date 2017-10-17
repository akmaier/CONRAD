package edu.stanford.rsl.tutorial.mipda;

import org.junit.internal.TextListener;
import org.junit.runner.JUnitCore;

import ModuleCourseIntroduction.IntroTestClass;

public class RunExerciseEvaluation {

	private static final String exerciseName = "Intro";  
	
	public static void main(String[] args) {
		
		JUnitCore junitCore = new JUnitCore();
		junitCore.addListener(new TextListener(System.out));
		
		if (exerciseName.compareTo("Intro") == 0) {
			junitCore.run(IntroTestClass.class);
			//org.junit.runner.JUnitCore.main("ModuleCourseIntroduction.IntroTestClass");
		}
//		else if (exerciseName.compareTo("SVD?") == 0) {
//			
//		}
		else {
			System.out.println("No test available.");
		}
	}

}
