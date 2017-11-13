package mi_lecke_2;

import java.util.ArrayList;
import java.util.Scanner;

public class MI2_IO {
	public static void writeMatrix(double[][] matrixData){
		for (int i = 0; i < matrixData.length; i++) {
			System.out.print(matrixData[i][0]);
			for (int j = 1; j < matrixData[i].length; j++) {
				System.out.print("," + matrixData[i][j]);
			}
	    	System.out.println();
		}
	}	
	public static void writeDoubles(double[] data){
		System.out.print(data[0]);
		
		for(int i=1;i<data.length;i++){
			System.out.print("," + data[i]);
		}
		System.out.println("");
	}
	public static void writeArchitecture(int[] architecture){
		
		if(architecture.length > 0)System.out.print(architecture[0]);
		for(int i=1; i<architecture.length; i++){
			System.out.print("," + architecture[i]);
		}
		System.out.println("");
	}
	
	
	public static void writeWeightAndBiasValues(ArrayList<double[]> weightsAndBias){
		for(double[] neuronWeightArray : weightsAndBias){	
			System.out.print(neuronWeightArray[0]);
			for(int i=1; i<neuronWeightArray.length;i++){
				System.out.print(","+neuronWeightArray[i]);
			}
			System.out.println("");
		}
	}
	public static int[] readArchitecture(Scanner reader){
		String line;
		
		line = reader.nextLine();

		String[] stringNumbers = line.split(",");
		int[] architecture = new int[stringNumbers.length];
		
		for (int i = 0; i < stringNumbers.length; i++) {
			architecture[i] = Integer.parseInt(stringNumbers[i]);
		}
		
		return architecture;
	}
	
	public static double[] readLineOfDoubles(Scanner reader){
		String line;
		
		
		line = reader.nextLine();
		
		
		String[] stringNumbers = line.split(",");
		double[] numbers = new double[stringNumbers.length];
		
		for (int i = 0; i < stringNumbers.length; i++) {
			numbers[i] = Double.parseDouble(stringNumbers[i]);
		}
		
		return numbers;
	}
	
	public static double[][] readInputs(Scanner reader,int inputVectorDimension){
		String line;
		
		line=reader.nextLine();
		int numberOfInputs=Integer.parseInt(line);
		
		double[][] inputs=new double[numberOfInputs][inputVectorDimension];
		
		for(int i=0; i<numberOfInputs; i++){
			double[] input=readLineOfDoubles(reader);
			
			for(int j=0; j < inputVectorDimension; j++){
				inputs[i][j]= input[j];			
			}
		}
		
		return inputs;
	}
	
	public static LearningParameters readLearningParameters(Scanner reader){
		String line;
		line=reader.nextLine();
		
		String[] stringNumbers=line.split(",");
		LearningParameters learningParameters=new LearningParameters();
		
		learningParameters.epochs=Integer.parseInt(stringNumbers[0]);
		learningParameters.u=Double.parseDouble(stringNumbers[1]);
		learningParameters.R=Double.parseDouble(stringNumbers[2]);
		
		return learningParameters;
	}
}
