package mi_lecke_2;

import java.util.ArrayList;
import java.util.Scanner;

public class NNSolutionOne {
	
	public static void main(String[] args){
		Scanner reader = new Scanner(System.in);
		nnSolutionOne(reader);
		
		reader.close();
	}
	
	public static void nnSolutionOne(Scanner reader){
		int[] inputArchitecture=MI2_IO.readArchitecture(reader);
		
		NeuralNetwork myNetwork=new NeuralNetwork(inputArchitecture);
		myNetwork.initializeWeightsRandom();
		
		int[] outputArchitecture=myNetwork.getArchitecture();
		
		ArrayList<double[]> weightAndBiasValues=myNetwork.getWeightAndBiasValues();
		
		MI2_IO.writeArchitecture(outputArchitecture);
		MI2_IO.writeWeightAndBiasValues(weightAndBiasValues);
	}
}
