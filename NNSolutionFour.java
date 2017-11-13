package mi_lecke_2;

import java.util.Scanner;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;

public class NNSolutionFour {
	public static void main(String[] args){
		Scanner reader = new Scanner(System.in);
		nnSolutionFour(reader);
		reader.close();
	}

	public static void nnSolutionFour(Scanner reader){
		LearningParameters params=MI2_IO.readLearningParameters(reader);
		
		int[] inputArchitecture=MI2_IO.readArchitecture(reader);
		
		NeuralNetwork myNetwork=new NeuralNetwork(inputArchitecture);	
		myNetwork.initializeWeightsFromConsole(reader);
				
		double[][] inputs=MI2_IO.readInputs(reader,inputArchitecture[0]
				+inputArchitecture[inputArchitecture.length-1]);
		
		double samplesCount=(double)inputs.length;
		int learningSamplesCount=(int)(samplesCount*params.R);
		int validationSamplesCount=inputs.length-learningSamplesCount;
		int xSize=inputArchitecture[0];
		int dSize=inputArchitecture[inputArchitecture.length-1];
		
		double[][] learningX=new double[learningSamplesCount][xSize];
		double[][] learningD=new double[learningSamplesCount][dSize];
		double[][] validationX=new double[validationSamplesCount][xSize];
		double[][] validationD=new double[validationSamplesCount][dSize];
		
		for(int i = 0; i<learningSamplesCount;i++){
			for(int j=0; j<xSize; j++){
				learningX[i][j]=inputs[i][j];
			}
			for(int j=0; j<dSize;j++){
				learningD[i][j]=inputs[i][xSize+j];
			}
		}
		for(int i=0; i<validationSamplesCount; i++){
			for(int j=0; j<xSize;j++){
				validationX[i][j]=inputs[learningSamplesCount+i][j];
			}
			for(int j=0; j<dSize;j++){
				validationD[i][j]=inputs[learningSamplesCount+i][xSize+j];
			}
		}
		
		
		for(int k=1; k<=params.epochs; k++){
			double[] epsilon;
			for(int i=0; i<learningSamplesCount; i++){
				myNetwork.setInput(learningX[i]);
				
				epsilon=myNetwork.getOutput();
				for(int j=0; j<dSize; j++){
					epsilon[j]=learningD[i][j] - epsilon[j];
				}
				myNetwork.calculateDeltas(epsilon);
				
				myNetwork.modifyWeigths(params.u);

			}
			
			double costAvg=0;
			for(int i=0; i<validationSamplesCount; i++){
				myNetwork.setInput(validationX[i]);
				
				epsilon=myNetwork.getOutput();
				for(int j=0; j<dSize; j++){
					epsilon[j]=validationD[i][j] - epsilon[j];
				}
				RealVector epsilonVec=new ArrayRealVector(epsilon);
				
				double C=epsilonVec.dotProduct(epsilonVec);
				
				costAvg+=(double)C/((double)validationSamplesCount*dSize);
			}
			System.out.println(costAvg);
		}
		
		
		MI2_IO.writeArchitecture(myNetwork.getArchitecture());
		
		MI2_IO.writeWeightAndBiasValues(myNetwork.getWeightAndBiasValues());
	}
}
