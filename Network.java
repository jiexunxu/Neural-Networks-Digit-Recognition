import java.util.*;
import java.io.*;
//
// The network always consist of N comvolution layers, each followed by one subsampling layer. The last subsampling layer
// will output some 1*1 pixel images for the last classifier layer
//
public class Network{
// Indicates if test or train mode
	boolean isTraining;
// The number of convolution layers	
	int num_of_conv_layers;
// The dimension of the label vector
	int dim_label;
// The images for all layers. images_all[0] is input for convolution layer 1, images_all[1] is output for convolution layer
// 1 and input for subsample layer 1, images_all[2] is output for subsample layer 1 and input for convolution layer 2 etc
	image[][] images_all;	
// The kernel vectors for all convolutional layers. kernels_all[i][] is the kernel vectors for convolution layer i
	vector[][] kernels_all;
// The bias vector for all layers (convolution, subsample and classify)
	double[][] bias_all;
// The scalar multiplier for all subsampling layers
	double[][] dev_all;
// The weights vector for the classifier
	double[] weights;		
// The class label of the input image. This one is null if in test mode.
	int[] Y;
// The step size used for updating weights, bias etc
	double step_size;

// The array of covolution layers. Each c_layers[i] is followed by s_layer[i]
	C_layer[] c_layers;
// The array of subsampling layers. Each s_layers[i] is followed by c_layer[i+1]
	S_layer[] s_layers;
// The classifier
	public Classifier classifier;

//
// Constructor
//	
	public Network(InfoReader ir, double ss){
		num_of_conv_layers=ir.num_of_conv_layers;
		images_all=ir.images_all;
		kernels_all=ir.kernels_all;
		bias_all=ir.bias_all;
		dev_all=ir.dev_all;
		weights=ir.weights;
		dim_label=ir.dim_label;
		isTraining=ir.isTraining;
		step_size=ss;	
		createLayersAndClassifier();				
	}

//
// Read the input image to test or train, store its pixel values in image_all[0][0]. 
// Input image's filename is passed as a parameter.
// Input text file is assumed to be of the following format:
// 1 -1 -1 -1 ... (First line is the class label)
// 5 12 25 57 ...... (The last 36*36 lines and columns are the pixel values of the input image)
// ......
//
	public void readInputImage(String filename) throws IOException {
		Scanner reader;		
		reader=new Scanner(new File(filename));
		createClassLabel(reader);
		createInput(reader);		
	}
	
//
// This method is used in test mode only (isTraining=false). In this case this method returns the label of the input
// image
//
	public int classify(){
		int[] label=classifier.classify();
		for(int i=0;i<label.length;i++){
			if(label[i]==1){
				return i;
			}
		}
		return -1;
	}

//
// Train this network with an input image 	
//
	public void train(String filename) throws IOException {
		readInputImage(filename);
		fprop();
		bprop();
	}

//
// Test this network on a particular input. If network guess the correct answer, return true; else return false.
//
	public int test(String filename) throws IOException{
		readInputImage(filename);
		fprop();
		return classify();
	}
	
//
// Forward propagation in this network
//
	public void fprop(){
		for(int i=0;i<num_of_conv_layers;i++){
			c_layers[i].fprop();
			s_layers[i].fprop();
		}
		classifier.fprop();
	}
	
//
// Backward propagation. Assumes that Y in classifier is not null, and out in classifier has been computed
//
	public void bprop(){
		vector dx=classifier.compute_energy_dx();
		dx=classifier.bprop(dx);		
		for(int i=num_of_conv_layers-1;i>=0;i--){
			dx=s_layers[i].bprop(dx);
			dx=c_layers[i].bprop(dx);
		}	
	}
	
	public void copyContents(){
		for(int i=0;i<c_layers.length;i++){
			kernels_all[i]=c_layers[i].kernels;
			bias_all[i*2]=copyArray(c_layers[i].bias);
			bias_all[i*2+1]=copyArray(s_layers[i].bias);
			dev_all[i]=copyArray(s_layers[i].dev);
		}
		bias_all[num_of_conv_layers*2]=copyArray(classifier.bias);
		weights=copyArray(classifier.weights);
	}
	
//
// Write all the information of this network to an output file, which will be read by InfoReader next time.
// type is used to specify whether next type InfoReader read the file, the network will be used for train or test.
// type must be either 0 or 1
//
	public void outputParameters(String filename, int type) throws IOException {		
		copyContents();
		PrintWriter output=new PrintWriter(new File(filename));
		// Print critical information
		output.println(type);
		output.println(dim_label);
		output.println(num_of_conv_layers);
		// Print information related to creating images
		for(int i=0;i<num_of_conv_layers;i++){
			output.print(images_all[i*2+1].length+" ");
		}
		output.println();
		for(int i=0;i<num_of_conv_layers*2+1;i++){
			output.print(images_all[i][0].dim+" ");
		}
		output.println();
		// Print out all the kernels
		for(int i=0;i<kernels_all.length;i++){
			for(int j=0;j<kernels_all[i].length;j++){
				output.print(kernels_all[i][j].toString());
				output.println();
			}
			output.println();
		}
		output.println();
		//Print out all the bias vectors
		for(int i=0;i<bias_all.length;i++){
			for(int j=0;j<bias_all[i].length;j++){
				output.print(bias_all[i][j]+" ");
			}
			output.println();
		}
		output.println();
		//Print out all the scalar multipliers
		for(int i=0;i<dev_all.length;i++){
			for(int j=0;j<dev_all[i].length;j++){
				output.print(dev_all[i][j]+" ");
			}
			output.println();
		}
		output.println();
		//Print out all the weights
		for(int i=0;i<classifier.weights.length;i++){
			output.print(weights[i]+" ");
		}
		output.close();
	}
	
//
// Create the layers and the classifier
//	
	public void createLayersAndClassifier(){
		c_layers=new C_layer[num_of_conv_layers];
		s_layers=new S_layer[num_of_conv_layers];
		for(int i=0;i<num_of_conv_layers;i++){
			c_layers[i]=new C_layer(images_all[i*2], images_all[i*2+1], kernels_all[i], bias_all[i*2], step_size);
			s_layers[i]=new S_layer(images_all[i*2+1], images_all[i*2+2], dev_all[i], bias_all[i*2+1], step_size);
		}
		classifier=new Classifier(images_all[num_of_conv_layers*2], weights, bias_all[num_of_conv_layers*2], Y, dim_label, step_size);
		if(!isTraining){
			classifier.Y=null;
		}
	}

//
// Create and return the class label
//
	public void createClassLabel(Scanner reader){
		Y=new int[dim_label];
		for(int i=0;i<dim_label;i++){
			Y[i]=reader.nextInt();
		}
		classifier.Y=Y;
	}

//
// Creates the input
//
	public void createInput(Scanner reader){
		int dim=images_all[0][0].dim;
		double[][] temp=new double[36][36];		
		for(int i=0;i<dim;i++){
			for(int j=0;j<dim;j++){
				images_all[0][0].set(i, j, reader.nextInt()/255.0);
			}
		}
	}
	
	public double average(double[][] array, int i, int j, int factor){
		double ave=0.0;
		for(int p=0;p<factor;p++){
			for(int q=0;q<factor;q++){
				ave=ave+array[i*factor+p][j*factor+q];
			}
		}
		return ave/(factor*factor);
	}
	
	public static double[] copyArray(double[] a){
		double[] temp=new double[a.length];
		for(int i=0;i<a.length;i++){
			temp[i]=a[i];
		}
		return temp;
	}
}