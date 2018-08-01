public class image{ 
	double[][] a;
	int dim;
	public image(int m){
		a=new double[m][m];
		dim=m;
	}
	
	public image(int m, boolean rand, double s){
		a=new double[m][m];
		dim=m;
		for(int i=0;i<dim;i++){
			for(int j=0;j<dim;j++){
				if(true){
					a[i][j]=s+(double)(i+1)*(j+1)/(dim*dim);
				}else{
					a[i][j]=Math.random();
				}
				
			}
		}
	}
	
	public double get(int m, int n){
		return a[m][n];
	}
	
	public void set(int m, int n, double x){
		a[m][n]=x;
	}
	
	public void acc(int m, int n, double x){
		a[m][n]=a[m][n]+x;
	}

//
//Unfold this image. Returns an array output of unfolded images in row major order. i.e output[0] is the window 
//starting at (0, 0), output[1] at (0, 1), output[2] at (0, 2) etc
//	
	public image[] unfold(int conv_n){
		//Maximum position of unfold window
		int max=dim-conv_n;
		//output
		image[] output=new image[(max+1)*(max+1)];
		int ctr=0;	
		//Start unfolding at position (i, j). 	
		for(int i=0;i<=max;i++){
			for(int j=0;j<=max;j++){
				output[ctr]=new image(conv_n);
				for(int m=0;m<conv_n;m++){
					for(int n=0;n<conv_n;n++){
						output[ctr].set(m, n, a[m+i][n+j]);
					}
				}
				ctr++;
			}
		}
		return output;
	}
	
	public String toString(){
		String temp="--------------image: \n";
		for(int i=0;i<dim;i++){
			for(int j=0;j<dim;j++){
				temp=temp+Math.round(a[i][j]*1000)/1000.0+" ";
			}
			temp=temp+"\n";
		}
		temp=temp+"---------------------------------------\n";
		return temp;
	}
}