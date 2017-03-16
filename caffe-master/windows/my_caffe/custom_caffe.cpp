#include <iostream>
#include <caffe\caffe.hpp>
#include <caffe\util\io.hpp>
#include <caffe\blob.hpp>
#include <caffe\common.hpp>
#include <caffe\filler.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <boost\smart_ptr\shared_ptr.hpp>

#pragma comment(lib, "caffe.lib")

using namespace caffe;
using namespace std;
using namespace boost;
using namespace cv;

typedef double Dtype;

// caffe training
int mnist_train(){
	
	// parse solver parameters
	string solver_prototxt = "lenet_solver-leveldb.prototxt";
	caffe::SolverParameter solver_params;
	caffe::ReadProtoFromTextFileOrDie(solver_prototxt, &solver_params);

	// set Device
	Caffe::SetDevice(0);
	Caffe::set_mode(Caffe::GPU);

	// solver handler
	caffe::shared_ptr<caffe::Solver<float>> solver(caffe::SolverRegistry<float>::CreateSolver(solver_params));

	// start Solver
	solver->Solve();
	LOG(INFO) << "Optimization Done.";

	return 0;
}

int main(){
	mnist_train();
}

void inlcude_test()
{
	Blob<Dtype>* const blob = new Blob<Dtype>(20, 30, 40, 50);
	if (blob){
		cout << "Size of blob:";
		cout << " N=" << blob->num();
		cout << " K=" << blob->channels();
		cout << " H=" << blob->height();
		cout << " W=" << blob->width();
		cout << " C=" << blob->count();
		cout << endl;
	}

	FillerParameter filler_param;
	filler_param.set_min(-3);
	filler_param.set_max(3);
	UniformFiller<Dtype> filler(filler_param);
	filler.Fill(blob);

	Dtype expected_asum = 0;
	const Dtype* data = blob->cpu_data();
	for (int i = 0; i < blob->count(); ++i) {
		expected_asum += fabs(data[i]);
	}
	cout << "expected asum of blob: " << expected_asum << endl;
	cout << "asum of blob on cpu: " << blob->asum_data() << endl;
}




