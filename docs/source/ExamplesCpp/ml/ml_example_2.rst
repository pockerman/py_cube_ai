K-nearest neigbors classification (C++)
=======================================

Overview
---------


Code
----

.. code-block::

	#include "cubeai/base/cubeai_config.h"
	#include "cubeai/base/cubeai_types.h"
	#include "cubeai/base/cubeai_consts.h"
	#include "cubeai/datasets/iris_data_set.h"
	#include "cubeai/io/data_set_loaders.h"
	#include "cubeai/ml/classifiers/k_nearest_neighbors.h"
	#include "cubeai/maths/lp_metric.h"
	#include "cubeai/maths/matrix_utilities.h"

	#include <iostream>


.. code-block::

	namespace ml_example_2{

	using cubeai::real_t;
	using cubeai::uint_t;
	using cubeai::DynVec;
	using cubeai::ml::classifiers::KNearestNeighbors;
	using cubeai::maths::LpMetric;

	class LpMetricWrapper
	{
	public:

	    typedef real_t value_type;

	    LpMetricWrapper()=default;

	    template<typename DataPair>
	    real_t evaluate(const DataPair& v1, const DataPair& v2)const{
		return LpMetric<2>::evaluate(v1.first, v2.first);
	    }
	};

	}

.. code-block::

int main(){

using namespace ml_example_2;


 try{

       cubeai::datasets::IrisDataSet data;

       std::cout<<cubeai::CubeAIConsts::info_str()<<data<<std::endl;


       KNearestNeighbors<cubeai::datasets::IrisDataSet::point_type> classifier(data.n_columns());

       auto comparison = [](const auto& v1, const auto& v2){
           return v1.first[0] == v2.first[0] && v1.first[1] == v2.first[1] && v1.first[2] == v2.first[2] && v1.first[3] == v2.first[3];
       };

       auto info = classifier.fit(data, comparison);
       std::cout<<cubeai::CubeAIConsts::info_str()<<info<<std::endl;

       auto row = data[0];
       std::cout<<"True class="<<row.second<<"->"<<data.get_class_name(row.second)<<std::endl;
       auto index = classifier.template predict<LpMetricWrapper>(row.first, 5);
       std::cout<<"Predicted class="<<row.second<<"->"<<data.get_class_name(index)<<std::endl;

       auto closest_points = classifier.template nearest_k_points<LpMetricWrapper>(row.first, 5);

       std::cout<<cubeai::CubeAIConsts::info_str()<<" Query point is "<<row<<std::endl;
       for(auto& p : closest_points){
           std::cout<<"Distance="<<p.first<<", "<<p.second<<std::endl;
       }

}
catch(std::exception& e){
   std::cout<<e.what()<<std::endl;
}
catch(...){

   std::cout<<"Unknown exception occured"<<std::endl;
}

return 0;
}

Results
-------

Number of rows=150
Number of columns=4
Path=/home/alex/qi3/cubeAI/data/iris_data.csv
Has ones column=false
Class map...
	0->Iris-setosa
	1->Iris-versicolor
	2->Iris-virginica


Trained examples=150
Total training time=0.000350952secs

True class=0->Iris-setosa
Predicted class=0->Iris-setosa
INFO:  Query point is (5.1 3.5 1.4 0.2, 0)
Distance=0, (5.1 3.5 1.4 0.2, 0)
Distance=0.141421, (5 3.6 1.4 0.2, 0)
Distance=0.141421, (5.2 3.5 1.5 0.2, 0)
Distance=0.173205, (5 3.5 1.3 0.3, 0)
Distance=0.469042, (4.9 3.1 1.5 0.1, 0)


