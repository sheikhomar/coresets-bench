#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <clustering/clustering_result.hpp>
#include <clustering/kmeans.hpp>
#include <coresets/coreset.hpp>
#include <utils/random.hpp>

namespace coresets
{
    class Ray
    {
    private:
        utils::Random random;
    public:
        /**
         * Number of points that the algorithm should aim to include in the coreset: T
         */
        const size_t TargetSamplesInCoreset;
        const double Epsilon;
        const size_t NumberOfClusters;

        Ray(size_t targetSamplesInCoreset, size_t k, double epsilon): TargetSamplesInCoreset(targetSamplesInCoreset), Epsilon(epsilon), NumberOfClusters(k)
        {

        }

        std::shared_ptr<Coreset>
        run(const blaze::DynamicMatrix<double> &data)
        {
            auto coreset = std::make_shared<Coreset>(TargetSamplesInCoreset);

            size_t d = data.columns();
            size_t k = NumberOfClusters;

            // Compute initial solution S
            clustering::KMeans kMeansAlg(k);
            auto initialSolution = kMeansAlg.pickInitialCentersViaKMeansPlusPlus(data, false);

            // Around the centers of S, shoot epsilon^-d many rays
            
            

            // Snap every data point to the closest ray.



            return coreset;
        }
    };
}
