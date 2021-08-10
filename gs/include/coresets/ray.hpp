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
#include <blaze/Math.h>

namespace coresets
{
    class RandomRay
    {
    private:
        utils::Random random;

    public:
        const size_t PointIndex;
        blaze::DynamicVector<double> Direction;
        
        RandomRay(const size_t pointIndex, const size_t dimensions) : PointIndex(pointIndex), Direction(dimensions)
        {
            random.normal(Direction);
            Direction = Direction / blaze::l2Norm(Direction);

            // std::cout << "Normalized direction:                 [";
            // for (auto &&entry : Direction)
            // {
            //     std::cout << entry << " ";
            // }
            // std::cout << "]\n";

            // std::cout << "norm(r) = " << blaze::l2Norm(Direction) << "\n";
        }

    };

    class RayMaker
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

        RayMaker(size_t targetSamplesInCoreset, size_t k, double epsilon): TargetSamplesInCoreset(targetSamplesInCoreset), Epsilon(epsilon), NumberOfClusters(k)
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
            size_t numberOfRaysPerCenter = std::ceil(std::pow(Epsilon, -static_cast<double>(d)));
            std::cout << "Number of rays: " << numberOfRaysPerCenter << "\n";

            std::vector<std::shared_ptr<RandomRay>> rays;
            for (auto &&center : initialSolution)
            {
                std::cout << "Generating random rays for center " << center << "\n";
                for (size_t i = 0; i < numberOfRaysPerCenter; i++)
                {
                    auto ray = std::make_shared<RandomRay>(center, d);
                    rays.push_back(ray);    
                }
            }

            // Snap every data point to the closest ray.
            return coreset;
        }
    };
}
