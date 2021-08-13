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
#include <utils/distances.hpp>
#include <blaze/Math.h>

namespace coresets
{
    class RandomRay
    {
    private:
        utils::Random random;

    public:
        const size_t OriginIndex;
        blaze::DynamicVector<double> Direction;
        
        RandomRay(const size_t originIndex, const size_t dimensions) : OriginIndex(originIndex), Direction(dimensions)
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

        double
        distanceToPoint(const blaze::DynamicMatrix<double> &data, const size_t otherPointIndex)
        {
            // To find the distance from a ray to a given point p, we first find the closest point on a ray.
            // This can be done by projecting that point on the line spanned by the ray.
            // We just have to remember that a ray does not span a line but starts at some point (origin)
            // and goes infinitely to a direction.
            const size_t d = data.columns();
            // blaze::DynamicVector<double> rayVector(d);
            
            // Suppose r = a + b is the ray vector where a is the origin and b is the direction 
            // Compute following quantities:
            //                rpDotProd = <r,p> where p is the point and <,> is the dot product
            //                rrDotProd = <r,r>
            //     projectedPointLength = <r,p> / <r,r>
            // The projected point is given by: ( <r,p> / <r,r> ) * r
            // If the length of the projected point is negative then the distance to from ray r to point p is:
            //   || a - p ||
            // Otherwise the distance is computed as:
            //   || p' - p ||
            //  where p' is the projected point onto the ray computed: p' = a + projectedPointLength * b 

            double rrDotProd = 0.0, rpDotProd = 0.0;
            for (size_t j = 0; j < d; j++)
            {
                double rayVector_j = data.at(OriginIndex, j) + Direction[j];
                rpDotProd += rayVector_j * data.at(otherPointIndex, j);
                rrDotProd += rayVector_j * rayVector_j;
            }

            auto projectedPointLength = rpDotProd / rrDotProd;
            double dotProd = 0.0;
            for (size_t j = 0; j < d; j++)
            {
                auto projectedPoint_j = data.at(OriginIndex, j) + projectedPointLength * Direction[j];
                auto diff = projectedPoint_j - data.at(otherPointIndex, j);
                dotProd += diff * diff;
            }
            return std::sqrt(dotProd);
        
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

        void
        clusterPoints(const blaze::DynamicMatrix<double> &data, const std::vector<size_t> centerPoints, clustering::ClusterAssignmentList &clusters)
        {
            const size_t n = data.rows();
            utils::L2NormCalculator l2Norm(data, false);

            for (size_t p = 0; p < n; p++)
            {
                double bestDistance = std::numeric_limits<double>::max();
                size_t bestCluster = 0;

                // Loop through all the clusters.
                for (auto &&c : centerPoints)
                {
                    // Compute the L2 norm between point p and centroid c.
                    const double distance = l2Norm.calc(p, c);

                    // Decide if current distance is better.
                    if (distance < bestDistance)
                    {
                        bestDistance = distance;
                        bestCluster = c;
                    }
                }

                // Assign cluster to the point p.
                clusters.assign(p, bestCluster, bestDistance);
            }
        }

        std::shared_ptr<Coreset>
        run(const blaze::DynamicMatrix<double> &data)
        {
            auto coreset = std::make_shared<Coreset>(TargetSamplesInCoreset);

            const size_t n = data.rows();
            const size_t d = data.columns();
            const size_t k = NumberOfClusters;

            // Compute initial solution S
            clustering::KMeans kMeansAlg(k);
            auto initialSolution = kMeansAlg.pickInitialCentersViaKMeansPlusPlus(data, false);

            // Around the centers of S, shoot epsilon^-d many rays
            size_t numberOfRaysPerCenter = std::ceil(std::pow(Epsilon, -static_cast<double>(d)));
            std::cout << "Number of rays: " << numberOfRaysPerCenter << "\n";

            clustering::ClusterAssignmentList clusters(n, k);
            clusterPoints(data, initialSolution, clusters);

            std::vector<std::shared_ptr<RandomRay>> rays;
            for (auto &&centerPoint : initialSolution)
            {
                std::cout << "Generating random rays for center " << centerPoint << "\n";
                for (size_t i = 0; i < numberOfRaysPerCenter; i++)
                {
                    auto ray = std::make_shared<RandomRay>(centerPoint, d);
                    rays.push_back(ray);
                }

                auto points = clusters.getPointsByCluster(centerPoint);
                std::cout << "Center " << centerPoint << " has " << points->size() << " points.\n";

            }

            // Snap every data point to the closest ray.
            return coreset;
        }
    };
}
