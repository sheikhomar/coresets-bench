#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <map>

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
        std::vector<size_t> points;
        std::vector<double> pointLengths;
        
        RandomRay(const size_t originIndex, const size_t dimensions) : OriginIndex(originIndex), Direction(dimensions)
        {
            random.normal(Direction);
            Direction = Direction / blaze::l2Norm(Direction);
        }

        double
        computeProjectedPointLength(const blaze::DynamicMatrix<double> &data, const size_t otherPointIndex)
        {
            const size_t d = data.columns();
            double rrDotProd = 0.0, rpDotProd = 0.0;
            for (size_t j = 0; j < d; j++)
            {
                double rayVector_j = data.at(OriginIndex, j) + Direction[j];
                rpDotProd += rayVector_j * data.at(otherPointIndex, j);
                rrDotProd += rayVector_j * rayVector_j;
            }
            return rpDotProd / rrDotProd;
        }

        double
        distanceToPoint(const blaze::DynamicMatrix<double> &data, const size_t otherPointIndex)
        {
            const size_t d = data.columns();
            auto projectedPointLength = computeProjectedPointLength(data, otherPointIndex);
            double dotProd = 0.0;
            for (size_t j = 0; j < d; j++)
            {
                auto projectedPoint_j = data.at(OriginIndex, j) + projectedPointLength * Direction[j];
                auto diff = projectedPoint_j - data.at(otherPointIndex, j);
                dotProd += diff * diff;
            }
            return std::sqrt(dotProd);
        }

        void assign(const blaze::DynamicMatrix<double> &data, const size_t pointIndex)
        {
            points.push_back(pointIndex);
            pointLengths.push_back(computeProjectedPointLength(data, pointIndex));
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
        const size_t NumberOfRaysPerCluster;
        const size_t NumberOfClusters;

        RayMaker(size_t targetSamplesInCoreset, size_t k, size_t numberOfRaysPerCluster): TargetSamplesInCoreset(targetSamplesInCoreset), NumberOfClusters(k), NumberOfRaysPerCluster(numberOfRaysPerCluster)
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

            clustering::ClusterAssignmentList clusters(n, k);
            clusterPoints(data, initialSolution, clusters);

            std::vector<std::shared_ptr<RandomRay>> rays;
            for (auto &&centerPoint : initialSolution)
            {
                std::vector<std::shared_ptr<RandomRay>> clusterRays;
                std::cout << "Generating random rays for center " << centerPoint << "\n";
                for (size_t i = 0; i < NumberOfRaysPerCluster; i++)
                {
                    auto ray = std::make_shared<RandomRay>(centerPoint, d);
                    rays.push_back(ray);
                    clusterRays.push_back(ray);
                }

                auto points = clusters.getPointsByCluster(centerPoint);
                std::cout << "Center " << centerPoint << " has " << points->size() << " points.\n";

                for (auto &&p : *points)
                {
                    double bestDistance = std::numeric_limits<double>::max();
                    size_t bestRayIndex = 0;

                    for (size_t r = 0; r < clusterRays.size(); r++)
                    {
                        const double distance = clusterRays[r]->distanceToPoint(data, p);
                        if (distance < bestDistance)
                        {
                            bestDistance = distance;
                            bestRayIndex = r;
                        }
                    }

                    // Assign point to the ray with smallest distance.
                    clusterRays[bestRayIndex]->assign(data, p);
                }
            }

            for (auto &&ray : rays)
            {
                std::cout <<  "Ray " << ray->OriginIndex << " - Number points: " << ray->points.size() << "\n";
            }
            
            return coreset;
        }
    };
}
