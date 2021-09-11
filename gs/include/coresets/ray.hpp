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
        std::vector<double> lengths;
        std::vector<double> distances;
        double DirectionDotProduct;
        
        RandomRay(const size_t originIndex, const size_t dimensions) : OriginIndex(originIndex), Direction(dimensions)
        {
            random.normal(Direction);
            Direction = Direction / blaze::l2Norm(Direction);

            DirectionDotProduct = 0;
            for (size_t j = 0; j < dimensions; j++)
            {
                double rayVector_j = Direction[j];
                DirectionDotProduct += Direction[j] * Direction[j];
            }
        }

        double
        computeProjectedPointLength(const blaze::DynamicMatrix<double> &data, const size_t otherPointIndex)
        {
            const size_t d = data.columns();
            double rpDotProd = 0.0;
            for (size_t j = 0; j < d; j++)
            {
                // Change the origin of the other point to the point given by OriginIndex
                // because the Direction vector has its origin at OriginIndex.
                auto otherPoint_j = data.at(otherPointIndex, j) - data.at(OriginIndex, j);
                rpDotProd += otherPoint_j * Direction[j];
            }
            return rpDotProd / DirectionDotProduct;
        }

        double
        distanceToPoint(const blaze::DynamicMatrix<double> &data, const size_t otherPointIndex, const double projectedPointLength)
        {
            const size_t d = data.columns();
            double dotProd = 0.0;
            for (size_t j = 0; j < d; j++)
            {
                // Change the origin of the other point to the point given by OriginIndex
                // because the Direction vector has its origin at OriginIndex.
                auto otherPoint_j = data.at(otherPointIndex, j) - data.at(OriginIndex, j);

                auto projectedPoint_j = projectedPointLength * Direction[j];
                auto diff = projectedPoint_j - otherPoint_j;
                dotProd += diff * diff;
            }
            return std::sqrt(dotProd);
        }

        void assign(const blaze::DynamicMatrix<double> &data, const size_t pointIndex, const double distance, const double projectedPointLength)
        {
            points.push_back(pointIndex);
            distances.push_back(distance);
            lengths.push_back(projectedPointLength);
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
            auto clusters = kMeansAlg.pickInitialCentersViaKMeansPlusPlus(data);
            auto centerIndicies = *clusters->getClusterIndices();

            std::vector<std::shared_ptr<RandomRay>> rays;
            for (auto &&centerPoint : centerIndicies)
            {
                std::vector<std::shared_ptr<RandomRay>> clusterRays;
                std::cout << "Generating random rays for center " << centerPoint << "\n";
                for (size_t i = 0; i < NumberOfRaysPerCluster; i++)
                {
                    auto ray = std::make_shared<RandomRay>(centerPoint, d);
                    rays.push_back(ray);
                    clusterRays.push_back(ray);
                }

                auto points = clusters->getPointsByCluster(centerPoint);
                std::cout << "Center " << centerPoint << " has " << points->size() << " points.\n";

                for (auto &&p : *points)
                {
                    double bestDistance = std::numeric_limits<double>::max();
                    double bestProjectedLength = std::numeric_limits<double>::max();
                    size_t bestRayIndex = 0;

                    for (size_t r = 0; r < clusterRays.size(); r++)
                    {
                        const double projectedLength = clusterRays[r]->computeProjectedPointLength(data, p);
                        const double distance = clusterRays[r]->distanceToPoint(data, p, projectedLength);
                        if (distance < bestDistance)
                        {
                            bestDistance = distance;
                            bestProjectedLength = projectedLength;
                            bestRayIndex = r;
                        }
                    }

                    // Assign point to the ray with smallest distance.
                    clusterRays[bestRayIndex]->assign(data, p, bestDistance, bestProjectedLength);
                }
            }
            
            return coreset;
        }

        void printPython(const std::vector<std::shared_ptr<RandomRay>> rays)
        {
            for (auto &&ray : rays)
            {
                std::cout << "Ray " << ray->OriginIndex << " - Number points: " << ray->points.size() << "\n";
                for (size_t i = 0; i < ray->points.size(); i++)
                {
                    auto p = ray->points[i];
                    auto l = ray->lengths[i];
                    auto d = ray->distances[i];
                    // std::cout << "  " << p << "  -  l=" << l << "  -  d=" << d << "\n";
                    printf("   Point %2d  length = %0.4f   distance = %0.4f\n", p, l, d);
                }
            }

            
            // Print out ray vectors
            std::cout << "\nrays = np.array([\n";
            std::cout << "  [\n";
            size_t lastIndex = rays[0]->OriginIndex;

            for (auto &&ray : rays)
            {
                // std::cout << "Processing " << ray->OriginIndex << "\n\n";
                if (ray->OriginIndex != lastIndex)
                {
                    std::cout << "  ],\n";
                    std::cout << "  [\n";
                }

                std::cout << "    [";
                for (size_t j = 0; j < ray->Direction.size(); j++)
                {
                    std::cout << ray->Direction[j] << ",";
                }
                std::cout << "],\n";
                
                lastIndex = ray->OriginIndex;
            }

            std::cout << "  ]\n";
            std::cout << ")]\n";
        }
    };
}
