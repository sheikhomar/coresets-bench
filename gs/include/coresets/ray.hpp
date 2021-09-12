#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <map>

#include <clustering/clustering_result.hpp>
#include <clustering/kmeans.hpp>
#include <clustering/kmeans1d.hpp>
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

        size_t
        getNumberOfPoints() const
        {
            return points.size();
        }
    };

    class RayContainer
    {
    private:
        std::map<size_t, std::vector<std::shared_ptr<RandomRay>>> clusterRays;
        const size_t Dimensions;
    public:
        RayContainer(const size_t &dimensions) : clusterRays(), Dimensions(dimensions)
        {
            
        }

        const std::vector<std::shared_ptr<RandomRay>>&
        createRays(const size_t &numberOfRays, const size_t &centerPoint)
        {
            std::vector<std::shared_ptr<RandomRay>> rays;
            for (size_t i = 0; i < numberOfRays; i++)
            {
                rays.push_back(std::make_shared<RandomRay>(centerPoint, Dimensions));
            }
            clusterRays.emplace(centerPoint, rays);
            return clusterRays.at(centerPoint);
        }

        const std::vector<std::shared_ptr<RandomRay>>&
        getRays(const size_t &centerPoint)
        {
            return clusterRays.at(centerPoint);
        }

        void printPython()
        {
            for (auto element : clusterRays)
            {
                auto centerIndex = element.first;
                auto rays = element.second;
                
                std::cout << "Center index " << centerIndex << " - Number rays: " << rays.size() << "\n";

                for (auto &&ray : rays)
                {
                    std::cout << "  Ray " << ray->OriginIndex << " - Number points: " << ray->points.size() << "\n";
                    for (size_t i = 0; i < ray->points.size(); i++)
                    {
                        auto p = ray->points[i];
                        auto l = ray->lengths[i];
                        auto d = ray->distances[i];
                        // std::cout << "  " << p << "  -  l=" << l << "  -  d=" << d << "\n";
                        // printf("     Point %2d  length = %0.4f   distance = %0.4f\n", p, l, d);
                    }
                }
            }

            // Print out ray vectors
            // std::cout << "\nrays = np.array([\n";
            // std::cout << "  [\n";
            // size_t lastIndex = rays[0]->OriginIndex;

            // for (auto &&ray : rays)
            // {
            //     // std::cout << "Processing " << ray->OriginIndex << "\n\n";
            //     if (ray->OriginIndex != lastIndex)
            //     {
            //         std::cout << "  ],\n";
            //         std::cout << "  [\n";
            //     }

            //     std::cout << "    [";
            //     for (size_t j = 0; j < ray->Direction.size(); j++)
            //     {
            //         std::cout << ray->Direction[j] << ",";
            //     }
            //     std::cout << "],\n";
                
            //     lastIndex = ray->OriginIndex;
            // }

            // std::cout << "  ]\n";
            // std::cout << ")]\n";
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
        const size_t MaxNumberOfRaysPerCluster;
        const size_t NumberOfClusters;
        const double TargetPointsFromEachCluster;

        RayMaker(size_t targetSamplesInCoreset, size_t k, size_t maxNumberOfRaysPerCluster): TargetSamplesInCoreset(targetSamplesInCoreset), NumberOfClusters(k), MaxNumberOfRaysPerCluster(maxNumberOfRaysPerCluster), TargetPointsFromEachCluster(static_cast<double>(targetSamplesInCoreset) / static_cast<double>(k))
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

            const size_t k = NumberOfClusters;

            // Compute initial solution S
            clustering::KMeans kMeansAlg(k);
            auto clusters = kMeansAlg.pickInitialCentersViaKMeansPlusPlus(data);
            auto rayContainer = createRays(data, clusters);

            rayContainer->printPython();

            std::cout << "Computing 1D clustering...\n";

            auto centerIndicies = *clusters->getClusterIndices();
            for (auto &&centerPoint : centerIndicies)
            {
                auto nClusterPoints = static_cast<double>(clusters->countPointsInCluster(centerPoint));
                auto rays = rayContainer->getRays(centerPoint);

                std::cout << "  Center index " << centerPoint <<  "  N: " << nClusterPoints << "  Target " << TargetPointsFromEachCluster << "\n";

                for (auto &&ray : rays)
                {
                    size_t nRayPoints = ray->getNumberOfPoints();
                    double rayTargetProportion = static_cast<double>(nRayPoints) / nClusterPoints;
                    double t = rayTargetProportion * TargetPointsFromEachCluster;
                    size_t n1dClusters = std::ceil(rayTargetProportion * TargetPointsFromEachCluster);

                    std::cout << "   Ray " << ray->OriginIndex << " - nPoints: " << nRayPoints <<  " % " << rayTargetProportion << "  n1Dcluster " << n1dClusters << "   t " << t <<  "\n";

                    if (n1dClusters > 1)
                    {
                        std::vector<size_t> clusterLabels(nRayPoints);
                        std::vector<double> centers(n1dClusters);
                        clustering::kmeans1d::cluster(ray->lengths, n1dClusters, clusterLabels.data(), centers.data());

                        std::map<size_t, std::vector<size_t>> clustersAndPoints;
                        for (size_t c = 0; c < n1dClusters; c++)
                        {
                            std::vector<size_t> clusterPoints;
                            clustersAndPoints.emplace(c, clusterPoints);
                        }

                        for (size_t p = 0; p < nRayPoints; p++)
                        {
                            auto pointIndex = ray->points[p];
                            auto clusterLabel = clusterLabels[p];
                            
                            clustersAndPoints.at(clusterLabel).push_back(pointIndex);
                        }

                        for (size_t c = 0; c < n1dClusters; c++)
                        {
                            std::vector<size_t> &clusterPoints = clustersAndPoints.at(c);
                            std::cout << "     Cluster " << c << " with center " << centers[c] << " has " << clusterPoints.size() << " points" << std::endl;
                            for (size_t i = 0; i < clusterPoints.size(); i++)
                            {
                                auto pointIndex = clusterPoints[i];
                                std::cout << "        Point " << pointIndex << std::endl;
                            }
                        }
                    }
                }
            }

            return coreset;
        }

        std::shared_ptr<RayContainer>
        createRays(const blaze::DynamicMatrix<double> &data, std::shared_ptr<clustering::ClusterAssignmentList> clusters)
        {
            const size_t d = data.columns();
            auto rayContainer = std::make_shared<RayContainer>(d);
            auto centerIndicies = *clusters->getClusterIndices();

            for (auto &&centerPoint : centerIndicies)
            {
                auto points = clusters->getPointsByCluster(centerPoint);

                std::cout << "Center " << centerPoint << " has " << points->size() << " points.\n";
                double numberOfPointInCluster = static_cast<double>(points->size());
                size_t numberOfRays = std::min(
                    static_cast<double>(MaxNumberOfRaysPerCluster),
                    std::ceil(numberOfPointInCluster / (TargetPointsFromEachCluster * 2))
                );
                
                auto clusterRays = rayContainer->createRays(numberOfRays, centerPoint);

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

            return rayContainer;
        }

        std::shared_ptr<blaze::DynamicVector<double>>
        calcCenter(const blaze::DynamicMatrix<double> &data, const std::vector<size_t> &pointIndicies) const
        {
            const size_t d = data.columns();

            auto center = std::make_shared<blaze::DynamicVector<double>>();
            center->resize(d);

            // Reset variables.
            center->reset();

            for (auto &&pointIndex : pointIndicies)
            {
                for (size_t j = 0; j < d; j++)
                {
                    center->at(j) += data.at(pointIndex, j);
                }
            }

            // Divide centers by the number of points in each cluster.
            for (size_t j = 0; j < d; j++)
            {
                center->at(j) /= pointIndicies.size();
            }

            return center;
        }

    };
}
