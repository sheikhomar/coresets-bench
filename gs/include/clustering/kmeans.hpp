#pragma once

#include <memory>
#include <iostream>
#include <random>
#include <string>

#include <blaze/Math.h>
#include <boost/array.hpp>
#include <boost/range/algorithm_ext/erase.hpp>

#include <clustering/cluster_assignment_list.hpp>
#include <clustering/clustering_result.hpp>
#include <utils/random.hpp>

namespace clustering
{
    /**
     * @brief Implementation of the k-Means clustering algorithm.
     */
    class KMeans
    {
    public:
        /**
         * @brief Creates a new instance of KMeans.
         * @param numOfClusters The number of clusters to generate.
         * @param initKMeansPlusPlus Initialise centroids using k-Means++.
         * @param precomputeDistances Precompute pairwise distances to speed up computation.
         * @param maxIterations Maximum number of iterations.
         * @param convergenceDiff The difference in the norms of the centroids when to stop k-Means iteration.
         */
        KMeans(uint numOfClusters, bool initKMeansPlusPlus = true, bool precomputeDistances = false, uint maxIterations = 100, double convergenceDiff = 0.0001);

        /**
         * @brief Runs the algorithm.
         * @param data A NxD data matrix containing N data points where each point has D dimensions.
         */
        std::shared_ptr<ClusteringResult>
        run(const blaze::DynamicMatrix<double> &data);

        /**
         * @brief Picks `k` points as the initial centers using the k-Means++ initialisation procedure.
         * @param dataMatrix A NxD data matrix containing N data points where each point has D dimensions.
         * @param precomputeDistances Whether to precompute pairwise distances.
         */
        std::vector<size_t>
        pickInitialCentersViaKMeansPlusPlus(const blaze::DynamicMatrix<double> &dataMatrix, const bool precomputeDistances);

        blaze::DynamicMatrix<double>
        copyRows(const blaze::DynamicMatrix<double> &data, const std::vector<size_t> &indicesToCopy);

    private:
        const size_t NumOfClusters;
        const bool InitKMeansPlusPlus;
        const size_t MaxIterations;
        const double ConvergenceDiff;
        const bool PrecomputeDistances;

        /**
         * @brief Run Lloyd's algorithm to perform the clustering of data points.
         * @param dataMatrix A NxD data matrix containing N data points where each point has D dimensions.
         * @param dataMatrix Initial k centroids where k is the number of required clusters.
         */
        std::shared_ptr<ClusteringResult>
        runLloydsAlgorithm(const blaze::DynamicMatrix<double> &dataMatrix, blaze::DynamicMatrix<double> initialCentroids);
    };

}
