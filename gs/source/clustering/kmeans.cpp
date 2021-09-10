#include <clustering/kmeans.hpp>

using namespace clustering;

KMeans::KMeans(size_t k, bool useKmeansPlusPlus, size_t nIter, double convDiff) : NumOfClusters(k), InitKMeansPlusPlus(useKmeansPlusPlus), MaxIterations(nIter), ConvergenceDiff(convDiff)
{
}

std::shared_ptr<ClusteringResult>
KMeans::run(const blaze::DynamicMatrix<double> &data)
{
  std::vector<size_t> initialCenters;

  if (this->InitKMeansPlusPlus)
  {
    initialCenters = this->pickInitialCentersViaKMeansPlusPlus(data);
  }
  else
  {
    utils::Random random;

    auto randomPointGenerator = random.getIndexer(data.rows());

    for (size_t c = 0; c < this->NumOfClusters; c++)
    {
      // Pick a random point p as a cluster center.
      auto randomPoint = randomPointGenerator.next();
      initialCenters.push_back(randomPoint);
    }
  }

  auto centers = copyRows(data, initialCenters);
  return this->runLloydsAlgorithm(data, centers);
}

blaze::DynamicMatrix<double>
KMeans::copyRows(const blaze::DynamicMatrix<double> &data, const std::vector<size_t> &indicesToCopy)
{
  size_t k = indicesToCopy.size();
  size_t d = data.columns();

  blaze::DynamicMatrix<double> centers(k, d);
  for (size_t c = 0; c < k; c++)
  {
    size_t pointIndex = indicesToCopy[c];
    blaze::row(centers, c) = blaze::row(data, pointIndex);
  }
  return centers;
}

void computeSquaredNorms(const blaze::DynamicMatrix<double> &dataPoints, std::vector<double> &squaredNorms)
{
  // std::cout << "Computing squared norms!";
  // utils::StopWatch sw(true);
  double val = 0.0;
  for (size_t i = 0; i < dataPoints.rows(); i++)
  {
    double sumOfSquares = 0.0;
    for (size_t j = 0; j < dataPoints.columns(); j++)
    {
      val = dataPoints.at(i, j);
      if (val != 0.0)
      {
        sumOfSquares += val * val;
      }
    }
    squaredNorms[i] = sumOfSquares;
  }

  //std::cout << " Done  in " << sw.elapsedStr() << "\n";
}

std::vector<size_t>
KMeans::pickInitialCentersViaKMeansPlusPlus(const blaze::DynamicMatrix<double> &data)
{
  utils::Random random;
  size_t n = data.rows();
  size_t d = data.columns();
  size_t k = this->NumOfClusters;
  utils::StopWatch sw(true);

  std::vector<double> dataSquaredNorms;
  dataSquaredNorms.resize(n);
  computeSquaredNorms(data, dataSquaredNorms);

  // Lambda function computes the squared L2 distance between any pair of points.
  // The function will automatically use any precomputed distance if it exists.
  auto calcSquaredL2Norm = [&data, &dataSquaredNorms, d](size_t p1, size_t p2) -> double
  {
    if (p1 == p2)
    {
      return 0.0;
    }

    double dotProd = 0.0, val1 = 0.0, val2 = 0.0;
    for (size_t i = 0; i < d; i++)
    {
      val1 = data.at(p1, i);
      val2 = data.at(p2, i);
      if (val1 != 0.0 && val2 != 0.0) // Only compute for non-zero
      {
        dotProd += val1 * val2;
      }
    }

    return dataSquaredNorms[p1] + dataSquaredNorms[p2] - 2 * dotProd;
  };

  // Track which points are picked as centers.
  std::vector<size_t> pickedPointsAsCenters;
  pickedPointsAsCenters.reserve(k);

  blaze::DynamicVector<double> weights(n);
  for (size_t p1 = 0; p1 < n; p1++)
  {
    weights[p1] = std::numeric_limits<double>::max();
  }

  size_t centerIndex = 0;
  for (size_t c = 0; c < k; c++)
  {
    utils::StopWatch pickCenterSW(true);

    if (c == 0)
    {
      // Pick the first centroid uniformly at random.
      auto randomPointGenerator = random.getIndexer(n);
      centerIndex = randomPointGenerator.next();
    }
    else
    {
      for (size_t p1 = 0; p1 < n; p1++)
      {
        // Compute dist2(p, C_k)
        double distance = calcSquaredL2Norm(p1, centerIndex);

        // Compute min_dist^2(p, C_k-1)
        // Decide if current distance is better.
        if (distance < weights[p1])
        {
          // Set the weight of a given point to be the smallest distance
          // to any of the previously selected center points. We want to
          // select points randomly such that points that are far from
          // any of the selected center points have higher likelihood of
          // being picked as the next candidate center.
          weights[p1] = distance;
        }
      }

      // Pick the index of a point randomly selected based on the weights.
      centerIndex = random.choice(weights);
    }

    std::cout << "Picked point " << centerIndex << " as center for cluster " << c << " in " << pickCenterSW.elapsedStr() << std::endl;
    pickedPointsAsCenters.push_back(centerIndex);
  }

  std::cout << "k-means++ initialization completed in " << sw.elapsedStr() << std::endl;

  return pickedPointsAsCenters;
}

std::shared_ptr<ClusteringResult>
KMeans::runLloydsAlgorithm(const blaze::DynamicMatrix<double> &matrix, blaze::DynamicMatrix<double> centroids)
{
  const size_t n = matrix.rows();
  const size_t d = matrix.columns();
  const size_t k = this->NumOfClusters;

  blaze::DynamicVector<size_t> clusterMemberCounts(k);
  ClusterAssignmentList cal(n, k);

  if (MaxIterations == 0)
  {
    cal.assignAll(matrix, centroids);
  }

  if (MaxIterations > 0)
  {
    std::vector<double> dataSquaredNorms;
    dataSquaredNorms.resize(n);
    computeSquaredNorms(matrix, dataSquaredNorms);

    std::vector<double> centerSquaredNorms;
    centerSquaredNorms.resize(centroids.rows());
    computeSquaredNorms(centroids, centerSquaredNorms);

    // Lambda function computes the squared L2 distance between any pair of points.
    // The function will automatically use any precomputed distance if it exists.
    auto calcSquaredL2Norm = [&matrix, &centroids, d, &dataSquaredNorms, &centerSquaredNorms](size_t p, size_t c) -> double
    {
      double dotProd = 0.0;
      for (size_t i = 0; i < d; i++)
      {
        dotProd += matrix.at(p, i) * centroids.at(c, i);
      }

      return dataSquaredNorms[p] + centerSquaredNorms[c] - 2 * dotProd;
    };

    for (size_t i = 0; i < this->MaxIterations; i++)
    {
      utils::StopWatch iterSW(true);
      // For each data point, assign the centroid that is closest to it.
      for (size_t p = 0; p < n; p++)
      {
        double bestDistance = std::numeric_limits<double>::max();
        size_t bestCluster = 0;

        // Loop through all the clusters.
        for (size_t c = 0; c < k; c++)
        {
          // Compute the L2 norm between point p and centroid c.
          // const double distance = blaze::norm(blaze::row(matrix, p) - blaze::row(centroids, c));
          const double distance = calcSquaredL2Norm(p, c);

          // Decide if current distance is better.
          if (distance < bestDistance)
          {
            bestDistance = distance;
            bestCluster = c;
          }
        }

        // Assign cluster to the point p.
        cal.assign(p, bestCluster, blaze::sqrt(bestDistance));
      }

      // Move centroids based on the cluster assignments.

      // First, save a copy of the centroids matrix.
      blaze::DynamicMatrix<double> oldCentrioids(centroids);

      // Set all elements to zero.
      centroids = 0;           // Reset centroids.
      clusterMemberCounts = 0; // Reset cluster member counts.

      for (size_t p = 0; p < n; p++)
      {
        const size_t c = cal.getCluster(p);
        blaze::row(centroids, c) += blaze::row(matrix, p);
        clusterMemberCounts[c] += 1;
      }

      for (size_t c = 0; c < k; c++)
      {
        const auto count = std::max<size_t>(1, clusterMemberCounts[c]);
        blaze::row(centroids, c) /= count;
      }

      std::cout << "Iteration " << (i + 1) << " took " << iterSW.elapsedStr() << ". ";

      // Recompute the squared distances again.
      computeSquaredNorms(centroids, centerSquaredNorms);

      // Compute the Frobenius norm
      auto diffAbsMatrix = blaze::abs(centroids - oldCentrioids);
      auto diffAbsSquaredMatrix = blaze::pow(diffAbsMatrix, 2); // Square each element.
      auto frobeniusNormDiff = blaze::sqrt(blaze::sum(diffAbsSquaredMatrix));

      std::cout << "Frobenius norm of centroids difference: " << frobeniusNormDiff << "!" << std::endl;

      if (frobeniusNormDiff < this->ConvergenceDiff)
      {
        std::cout << "Stopping k-Means as centroids do not improve. Frobenius norm Diff: " << frobeniusNormDiff << "\n";
        break;
      }
    }
  }

  return std::make_shared<ClusteringResult>(cal, centroids);
}
