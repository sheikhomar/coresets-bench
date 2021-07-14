#include <coresets/stream_km.hpp>

using namespace coresets;

StreamKMeans::StreamKMeans(size_t targetSamplesInCoreset) : TargetSamplesInCoreset(targetSamplesInCoreset)
{
}

std::shared_ptr<Coreset>
StreamKMeans::run(const blaze::DynamicMatrix<double> &data)
{
    auto coreset = std::make_shared<Coreset>(TargetSamplesInCoreset);

    // Run k-Means++ where k=T where T is the number of points to be included in the coreset
    clustering::KMeans kMeansAlg(TargetSamplesInCoreset);

    auto result = kMeansAlg.run(data);

    auto clusterAssignments = result->getClusterAssignments();

    for (size_t c = 0; c < clusterAssignments.getNumberOfClusters(); c++)
    {
        size_t nPointsInCluster = clusterAssignments.countPointsInCluster(c);
        double weight = static_cast<double>(nPointsInCluster);
        coreset->addCenter(c, weight);
    }
    
    return coreset;
}
