#include <coresets/coreset.hpp>
#include <map>

using namespace coresets;

Coreset::Coreset(size_t targetSize) : TargetSize(targetSize), clusterAssignments(0, 0)
{
}

std::shared_ptr<WeightedPoint>
Coreset::findPoint(size_t index, bool isCenter)
{
    for (size_t i = 0; i < this->points.size(); i++)
    {
        if (points[i]->Index == index && points[i]->IsCenter == isCenter)
        {
            return points[i];
        }
    }

    return nullptr;
}

void Coreset::addPoint(size_t pointIndex, double weight)
{
    auto coresetPoint = findPoint(pointIndex, false);
    if (coresetPoint == nullptr)
    {
        coresetPoint = std::make_shared<WeightedPoint>(pointIndex, 0.0, false);
        this->points.push_back(coresetPoint);
        //printf("            Adding");
    }
    else
    {
        //printf("            Updating");
    }

    coresetPoint->Weight += weight;
    //printf(" point %ld with weight %0.2f to the coreset\n", coresetPoint->Index, coresetPoint->Weight);
}

void Coreset::addCenter(size_t clusterIndex, double weight)
{
    auto coresetPoint = findPoint(clusterIndex, true);
    if (coresetPoint == nullptr)
    {
        coresetPoint = std::make_shared<WeightedPoint>(clusterIndex, 0.0, true);
        this->points.push_back(coresetPoint);
        //printf("            Adding");
    }
    else
    {
        //printf("            Updating");
    }
    coresetPoint->Weight += weight;

    //printf(" center c_%ld with weight %0.2f to the coreset\n", coresetPoint->Index, coresetPoint->Weight);
}

std::shared_ptr<WeightedPoint>
Coreset::at(size_t index) const
{
    return this->points[index];
}

size_t
Coreset::size() const
{
    return this->points.size();
}

void Coreset::setClusterAssignments(const clustering::ClusterAssignmentList &assignments)
{
    this->clusterAssignments = assignments;
}

void Coreset::writeToStream(const blaze::DynamicMatrix<double> &originalDataPoints, std::ostream &out)
{
    std::cout << "Write coreset data to a stream." << std::endl;

    const size_t m = this->points.size();
    const size_t d = originalDataPoints.columns();

    // Output coreset size
    out << m << "\n";

    std::shared_ptr<blaze::DynamicVector<double>> center;

    // Output coreset points
    for (auto &&point : points)
    {
        std::cout << "Writing " << point->Index << std::endl;

        // Output coreset point weight
        out << point->Weight << " ";

        if (point->IsCenter)
        {
            center = this->clusterAssignments.calcCenter(originalDataPoints, point->Index);
        }

        // Output coreset point entries.
        for (size_t j = 0; j < d; ++j)
        {
            if (point->IsCenter)
            {
                out << center->at(j);
            }
            else
            {
                out << originalDataPoints.at(point->Index, j);
            }

            if (j < d - 1)
            {
                out << " ";
            }
        }
        out << "\n";
    }
}
