#include <coresets/coreset.hpp>
#include <map>

using namespace coresets;

Coreset::Coreset(size_t targetSize) : TargetSize(targetSize)
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
        printf("            Adding");
    } 
    else
    {
        printf("            Updating");
    }
    
    coresetPoint->Weight += weight;
    printf(" point %ld with weight %0.2f to the coreset\n", coresetPoint->Index, coresetPoint->Weight);
}

void Coreset::addCenter(size_t clusterIndex, double weight)
{
    auto coresetPoint = findPoint(clusterIndex, true);
    if (coresetPoint == nullptr)
    {
        coresetPoint = std::make_shared<WeightedPoint>(clusterIndex, 0.0, true);
        this->points.push_back(coresetPoint);
        printf("            Adding");
    } 
    else
    {
        printf("            Updating");
    }
    coresetPoint->Weight += weight;

    printf(" center c_%ld with weight %0.2f to the coreset\n", coresetPoint->Index, coresetPoint->Weight);
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
