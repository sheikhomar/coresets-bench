#include <utils/random.hpp>

using namespace utils;

RandomIndexer::RandomIndexer(std::mt19937 re, size_t s) : randomEngine2(re), sampler(0, s - 1)
{
}

size_t
RandomIndexer::next()
{
    return sampler(randomEngine2);
}

RandomIndexer
Random::getIndexer(size_t size)
{
    return RandomIndexer(randomEngine, size);
}

double
Random::getDouble()
{
    return pickRandomValue(randomEngine);
}

Random::Random(int fixedSeed)
{
    if (fixedSeed == -1)
    {
        // Source: https://stackoverflow.com/questions/15509270/does-stdmt19937-require-warmup
        std::array<int, 624> seedData;
        std::random_device randomDevice;
        std::generate_n(seedData.data(), seedData.size(), std::ref(randomDevice));
        std::seed_seq randomSeq(std::begin(seedData), std::end(seedData));
        randomEngine.seed(randomSeq);
    }
    else
    {
        randomEngine.seed(static_cast<uint>(fixedSeed));
    }
}

std::shared_ptr<blaze::DynamicVector<size_t>>
Random::runWeightedReservoirSampling(const size_t k, const size_t n, blaze::DynamicVector<size_t> weights)
{
    assert(weights.size() == n);

    auto indexSampler = this->getIndexer(k);
    auto data = std::make_shared<blaze::DynamicVector<size_t>>(k);
    data->reset();

    // Algorithm by M. T. Chao
    double sum = 0;

    // Fill the reservoir array
    for (size_t i = 0; i < k; i++)
    {
        (*data)[i] = i;
        sum = sum + static_cast<double>(weights[i]);
    }

    for (size_t i = k; i < n; i++)
    {
        sum = sum + static_cast<double>(weights[i]);

        // Compute the probability for item i
        double p_i = static_cast<double>(k * weights[i]) / sum;

        // Random value between 0 and 1
        auto q = this->getDouble();

        if (q <= p_i)
        {
            auto sampleIndex = indexSampler.next();
            (*data)[sampleIndex] = i;
        }
    }

    return data;
}

std::shared_ptr<blaze::DynamicVector<size_t>>
Random::choice(const size_t k, const size_t n, blaze::DynamicVector<size_t> weights)
{
    assert(weights.size() == n);

    auto result = std::make_shared<blaze::DynamicVector<size_t>>(k);
    result->reset();

    std::discrete_distribution<size_t> weightedChoice(weights.begin(), weights.end());

    for (size_t i = 0; i < k; i++)
    {
        size_t pickedIndex = weightedChoice(this->randomEngine);
        (*result)[i] = pickedIndex;
    }

    return result;
}

size_t
Random::choice(blaze::DynamicVector<double> weights)
{
    std::discrete_distribution<size_t> weightedChoice(weights.begin(), weights.end());
    size_t pickedIndex = weightedChoice(this->randomEngine);
    return pickedIndex;
}

size_t
Random::stochasticRounding(double value)
{
    auto valueHigh = floor(value);
    auto valueLow = ceil(value);
    auto proba = (value - valueLow) / (valueHigh - valueLow);
    auto randomVal = this->getDouble();
    if (randomVal < proba)
    {
        return static_cast<size_t>(round(valueHigh)); // Round up
    }
    return static_cast<size_t>(round(valueLow)); // Round down
}
