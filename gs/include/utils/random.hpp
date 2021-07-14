#pragma once

#include <algorithm>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include <blaze/Math.h>
#include <boost/array.hpp>

namespace utils
{
    class RandomIndexer
    {
    public:
        RandomIndexer(std::mt19937 randomEngine, size_t size);
        size_t next();

    private:
        std::mt19937 randomEngine2;
        std::uniform_int_distribution<size_t> sampler;
    };

    class Random
    {
    public:
        /**
         * Returns a random real number in the interval [0.0, 1.0).
         */
        double
        getDouble();

        RandomIndexer
        getIndexer(size_t size);

        std::shared_ptr<blaze::DynamicVector<size_t>>
        runWeightedReservoirSampling(const size_t k, const size_t n, blaze::DynamicVector<size_t> weights);

        /**
         * @brief Randomly select `k` indices from an array of size `n` with replacement.
         * @param k The number of indices to pick from.
         * @param weight A collection of weights associated with each entry.
         */
        std::shared_ptr<blaze::DynamicVector<size_t>>
        choice(const size_t k, const size_t n, blaze::DynamicVector<size_t> weights);

        /**
         * @brief Randomly select an index using the given weights.
         */
        size_t
        choice(blaze::DynamicVector<double> weights);

        /**
         * @brief Select a number of elements from vector uniformly at random.
         * @param elements A collection of elements to sample from.
         * @param numberOfElements The number of samples to return.
         */
        template <typename T>
        std::vector<T>
        choice(const std::vector<T> &elements, const size_t numberOfElements)
        {
            // Notice that the templated class function is implemented in this header
            // file because if we tried to move it in the source file then the linker
            // error with "undefined reference to". For more information about this issue
            // read https://bytefreaks.net/programming-2/c/c-undefined-reference-to-templated-class-function
            std::vector<T> samples;
            auto indexSampler = getIndexer(elements.size());

            for (size_t i = 0; i < numberOfElements; i++)
            {
                auto sampledIndex = indexSampler.next();
                samples.push_back(elements[sampledIndex]);
            }

            return samples;
        }

        /**
         * @brief Stochastically rounds up or down a real number `v` with probability (v-⌊v⌋)/(⌈v⌉-⌊v⌋).
         * See https://nhigham.com/2020/07/07/what-is-stochastic-rounding/
         * @param value The floating point number to be round up or down.
         */
        size_t
        stochasticRounding(double value);

        /**
         * @brief Initialises random class.
         * @param fixedSeed The seed for random number generators. Use a value other than -1 to make the randomized algorithms deterministic. Choose -1 to generate a random seed.
         */
        Random(int fixedSeed = 42);

    private:
        std::mt19937 randomEngine;
        std::uniform_real_distribution<> pickRandomValue;
    };
}
