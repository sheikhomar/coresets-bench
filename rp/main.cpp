#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <stddef.h>
#include <cstdlib>
#include <stdexcept>
#include <random>
#include <map>
#include <algorithm>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <iterator>
#include <unordered_map>

#include <boost/array.hpp>
#include <boost/range/algorithm_ext/erase.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filter/bzip2.hpp>
#include "random_engine.h"
#include "matrices.h"
#include "stop_watch.h"
#include "sketches.h"
#include "parsers.h"


template <typename MatrixType>
void printSquaredDistance(MatrixType &data, size_t p1, size_t p2)
{
    size_t D = data.rows();

    double squaredDistance = 0.0;
    double diff = 0.0;
    for (size_t j = 0; j < D; j++)
    {
        diff = data.at(j, p1) - data.at(j, p2);
        squaredDistance += diff * diff;
    }

    printf(" %10.2f ", squaredDistance);
}

template <typename MatrixType>
void printPairwiseSquaredDistances(MatrixType &data, int indices[], size_t numberOfSamples)
{
    auto printLine = [numberOfSamples]()
    {
        std::cout << "  ";
        for (size_t i = 0; i < numberOfSamples; i++)
        {
            std::cout << "------------";
        }
        std::cout << "\n";
    };

    std::cout << "Pairwise distances for points:\n ";
    for (size_t i = 0; i < numberOfSamples; i++)
    {
        printf("%12d", indices[i]);
    }
    std::cout << "\n";

    printLine();
    for (size_t i = 0; i < numberOfSamples; i++)
    {
        printf("  ");
        for (size_t j = 0; j < numberOfSamples; j++)
        {
            auto p1 = static_cast<size_t>(indices[i]);
            auto p2 = static_cast<size_t>(indices[j]);
            printSquaredDistance(data, p1, p2);
        }
        printf("\n");
    }
    printLine();
}

void testPairwiseSquaredDistances()
{
    Matrix data;
    data.allocate(2, 10);
    data.set(0, 0, -7.237310391208174);
    data.set(1, 0, -9.031086522545417);
    data.set(0, 1, -8.16550136087066);
    data.set(1, 1, -7.008504394784431);
    data.set(0, 2, -7.022668436942146);
    data.set(1, 2, -7.570412890908223);
    data.set(0, 3, -8.863943061317665);
    data.set(1, 3, -5.0532398146772355);
    data.set(0, 4, 0.08525185826796045);
    data.set(1, 4, 3.6452829679480585);
    data.set(0, 5, -0.794152276623841);
    data.set(1, 5, 2.104951171962879);
    data.set(0, 6, -1.340520809891421);
    data.set(1, 6, 4.157119493365752);
    data.set(0, 7, -10.32012970766661);
    data.set(1, 7, -4.33740290203162);
    data.set(0, 8, -2.187731658211975);
    data.set(1, 8, 3.333521246686991);
    data.set(0, 9, -8.535604566608127);
    data.set(1, 9, -6.013489256860859);

    int indices[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    printPairwiseSquaredDistances(data, indices, 10);

    // Expected result:
    //    0.00       4.95      2.18     18.47    214.31    165.53    208.7      31.53    178.38     10.79
    //    4.95       0.00      1.62      4.31    181.58    137.39    171.25     11.78    142.69      1.13
    //    2.18       1.62      0.00      9.73    176.31    132.41    169.82     21.33    142.27      4.71
    //   18.47       4.31      9.73      0.00    155.75    116.36    141.43      2.63    114.91      1.03
    //  214.31     181.58    176.31    155.75      0.00      3.15      2.29    172.        5.26    167.61
    //  165.53     137.39    132.41    116.36      3.15      0.00      4.51    132.25      3.45    125.84
    //  208.7      171.25    169.82    141.43      2.29      4.51      0.00    152.79      1.4     155.21
    //   31.53      11.78     21.33      2.63    172.      132.25    152.79      0.00    124.98      5.99
    //  178.38     142.69    142.27    114.91      5.26      3.45      1.4     124.98      0.00    127.66
    //   10.79       1.13      4.71      1.03    167.61    125.84    155.21      5.99    127.66      0.00
}

void runDense()
{
    Matrix data, sketch;

    std::string datasetName = "enron";
    std::string inputPath = "data/input/docword."+datasetName+".txt.gz";
    std::string outputPath = "data/input/sketch-dense-docword."+datasetName+".txt.gz";

    parseBoW("data/input/docword.enron.txt.gz", data, false);

    std::cout << "Running Clarkson Woodruff (CW) algorithm...\n";

    // Use Clarkson Woodruff (CW) algoritm reduce number of dimensions.
    sketch_cw(data, static_cast<size_t>(pow(2, 12)), sketch);

    std::cout << "Sketch generated!\n";

    std::cout << "Writing sketch to " << outputPath << std::endl;
    namespace io = boost::iostreams;
    std::ofstream fileStream(outputPath, std::ios_base::out | std::ios_base::binary);
    io::filtering_streambuf<io::output> fos;
    fos.push(io::gzip_compressor(io::gzip_params(io::gzip::best_compression)));
    fos.push(fileStream);
    std::ostream outData(&fos);

    auto sketch_size = sketch.rows();
    outData << sketch_size << "\n";
    outData << sketch.columns() << "\n";
    outData << "0\n"; // Number of non-zero values is unknown
    size_t columnIndex;
    double value;

    for (size_t i = 0; i < sketch.rows(); i++)
    {
        for (size_t j = 0; j < sketch.columns(); j++)
        {
            value = sketch.at(i, j);
            if (value != 0.0)
            {
                outData << (i+1) << " " << (j+1) << " " << value << "\n";
            }
        }
        
    }
}


void runSparseCsrMatrix()
{
    CooSparseMatrix cooData;
    size_t sketchSize = static_cast<size_t>(pow(2, 16));
    std::string datasetName = "nytimes";
    std::string inputPath = "data/input/docword."+datasetName+".txt.gz";
    std::string outputPath = "data/input/sketch-docword."+datasetName+"."+std::to_string(sketchSize)+".txt.gz";

    parseSparseBoW(inputPath, cooData);

    std::cout << "Data parsing completed!!\n";

    std::cout << "Running Clarkson Woodruff (CW) algorithm...\n";

    CsrMatrix csrData(cooData);
    std::vector<std::map<size_t, double>> sketch;

    sketch_cw_sparse(csrData, sketchSize, sketch);

    std::cout << "Sketch generated!\n";

    std::cout << "Writing sketch to " << outputPath << std::endl;
    namespace io = boost::iostreams;
    std::ofstream fileStream(outputPath, std::ios_base::out | std::ios_base::binary);
    io::filtering_streambuf<io::output> fos;
    fos.push(io::gzip_compressor(io::gzip_params(io::gzip::best_compression)));
    fos.push(fileStream);
    std::ostream outData(&fos);

    auto sketch_size = sketch.size();
    outData << sketch_size << "\n";
    outData << cooData.columns() << "\n";
    outData << "0\n"; // Number of non-zero values is unknown
    size_t columnIndex;
    double value;

    for (size_t rowIndex = 0; rowIndex < sketch_size; rowIndex++)
    {
        for (auto &&pair : sketch[rowIndex])
        {
            columnIndex = pair.first;
            value = pair.second;
            if (value != 0.0)
            {
                outData << (rowIndex+1) << " " << (columnIndex+1) << " " << value << "\n";
            }
        }
    }
}

int main()
{
    /*
    Seeder produces uniformly-distributed unsigned integers with 32 bits of length.
    The entropy of the random_device may be lower than 32 bits.
    It is not a good idea to use std::random_device repeatedly as this may
    deplete the entropy in the system. It relies on system calls which makes it a very slow.
    Ref: https://diego.assencio.com/?index=6890b8c50169ef45b74db135063c227c
    */
    // std::random_device seeder;
    // engine.seed(seeder());
    RandomEngine::get().seed(5489UL); // Use fix seed.

    // runDense();

    runSparseCsrMatrix();

    return 0;
}
