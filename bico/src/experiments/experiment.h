#ifndef EXPERIMENT_H
#define EXPERIMENT_H

#include <iostream>
#include <sstream>
#include <fstream>
#include <random>
#include <ctime>
#include <time.h>
#include <chrono>
#include <iomanip>

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filter/bzip2.hpp>

#include "../point/l2metric.h"
#include "../point/squaredl2metric.h"
#include "../point/point.h"
#include "../point/pointweightmodifier.h"
#include "../clustering/bico.h"
#include "../misc/randomness.h"
#include "../misc/randomgenerator.h"
#include "../misc/stopwatch.h"
#include "../datastructure/proxysolution.h"
#include "../point/pointcentroid.h"
#include "../point/pointweightmodifier.h"
#include "../point/realspaceprovider.h"

using namespace CluE;

class Experiment
{
protected:
    size_t DimSize;
    size_t DataSize;
    size_t ClusterSize;
    size_t LowDimSize;
    size_t TargetCoresetSize;
    std::string InputFilePath;
    std::string OutputFilePath;

public:
    void outputResultsToFile(ProxySolution<Point> *sol)
    {
        printf("Write results to %s...\n", OutputFilePath.c_str());

        std::ofstream outData(OutputFilePath, std::ifstream::out);

        // Output coreset size
        outData << sol->proxysets[0].size() << "\n";

        // Output coreset points
        for (size_t i = 0; i < sol->proxysets[0].size(); ++i)
        {
            // Output weight
            outData << sol->proxysets[0][i].getWeight() << " ";
            // Output center of gravity
            for (size_t j = 0; j < sol->proxysets[0][i].dimension(); ++j)
            {
                outData << sol->proxysets[0][i][j];
                if (j < sol->proxysets[0][i].dimension() - 1)
                    outData << " ";
            }
            outData << "\n";
        }
        outData.close();
    }

    virtual void parsePoint(std::vector<double> &result, std::istream &inData)
    {
        throw std::logic_error("parsePoint not yet implemented");
    }

    virtual void prepareFileStream(std::istream &inData)
    {
        throw std::logic_error("prepareFileStream not yet implemented");
    }

    void run()
    {
        printf("Opening input file %s...\n", InputFilePath.c_str());

        namespace io = boost::iostreams;
        std::ifstream fileStream(InputFilePath, std::ios_base::in | std::ios_base::binary);
        io::filtering_streambuf<io::input> filteredInputStream;
        if (boost::ends_with(InputFilePath, ".gz"))
        {
            filteredInputStream.push(io::gzip_decompressor());
        }
        filteredInputStream.push(fileStream);
        std::istream inData(&filteredInputStream);

        prepareFileStream(inData);

        std::string line;
        size_t pointCount = 0;
        StopWatch sw(true);
        Bico<Point> bico(DimSize, DataSize, ClusterSize, LowDimSize, TargetCoresetSize, new SquaredL2Metric(), new PointWeightModifier());

        while (inData.good())
        {
            std::vector<double> coords;
            parsePoint(coords, inData);
            CluE::Point p(coords);

            if (p.dimension() != DimSize)
            {
                std::clog << "Line skipped because line dimension is " << p.dimension() << " instead of " << DimSize << std::endl;
                continue;
            }

            pointCount++;

            if (pointCount % 10000 == 0)
            {
                std::cout << "Read " << pointCount << " points. Run time: " << sw.elapsedStr() << std::endl;
            }

            // Call BICO point update
            bico << p;

            // p.debugNonZero(pointCount, "%2.0f", 15);
            // p.debug(pointCount, "%5.0f", 15);
            // if (pointCount > 5) {
            //     break;
            // }
        }

        std::cout << "Processed " << pointCount << " points. Run time: " << sw.elapsedStr() << "s" << std::endl;

        outputResultsToFile(bico.compute());
    }
};

class CensusExperiment : public Experiment
{
public:
    CensusExperiment()
    {
        this->DimSize = 68UL;
        this->DataSize = 2458285UL;
        this->ClusterSize = 200UL;
        this->LowDimSize = 50UL;
        this->TargetCoresetSize = 40000UL;
        this->InputFilePath = "data/raw/USCensus1990.data.txt";
        this->OutputFilePath = "data/results/USCensus1990.data.txt";
    }

    void prepareFileStream(std::istream &inData)
    {
        std::string line;
        std::getline(inData, line); // Ignore the header line.
        printf("Preparing Census Dataset. Skip first line: %s\n", line.c_str());
    }

    void parsePoint(std::vector<double> &result, std::istream &inData)
    {
        std::string line;
        std::getline(inData, line);

        std::vector<std::string> stringcoords;
        boost::split(stringcoords, line, boost::is_any_of(","));

        result.reserve(stringcoords.size());

        // Skip the first attribute which is `caseid`
        for (size_t i = 1; i < stringcoords.size(); ++i)
            result.push_back(atof(stringcoords[i].c_str()));
    }
};

class CovertypeExperiment : public Experiment
{
public:
    CovertypeExperiment()
    {
        this->DimSize = 54UL;
        this->DataSize = 581012UL;
        this->ClusterSize = 200UL;
        this->LowDimSize = 50UL;
        this->TargetCoresetSize = 40000UL;
        this->InputFilePath = "data/raw/covtype.data.gz";
        this->OutputFilePath = "data/results/covtype.txt";
    }

    void prepareFileStream(std::istream &inData)
    {
        printf("Preparing Covertype.\n");
    }

    void parsePoint(std::vector<double> &result, std::istream &inData)
    {
        std::string line;
        std::getline(inData, line);

        std::vector<std::string> stringcoords;
        boost::split(stringcoords, line, boost::is_any_of(","));

        result.reserve(stringcoords.size());

        // Skip the last attribute because it is the label attribute
        // StreamKM++ paper removed the classification attribute so 
        // in total they have 54 attributes.
        for (size_t i = 0; i < stringcoords.size() - 1; ++i)
            result.push_back(atof(stringcoords[i].c_str()));
    }
};

class EnronExperiment : public Experiment
{
    std::string previousLine;
    size_t previousDocId;
    bool firstPoint;

public:
    EnronExperiment()
    {
        this->ClusterSize = 200UL;
        this->LowDimSize = 50UL;
        this->TargetCoresetSize = 40000UL;
        this->InputFilePath = "data/raw/docword.enron.txt.gz";
        this->OutputFilePath = "data/results/docword.enron.txt";
    }

    void prepareFileStream(std::istream &inData)
    {
        // The format of the docword.*.txt file is 3 header lines, followed by triples:
        // ---
        // D    -> the number of documents
        // W    -> the number of words in the vocabulary
        // NNZ  -> the number of nonzero counts in the bag-of-words
        // docID wordID count
        // docID wordID count
        // ...
        // docID wordID count
        // docID wordID count
        // ---
        
        std::string line;
        std::getline(inData, line); // Read line with D
        this->DataSize = std::stoul(line.c_str());

        printf("Read D = %ld\n", this->DataSize);

        std::getline(inData, line); // Read line with W
        this->DimSize = std::stoul(line.c_str());

        printf("Read W = %ld\n", this->DimSize);

        std::getline(inData, line); // Skip line with NNZ
        printf("Read NNZ = %s\n", line.c_str());

        previousDocId = 0;
        firstPoint = true;
    }

    void parsePoint(std::vector<double> &result, std::istream &inData)
    {
        // Allocate memory and initialise all elements to zero.
        result.resize(this->DimSize, 0.0);

        std::string line;
        if (previousLine.empty()) {
            std::getline(inData, line);
        } else {
            line = previousLine;
        }

        do
        {
            std::vector<std::string> splits;
            boost::split(splits, line, boost::is_any_of(" "));

            auto docId = std::stoul(splits[0]);
            auto wordId = std::stoul(splits[1]) - 1; // Convert to zero-based array indexing
            auto count = static_cast<double>(std::stoul(splits[2]));

            if (firstPoint)
            {
                firstPoint = false;
                previousLine = line;
                previousDocId = docId;
            }

            if (previousDocId != docId)
            {
                // Current line belongs to the next point. Store it for later.
                previousLine = line;
                previousDocId = docId;
                break;
            }

            result[wordId] = count;

            // Read next line
            std::getline(inData, line);

        } while (inData.good());
    }
};

class TowerExperiment : public Experiment
{
public:
    TowerExperiment()
    {
        this->DimSize = 3UL;
        this->DataSize = 4915200UL;
        this->ClusterSize = 200UL;
        this->LowDimSize = 3UL;
        this->TargetCoresetSize = 40000UL;
        this->InputFilePath = "data/raw/Tower.txt";
        this->OutputFilePath = "data/results/Tower.txt";
    }

    void prepareFileStream(std::istream &inData)
    {
        printf("Preparing Tower.\n");
    }

    void parsePoint(std::vector<double> &result, std::istream &inData)
    {
        result.resize(this->DimSize);

        std::string line;
        for (size_t i = 0; i < this->DimSize; i++)
        {
            std::getline(inData, line);
            result[i] = static_cast<double>(std::stol(line));
        }
    }
};


#endif
