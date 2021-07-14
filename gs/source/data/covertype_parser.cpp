#include <data/covertype_parser.hpp>

using namespace data;
namespace io = boost::iostreams;

std::shared_ptr<blaze::DynamicMatrix<double>>
CovertypeParser::parse(const std::string &filePath)
{
    printf("Opening input file %s...\n", filePath.c_str());

    std::ifstream fileStream(filePath, std::ios_base::in | std::ios_base::binary);
    io::filtering_streambuf<io::input> filteredInputStream;
    if (boost::ends_with(filePath, ".gz"))
    {
        filteredInputStream.push(io::gzip_decompressor());
    }
    filteredInputStream.push(fileStream);
    std::istream inData(&filteredInputStream);

    printf("Preparing Covertype dataset.\n");

    auto dimSize = 54UL;
    auto dataSize = 581012UL;
    printf("Data size: %ld, Dimensions: %ld\n", dataSize, dimSize);

    auto data = std::make_shared<blaze::DynamicMatrix<double>>(dataSize, dimSize);
    data->reset();
    
    size_t lineNo = 0, currentRow = 0;

    while (inData.good())
    {
        std::string line;
        std::getline(inData, line);
        lineNo++;

        std::vector<std::string> splits;
        boost::split(splits, line, boost::is_any_of(","));

        if (splits.size() != dimSize + 1)
        {
            printf("Skipping line no %ld: expected %ld values but got %ld.\n", lineNo, dimSize+1, splits.size());
            continue;
        }

        // By looping up to dimSize, we skip the last attribute (class attribute).
        // This follows the StreamKM++ paper which removed the classification
        // attribute so in total they had 54 attributes.
        for (size_t j = 0; j < dimSize; j++)
        {
            data->at(currentRow, j) = atof(splits[j].c_str());
        }

        currentRow++;
    }

    return data;
}
