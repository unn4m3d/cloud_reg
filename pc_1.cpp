#include "CloudReg.hpp"

int main(int argc, char** argv)
{
    try
    {
        clouds::CloudReg reg;
        reg.run(argc, argv);
    }
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        return -1;
    }
    catch(...)
    {
        std::cerr << "Unknown exception occured" << std::endl;
        return -1;
    }
    return 0;
}