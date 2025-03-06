#include <iostream>

class logger
{
public:
    void info(const std::string a, bool nl_flag = true)
    {
        if(nl_flag)
        {
            std::cout<<"Info: "<<a<<std::endl;
        }
        else
        {
            std::cout<<"Info: "<<a;
        }
    }

    void warning(const std::string a, bool nl_flag = true)
    {
        if(nl_flag)
        {
            std::cout<<"\033[1;33m"<<"Warning: "<<a<<"\033[0m"<<std::endl;
        }
        else
        {
            std::cout<<"\033[1;33m"<<"Warning: "<<a<<"\033[0m";
        }
    }

    void error(const std::string a, bool nl_flag = true)
    {
        if(nl_flag)
        {
            std::cout<<"\033[1;31m"<<"Error: "<<a<<"\033[0m"<<std::endl;
        }
        else
        {
            std::cout<<"\033[1;31m"<<"Error: "<<a<<"\033[0m";
        }
    }


};