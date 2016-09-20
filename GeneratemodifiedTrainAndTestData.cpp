#include <fstream>
#include <iostream>
#include <string.h>
int main(int argc,char**argv)
{
	char* infileName = argv[1];
	char* outfileName = argv[2];

	std::ifstream infile(infileName);
	std::cout << infile.is_open();
	std::string str;
	std::string old_ppl_id("");
	std::string old_date("");
	std::ofstream outfile(outfileName);

	std::cout<<std::endl<<"Generating duration data from "<< infileName << " to " << outfileName;

	while (std::getline(infile, str))
    {
		std:: string line1 = str;
        const char* cstr = line1.c_str();
		char *saveptr;
		const char* ppl_id = strtok_r((char*)cstr, ",", &saveptr);

		std::string ppl_id_str(ppl_id);
		if(ppl_id == old_ppl_id)
		{
			old_ppl_id = ppl_id;
			for(int i=0;i<2;i++)
			{
				const char* date = strtok_r(NULL, ",", &saveptr);
				if(i==1)
				{
					std::string date_str(date);
					str += ",";
					str += old_date;
					old_date = date_str;
				}
			}
		}else
		{
			old_ppl_id = ppl_id;
			for(int i=0;i<2;i++)
			{
				const char* date = strtok_r(NULL, ",", &saveptr);
				if(i==1)
				{
					std::string date_str(date);
					str += ",";
					str += date_str;
					old_date = date_str;
				}
			}
		}
		outfile<<str<<std::endl;
    }
	outfile.close();
	exit(0);
}
