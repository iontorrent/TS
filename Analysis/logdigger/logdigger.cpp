/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <fstream>

#include <boost/filesystem.hpp>
#include "json/json.h"

#define LOGDIGGER_VERSION "1.0.0"

using namespace std;
namespace fs = boost::filesystem;

typedef struct TimePair
{
	string start;
	string end;
} TimePair;

typedef struct TimingData
{
	string name;
	string date;
	string start;
	string end;
	unsigned int totalTime;
	string bcTrainStart;
	string bcTrainEnd;
	unsigned int bcTrainTime;
	unsigned int bcTrainReads;
	/*string alignTrainStart;
	string alignTrainEnd;
	unsigned int alignTrainTime;
	unsigned int alignTrainReads;*/
	string flowAlignStart;
	string flowAlignEnd;
	unsigned int flowAlignTime;
	string bcRecalStart;
	string bcRecalEnd;
	unsigned int bcRecalTime;
	unsigned int bcRecalReads;
	string ionstatsStart;
	string ionstatsEnd;
	unsigned int ionstatsTime;
	string alignAllStart;
	string alignAllEnd;
	unsigned int alignAllTime;

	TimingData() : name("")
	, date("")
	, start("")
	, end("")
	, totalTime(0)
	, bcTrainStart("")
	, bcTrainEnd("")
	, bcTrainTime(0)
	, bcTrainReads(0)
	/*, alignTrainStart("")
	, alignTrainEnd("")
	, alignTrainTime(0)
	, alignTrainReads(0)*/
	, flowAlignStart("")
	, flowAlignEnd("")
	, flowAlignTime(0)
	, bcRecalStart("")
	, bcRecalEnd("")
	, bcRecalTime(0)
	, bcRecalReads(0)
	, ionstatsStart("")
	, ionstatsEnd("")
	, ionstatsTime(0)
	, alignAllStart("")
	, alignAllEnd("")
	, alignAllTime(0)
	{};

} TimingData;

unsigned int GetTime(const string& start, const string& end)
{
	if(start.length() != 8 || end.length() != 8 || start.find(":") != 2 || end.find(":") != 2 || start.rfind(":") != 5 || end.rfind(":") != 5)
	{
		return 0;
	}

	string shours = start.substr(0, 2);
	string sminutes = start.substr(3, 2);
	string sseconds = start.substr(6, 2);
	string shoure = end.substr(0, 2);
	string sminutee = end.substr(3, 2);
	string sseconde = end.substr(6, 2);

	unsigned int ihours = atoi(shours.c_str());
	unsigned int iminutes = atoi(sminutes.c_str());
	unsigned int iseconds = atoi(sseconds.c_str());
	unsigned int ihoure = atoi(shoure.c_str());
	unsigned int iminutee = atoi(sminutee.c_str());
	unsigned int iseconde = atoi(sseconde.c_str());

	if(ihours > ihoure)
	{
		ihoure += 24;
	}

	return (ihoure * 3600 + iminutee * 60 + iseconde) - (ihours * 3600 + iminutes * 60 + iseconds);
}

void UpdateTimeFields(TimingData& td)
{
	td.totalTime = GetTime(td.start, td.end);
	td.bcTrainTime = GetTime(td.bcTrainStart, td.bcTrainEnd);
	//td.alignTrainTime = GetTime(td.alignTrainStart, td.alignTrainEnd);
	td.flowAlignTime = GetTime(td.flowAlignStart, td.flowAlignEnd);
	td.bcRecalTime = GetTime(td.bcRecalStart, td.bcRecalEnd);
	td.ionstatsTime = GetTime(td.ionstatsStart, td.ionstatsEnd);
	td.alignAllTime = GetTime(td.alignAllStart, td.alignAllEnd);
}

void UpdateMins(TimingData& minTimes, const TimingData& td)
{
	if(minTimes.totalTime > td.totalTime)
	{
		minTimes.totalTime = td.totalTime;
	}
	if(minTimes.bcTrainTime > td.bcTrainTime)
	{
		minTimes.bcTrainTime = td.bcTrainTime;
	}	
	if(minTimes.bcTrainReads > td.bcTrainReads)
	{
		minTimes.bcTrainReads = td.bcTrainReads;
	}	
	/*if(minTimes.alignTrainTime > td.alignTrainTime)
	{
		minTimes.alignTrainTime = td.alignTrainTime;
	}	
	if(minTimes.alignTrainReads > td.alignTrainReads)
	{
		minTimes.alignTrainReads = td.alignTrainReads;
	}*/	
	if(minTimes.flowAlignTime > td.flowAlignTime)
	{
		minTimes.flowAlignTime = td.flowAlignTime;
	}	
	if(minTimes.bcRecalTime > td.bcRecalTime)
	{
		minTimes.bcRecalTime = td.bcRecalTime;
	}	
	if(minTimes.bcRecalReads > td.bcRecalReads)
	{
		minTimes.bcRecalReads = td.bcRecalReads;
	}	
	if(minTimes.ionstatsTime > td.ionstatsTime)
	{
		minTimes.ionstatsTime = td.ionstatsTime;
	}	
	if(minTimes.alignAllTime > td.alignAllTime)
	{
		minTimes.alignAllTime = td.alignAllTime;
	}	
}

void UpdateMaxs(TimingData& maxTimes, const TimingData& td)
{
	if(maxTimes.totalTime < td.totalTime)
	{
		maxTimes.totalTime = td.totalTime;
	}
	if(maxTimes.bcTrainTime < td.bcTrainTime)
	{
		maxTimes.bcTrainTime = td.bcTrainTime;
	}	
	if(maxTimes.bcTrainReads < td.bcTrainReads)
	{
		maxTimes.bcTrainReads = td.bcTrainReads;
	}	
	/*if(maxTimes.alignTrainTime < td.alignTrainTime)
	{
		maxTimes.alignTrainTime = td.alignTrainTime;
	}	
	if(maxTimes.alignTrainReads < td.alignTrainReads)
	{
		maxTimes.alignTrainReads = td.alignTrainReads;
	}*/	
	if(maxTimes.flowAlignTime < td.flowAlignTime)
	{
		maxTimes.flowAlignTime = td.flowAlignTime;
	}	
	if(maxTimes.bcRecalTime < td.bcRecalTime)
	{
		maxTimes.bcRecalTime = td.bcRecalTime;
	}	
	if(maxTimes.bcRecalReads < td.bcRecalReads)
	{
		maxTimes.bcRecalReads = td.bcRecalReads;
	}	
	if(maxTimes.ionstatsTime < td.ionstatsTime)
	{
		maxTimes.ionstatsTime = td.ionstatsTime;
	}	
	if(maxTimes.alignAllTime < td.alignAllTime)
	{
		maxTimes.alignAllTime = td.alignAllTime;
	}	
}

void UpdateTotals(TimingData& totalTimes, const TimingData& td)
{
	totalTimes.totalTime += td.totalTime;
	totalTimes.bcTrainTime += td.bcTrainTime;
	totalTimes.bcTrainReads += td.bcTrainReads;
	//totalTimes.alignTrainTime += td.alignTrainTime;
	//totalTimes.alignTrainReads += td.alignTrainReads;
	totalTimes.flowAlignTime += td.flowAlignTime;
	totalTimes.bcRecalTime += td.bcRecalTime;
	totalTimes.bcRecalReads += td.bcRecalReads;
	totalTimes.ionstatsTime += td.ionstatsTime;
	totalTimes.alignAllTime += td.alignAllTime;
}

void GetAverages(TimingData& aveTimes, const TimingData& totalTimes, const int factor)
{
	if(factor == 0)
	{
		return;
	}

	aveTimes.totalTime = totalTimes.totalTime / factor;
	aveTimes.bcTrainTime = totalTimes.bcTrainTime / factor;
	aveTimes.bcTrainReads = totalTimes.bcTrainReads / factor;
	//aveTimes.alignTrainTime = totalTimes.alignTrainTime / factor;
	//aveTimes.alignTrainReads = totalTimes.alignTrainReads / factor;
	aveTimes.flowAlignTime = totalTimes.flowAlignTime / factor;
	aveTimes.bcRecalTime = totalTimes.bcRecalTime / factor;
	aveTimes.bcRecalReads = totalTimes.bcRecalReads / factor;
	aveTimes.ionstatsTime = totalTimes.ionstatsTime / factor;
	aveTimes.alignAllTime = totalTimes.alignAllTime / factor;
}

string CatDateTime(string& date, const string& t1, const string& t2)
{
    string y = date.substr(0, 4);
    string m = date.substr(5, 2);
    string d = date.substr(8, 2);

	if(t1 > t2)
	{

		int nd = atoi(d.c_str());
		++nd;

		char buf[20];
		if(nd < 10)
		{
			sprintf(buf, "0%d", nd);
		}
		else
		{
			sprintf(buf, "%d", nd);
		}
		d = buf;

		if(nd > 31)
		{
			int nm = atoi(m.c_str());
			d = "01";
			++nm;
			if(nm > 12)
			{
				int ny = atoi(y.c_str());
				++ny;
				m = "01";
				sprintf(buf, "%d", ny);
				y = buf;
			}
			else
			{
				if(nm < 10)
				{
					sprintf(buf, "0%d", nm);
				}
				else
				{
					sprintf(buf, "00%d", nm);
				}
				m = buf;
			}
		}
		else if(nd > 30)
		{
			if(m == "04")
			{
				d = "01";
				m = "05";
			}
			else if(m == "06")
			{
				d = "01";
				m = "07";
			}
			else if(m == "09")
			{
				d = "01";
				m = "10";
			}
			else if(m == "11")
			{
				d = "01";
				m = "12";
			}
		}
		else if(nd > 29)
		{
			if(m == "02")
			{
				d = "01";
				m = "03";
			}
		}
		else if(nd > 28)
		{
			if(m == "02")
			{
				int ny = atoi(y.c_str());
				if((ny % 4) > 0)
				{
					d = "01";
					m = "03";
				}
			}
		}
    }

    date = y;
    date += "-";
    date += m;
    date += "-";
    date += d;
	date += " ";

    string timestr;

    timestr = y;
    timestr += "-";
    timestr += m;
    timestr += "-";
    timestr += d;
    timestr += " ";
    timestr += t2;

    return timestr;
}

void GetTimePair(TimePair& tp, const string& hoststart, const string& hostend)
{
    if(hoststart.length() == 20 && hostend.length() == 8)
	{
		string start = hoststart.substr(0, 8);
		string date = hoststart.substr(16, 4);
		string d1 = hoststart.substr(9, 3);
		if(d1 == "Jan")
		{
			date += "-01-";
		}
		else if(d1 == "Feb")
		{
			date += "-02-";
		}
		else if(d1 == "Mar")
		{
			date += "-03-";
		}
		else if(d1 == "Apr")
		{
			date += "-04-";
		}
		else if(d1 == "May")
		{
			date += "-05-";
		}
		else if(d1 == "Jun")
		{
			date += "-06-";
		}
		else if(d1 == "Jul")
		{
			date += "-07-";
		}
		else if(d1 == "Aug")
		{
			date += "-08-";
		}
		else if(d1 == "Sep")
		{
			date += "-09-";
		}
		else if(d1 == "Oct")
		{
			date += "-10-";
		}
		else if(d1 == "Nov")
		{
			date += "-11-";
		}
		else if(d1 == "Dec")
		{
			date += "-12-";
		}

		date += hoststart.substr(13, 2);
		date += " ";

		tp.start = date;
        tp.start += start;

        tp.end = CatDateTime(date, start, hostend);
	}
}

void SaveTimeStampsToJson(Json::Value & json, const TimingData& td)
{
	string date = "2012-12-21 ";
	if(td.date.length() == 11)
	{
		string d0 = td.date;
		date = d0.substr(7, 4);

		string d1 = d0.substr(0, 3);
		if(d1 == "Jan")
		{
			date += "-01-";
		}
		else if(d1 == "Feb")
		{
			date += "-02-";
		}
		else if(d1 == "Mar")
		{
			date += "-03-";
		}
		else if(d1 == "Apr")
		{
			date += "-04-";
		}
		else if(d1 == "May")
		{
			date += "-05-";
		}
		else if(d1 == "Jun")
		{
			date += "-06-";
		}
		else if(d1 == "Jul")
		{
			date += "-07-";
		}
		else if(d1 == "Aug")
		{
			date += "-08-";
		}
		else if(d1 == "Sep")
		{
			date += "-09-";
		}
		else if(d1 == "Oct")
		{
			date += "-10-";
		}
		else if(d1 == "Nov")
		{
			date += "-11-";
		}
		else if(d1 == "Dec")
		{
			date += "-12-";
		}

		date += d0.substr(4, 2);
		date += " ";
	}

	string timestr = date;
	timestr += td.start;
	json["PipelineTiming"][td.name]["Start"] = timestr;
	timestr = CatDateTime(date, td.start, td.bcTrainEnd);
	json["PipelineTiming"][td.name]["BaseCallerTrain"] = timestr;
	//timestr = CatDateTime(date, td.bcTrainEnd, td.alignTrainEnd);
	//json["PipelineTiming"][td.name]["TrainingAlign"] = timestr;
	timestr = CatDateTime(date, td.bcTrainEnd, td.flowAlignEnd);
	json["PipelineTiming"][td.name]["FlowSpaceAlign"] = timestr;
	timestr = CatDateTime(date, td.flowAlignEnd, td.bcRecalEnd);
	json["PipelineTiming"][td.name]["BaseCallerRecal"] = timestr;
	timestr = CatDateTime(date, td.bcRecalEnd, td.ionstatsEnd);
	json["PipelineTiming"][td.name]["Ionstats"] = timestr;
	timestr = CatDateTime(date, td.ionstatsEnd, td.end);
	json["PipelineTiming"][td.name]["Alignment"] = timestr;
}

void SaveHostTimeToJson(Json::Value & json, const map<string, vector<TimePair> >& timepairs)
{
	if(timepairs.empty())
	{
		return;
	}

	char buf[100];	
    for(map<string, vector<TimePair> >::const_iterator ittp = timepairs.begin(); ittp != timepairs.end(); ++ittp)
	{
		vector<TimePair> vtp = ittp->second;
		for(size_t i = 0; i < vtp.size(); ++i)
		{
			sprintf(buf, "%d", i);
			string index = buf;
			json["HostTiming"][ittp->first][index]["Start"] = vtp[i].start;
			json["HostTiming"][ittp->first][index]["End"] = vtp[i].end;
		}
	}
}

int GetTimeValue(const string& ts0, const string& ts)
{
	string h0 = ts0.substr(0, 2);
	string m0 = ts0.substr(3, 2);
	string s0 = ts0.substr(6, 2);
	int v0 = atoi(h0.c_str()) * 3600 + atoi(m0.c_str()) * 60 + atoi(s0.c_str());
	string h1 = ts.substr(0, 2);
	string m1 = ts.substr(3, 2);
	string s1 = ts.substr(6, 2);
	int v1 = atoi(h1.c_str()) * 3600 + atoi(m1.c_str()) * 60 + atoi(s1.c_str());

	int val = v1 - v0;
	if(val < 0)
	{
		val += (24 * 3600);
	}

	return val;
}

void saveJson(const Json::Value & json, const string& filename_json)
{
	ofstream out(filename_json.c_str(), ios::out);
	if (out.good())
	{
		out << json.toStyledString();
	}
	else
	{
		cout << "logdigger ERROR: unable to write JSON file " << filename_json << endl;
	}
}

void usage() 
{
	cerr << "logdigger - Get timing information from log files" << endl;
	cerr << "Usage: " << endl
	   << "  logdigger input_path [output_path]" << endl;
	exit(1);
}

int main(int argc, const char *argv[]) 
{
	if(argc < 2)
	{
		usage();
	}
	
	if(argc == 2)
	{
		string option = argv[1];
		if("-h" == option)
		{
			usage();
		}
		else if("-v" == option)
		{
			cerr << "logdigger version: " << LOGDIGGER_VERSION << endl;
			usage();
		}
	}

	string path = argv[1];
	if(!fs::exists(path))
	{
		cerr << "logdigger ERROR: " << path << " does not exist." << endl;
		exit(1);	
	}
	
	int index = path.rfind("/");
	string runName(path);
	if(index < path.length() - 1)
	{
		runName = path.substr(index + 1, path.length() - index - 1);
	}

	string timing_txt(path);
	timing_txt += "/sigproc_results/timing.txt";
		
	string explog_txt(path);
	explog_txt += "/explog_final.txt";

	string filename_json(".");
	if(argc > 2)
	{
		filename_json = argv[2];
	}
	filename_json += "/proc_timing.json";

	Json::Value json(Json::objectValue);
	json["RunName"] = runName;
	vector<TimingData> blockTimes;
	TimingData minTimes;
	TimingData maxTimes;
	TimingData aveTimes;
	TimingData totalTimes;

	minTimes.totalTime = 4000000000;
	minTimes.bcTrainTime = 4000000000;
	minTimes.bcTrainReads = 4000000000;
	//minTimes.alignTrainTime = 4000000000;
	//minTimes.alignTrainReads = 4000000000;
	minTimes.flowAlignTime = 4000000000;
	minTimes.bcRecalTime = 4000000000;
	minTimes.bcRecalReads = 4000000000;
	minTimes.ionstatsTime = 4000000000;
	minTimes.alignAllTime = 4000000000;

	map<string, vector<TimePair> > timepairs;

	string pipelineStart("2222-12-31 23:59:59");
	string pipelineEnd("1900-01-01 00:00:00");

    fs::directory_iterator end_iter;
    for(fs::directory_iterator dir_itr(path); dir_itr != end_iter; ++dir_itr)
    {
        if(fs::is_directory(dir_itr->status()))
        {
			string dirname = dir_itr->path().filename();
			if(dirname.length() > 10 && "block_" == dirname.substr(0, 6))
			{
                string logfile = dir_itr->path().string();
				logfile += "/drmaa_stdout_block.txt";

				if(!fs::exists(logfile))
				{
					cerr << "logdigger WARNING: " << logfile << " does not exist." << endl;
					continue;	
				}

                ifstream ifs(logfile.c_str());
				if(ifs)
				{
                    char buf[20000];
					
					bool btrain = false;
					TimingData td;
					td.name = dirname.substr(6, dirname.length() - 6);	
					
					string hostname("");
					string hoststart;
					string hostend;

                    while(ifs.getline(buf, 20000))
					{
						string ss = buf;
						int index = -1;

						if(ss.find("* Hostname") == 0)
                        {
							
							if(ss.length() > 13)
							{
                                string ss2 = ss.substr(11, ss.length() - 13);
                                int index2 = ss2.rfind(" ");
								if(index2 >= 0)
								{
                                    hostname = ss2.substr(index2, ss2.length() - index2);
								}
							}
						}
						else if(ss.find("* Start Time") == 0)
						{
							if(td.start.length() != 8)
							{
								index = ss.find(":");
								if(index > 15)
								{
									td.start = ss.substr(index - 2, 8);
									td.date = ss.substr(index + 7, 11);	
								}
							}
							if(ss.length() > 35)
							{
                                hoststart = ss.substr(ss.length() - 22, 20);
							}
						}
                        else if(ss.find(" ] DEBUG: Calling ") == 10)
						{
							string ss2 = ss.substr(28, ss.length() - 28);
							int index3 = ss2.find(" ");
							if(index3 > 7)
							{

								ss2 = ss2.substr(0, index3);
								int indexBaseCaller = ss2.find("BaseCaller");
								int indexAlignmentQC = ss2.find("alignmentQC");
								int indexIonstats = ss2.find("ionstats");

								if(indexBaseCaller >= 0)
								{
									int index2 = ss.find("--calibration-training");
									if(index2 < 0)
									{
										td.bcRecalStart = ss.substr(2, 8);
									}
									else
									{
										td.bcTrainStart = ss.substr(2, 8);
										btrain = true;
									}
								}
								
								if(indexAlignmentQC >= 0)
								{
									if(td.flowAlignStart.length() != 8)
									{
										td.flowAlignStart = ss.substr(2, 8);
									}
									else if(td.alignAllStart.length() != 8)
									{
										td.alignAllStart = ss.substr(2, 8);
									}
								}

								if(indexIonstats >= 0)
								{
									if(td.ionstatsStart.length() != 8)
									{
										td.ionstatsStart = ss.substr(2, 8);
									}
								}
							}
						}
						else if(ss.find("Basecalling Complete: Elapsed: ") == 0)
						{
							int index2 = ss.find("Timestamp: ");
							if(index2 > 0)
							{
								if(btrain)
								{
									td.bcTrainEnd = ss.substr(index2 + 22, 8);
									btrain = false;
								}
								else
								{
									td.bcRecalEnd = ss.substr(index2 + 22, 8);
								}
							}
						}
						else if(ss.find("BASECALLING: called ") == 0)
						{
							index = ss.find(" of ");
							if(index > 21)
							{
								string nreads = ss.substr(20, index - 20);
								if(btrain)
								{
									td.bcTrainReads = atoi(nreads.c_str());
								}
								else
								{
									td.bcRecalReads = atoi(nreads.c_str());		
								}
							}
						}
						/*else if(td.alignTrainStart.length() != 8 && ss.find(" ] DEBUG: Calling \'alignmentQC.pl ") == 10)
						{
                            td.alignTrainStart = ss.substr(2, 8);
						}
                        else if(td.flowAlignStart.length() != 8 && ss.find(" ] DEBUG: Calling \'java ") == 10)
                        {
                            int index2 = ss.find("FlowspaceCalibration.jar");
                            if(td.alignTrainEnd.length() != 8 && index2 >= 37)
							{
                                td.alignTrainEnd = ss.substr(2, 8);
                                td.flowAlignStart = td.alignTrainEnd;
							}
                        }*/
						else if(ss.find(" ] Finished HP ") == 10)
						{
							td.flowAlignEnd = ss.substr(2, 8);
						}
						else if(ss.find(" ] Finished flow QV ") == 10)
						{
							td.flowAlignEnd = ss.substr(2, 8);
						}
						else if(ss.find(" ] **** Alignment completed ****") == 10)
						{
							td.alignAllEnd = ss.substr(2, 8);
						}
						else if(ss.find(" ] Finished basecaller post processing") == 10)
						{
							td.ionstatsEnd = ss.substr(2, 8);
						}
						else if(ss.find(" ] Completed TFPipeline.processBlock") == 10)
						{
							td.ionstatsEnd = ss.substr(2, 8);
						}
						else if(ss.find(" ] BlockTLScript exit") == 10)
						{
							td.end = ss.substr(2, 8);

                            hostend = ss.substr(2, 8);
							if(hostname.length() > 0)
							{
								TimePair tp;
								GetTimePair(tp, hoststart, hostend);

								map<string, vector<TimePair> >::iterator ittp = timepairs.find(hostname);
								if(ittp == timepairs.end())
								{
                                    vector<TimePair> vtp;
									vtp.push_back(tp);
									timepairs[hostname] = vtp;
								}
								else
								{
									ittp->second.push_back(tp);
								}

								hostname = "";
							}
						}
					}

					ifs.close();

					UpdateTimeFields(td);

					UpdateMins(minTimes, td);
					UpdateMaxs(maxTimes, td);
					UpdateTotals(totalTimes, td);

					SaveTimeStampsToJson(json, td);
					SaveHostTimeToJson(json, timepairs);

					blockTimes.push_back(td);
				}
			}
		}
	}
	
	ifstream ifs(timing_txt.c_str());
	if(ifs)
	{
		int flowTime = 0;
		string flowStart("2222-12-31 23:59:59");
		string flowEnd("1900-01-01 00:00:00");
		char buf[20000];
		ifs.getline(buf, 20000);
		while(ifs.getline(buf, 20000))
		{
			string ss = buf;
			if(ss.length() < 66)
			{
				continue;
			}

			ss = ss.substr(7, ss.length() - 7); // skip TIMING
			int index = ss.find(" ");
			if(index < 1)
			{
				continue;
			}
			
			ss = ss.substr(index + 1, ss.length() - index - 1); // skip run
			index = ss.find(" ");
			if(index < 5)
			{
				continue;
			}

			string block = ss.substr(0, index);
			ss = ss.substr(index + 1, ss.length() - index - 1); // skip block
			index = ss.find(" ");
			if(index < 3)
			{
				continue;
			}

			string chunk = ss.substr(0, index);
			ss = ss.substr(index + 1, ss.length() - index - 1); // skip chunk
			index = ss.find(" ");
			if(index < 1)
			{
				continue;
			}

			string dura = ss.substr(0, index);
			flowTime += atoi(dura.c_str());
			ss = ss.substr(index + 1, ss.length() - index - 1); // skip duration
			index = ss.find(" ");
			if(index < 1)
			{
				continue;
			}

			ss = ss.substr(index + 2, ss.length() - index - 2); // skip threadid
			if(ss.length() < 41)
			{
				continue;
			}

			string start_time = ss.substr(0, 19);
			string end_time = ss.substr(22, 19);

			if(flowStart > start_time)
			{
				flowStart = start_time;
			}

			if(flowEnd < end_time)
			{
				flowEnd = end_time;
			}

			json["FlowTiming"][block][chunk]["Start"] = start_time;
			json["FlowTiming"][block][chunk]["End"] = end_time;
		}

		json["FlowStartTime"] = flowStart;
		json["FlowEndTime"] = flowEnd;
		json["TotalFlowTime"] = flowTime;

		ifs.close();
	}

	ifstream ifs2(explog_txt.c_str());
	if(ifs2)
	{
		char buf2[20000];
		bool skip = true;
		ifs2.getline(buf2, 20000);
		string ts0;
		while(ifs2.getline(buf2, 20000))
		{
			string ss = buf2;
			if(ss.find("ExperimentErrorLog:") == 0)
			{
				break;
			}

			if(ss.find("ExperimentInfoLog:") == 0)
			{
				skip = false;
			}

			if(skip)
			{
				continue;
			}

			string flowName;
			int index = ss.find(".dat: ");
			if(index > 0)
			{
				flowName = ss.substr(0, index);
				int index2 = flowName.rfind(" ");
				flowName = flowName.substr(index2 + 1, flowName.length() - index2 - 1);
				index2 = flowName.rfind("\t");
				flowName = flowName.substr(index2 + 1, flowName.length() - index2 - 1);
			}

			string ts;
			index = ss.find(" time=");
			if(index > 7)
			{
				ts = ss.substr(index + 6, 8);
			}

			if(flowName.length() > 0 && ts.length() == 8)
			{
				if(ts0.length() != 8)
				{
					ts0 = ts;
				}
				json["FlowCaptureTime"][flowName]["Timestamp"] = ts;
				int val = GetTimeValue(ts0, ts);
				json["FlowCaptureTime"][flowName]["TimeValue"] = val;
			}
		}

		ifs2.close();
	}

	cout << "**** logdigger report ****" << endl << endl;
	cout << "Total " << blockTimes.size() << " blocks found for this run" << endl << endl;

	if(blockTimes.size() > 0)
	{
		GetAverages(aveTimes, totalTimes, blockTimes.size());

		cout << "\t\tmin\t\tmax\t\taverage\t\ttotal" << endl;
		cout << "bcTrain\t\t" << minTimes.bcTrainTime << "\t\t" << maxTimes.bcTrainTime << "\t\t" << aveTimes.bcTrainTime << "\t\t" << totalTimes.bcTrainTime << endl;
		//cout << "alignTrain\t" << minTimes.alignTrainTime << "\t\t" << maxTimes.alignTrainTime << "\t\t" << aveTimes.alignTrainTime << "\t\t" << totalTimes.alignTrainTime << endl;
		cout << "flowAlign\t" << minTimes.flowAlignTime << "\t\t" << maxTimes.flowAlignTime << "\t\t" << aveTimes.flowAlignTime << "\t\t" << totalTimes.flowAlignTime << endl;
		cout << "bcRecal\t\t" << minTimes.bcRecalTime << "\t\t" << maxTimes.bcRecalTime << "\t\t" << aveTimes.bcRecalTime << "\t\t" << totalTimes.bcRecalTime << endl;
		cout << "ionstats\t" << minTimes.ionstatsTime << "\t\t" << maxTimes.ionstatsTime << "\t\t" << aveTimes.ionstatsTime << "\t\t" << totalTimes.ionstatsTime << endl;
		cout << "alignAll\t" << minTimes.alignAllTime << "\t\t" << maxTimes.alignAllTime << "\t\t" << aveTimes.alignAllTime << "\t\t" << totalTimes.alignAllTime << endl;
		cout << "blockTime\t" << minTimes.totalTime << "\t\t" << maxTimes.totalTime << "\t\t" << aveTimes.totalTime << "\t\t" << totalTimes.totalTime << endl;

		char buf[100];
		sprintf(buf, "%d", minTimes.bcTrainTime);
		json["bcTrain"]["min"] = buf;	
		sprintf(buf, "%d", maxTimes.bcTrainTime);
		json["bcTrain"]["max"] = buf;	
		sprintf(buf, "%d", aveTimes.bcTrainTime);
		json["bcTrain"]["average"] = buf;	
		sprintf(buf, "%d", totalTimes.bcTrainTime);
		json["bcTrain"]["total"] = buf;	

		/*sprintf(buf, "%d", minTimes.alignTrainTime);
		json["alignTrain"]["min"] = buf;	
		sprintf(buf, "%d", maxTimes.alignTrainTime);
		json["alignTrain"]["max"] = buf;	
		sprintf(buf, "%d", aveTimes.alignTrainTime);
		json["alignTrain"]["average"] = buf;	
		sprintf(buf, "%d", totalTimes.alignTrainTime);
		json["alignTrain"]["total"] = buf;*/		
	
		sprintf(buf, "%d", minTimes.flowAlignTime);
		json["flowAlign"]["min"] = buf;	
		sprintf(buf, "%d", maxTimes.flowAlignTime);
		json["flowAlign"]["max"] = buf;	
		sprintf(buf, "%d", aveTimes.flowAlignTime);
		json["flowAlign"]["average"] = buf;	
		sprintf(buf, "%d", totalTimes.flowAlignTime);
		json["flowAlign"]["total"] = buf;	

		sprintf(buf, "%d", minTimes.bcRecalTime);
		json["bcRecal"]["min"] = buf;	
		sprintf(buf, "%d", maxTimes.bcRecalTime);
		json["bcRecal"]["max"] = buf;	
		sprintf(buf, "%d", aveTimes.bcRecalTime);
		json["bcRecal"]["average"] = buf;	
		sprintf(buf, "%d", totalTimes.bcRecalTime);
		json["bcRecal"]["total"] = buf;	

		sprintf(buf, "%d", minTimes.ionstatsTime);
		json["ionstats"]["min"] = buf;	
		sprintf(buf, "%d", maxTimes.ionstatsTime);
		json["ionstats"]["max"] = buf;	
		sprintf(buf, "%d", aveTimes.ionstatsTime);
		json["ionstats"]["average"] = buf;	
		sprintf(buf, "%d", totalTimes.ionstatsTime);
		json["ionstats"]["total"] = buf;	

		sprintf(buf, "%d", minTimes.alignAllTime);
		json["alignAll"]["min"] = buf;	
		sprintf(buf, "%d", maxTimes.alignAllTime);
		json["alignAll"]["max"] = buf;	
		sprintf(buf, "%d", aveTimes.alignAllTime);
		json["alignAll"]["average"] = buf;	
		sprintf(buf, "%d", totalTimes.alignAllTime);
		json["alignAll"]["total"] = buf;	

		sprintf(buf, "%d", minTimes.totalTime);
		json["blockTime"]["min"] = buf;	
		sprintf(buf, "%d", maxTimes.totalTime);
		json["blockTime"]["max"] = buf;	
		sprintf(buf, "%d", aveTimes.totalTime);
		json["blockTime"]["average"] = buf;	
		sprintf(buf, "%d", totalTimes.totalTime);
		json["blockTime"]["total"] = buf;	
	}
	
	if(blockTimes.size() > 0)
	{
		saveJson(json, filename_json);
	}
	
	exit(0);
}
