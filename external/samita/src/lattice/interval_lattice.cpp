/*
 * Copyright (C) 2010 Life Technologies Corporation. All rights reserved.
 */

/*
 * interval_lattice.cpp
 *
 *  Created on: Jul 28, 2010
 *      Author: kennedcj
 */

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/string.hpp>

#include <boost/program_options.hpp>
#include <algorithm>
#include <math.h>
#include "interval_lattice.hpp"
#include "string_util.hpp"
#include "samita.hpp"
#include "gff.hpp"
#include "prune.hpp"

using namespace lifetechnologies;
namespace po = boost::program_options;

typedef Print<Interval> PRINT;
typedef None<Interval> UNFILTERED;

typedef class Down : public None<Interval>
{
	public:

	  virtual bool operator()(Node<Interval> const& node)
	  {
	  	m_current = &node;
	  	return false;
	  }

	  virtual bool operator()(Node<Interval>::iterator edge)
	  {
	  	return m_contains(m_current->label(), (*edge).target().label());
	  }

	protected:
	  const Node<Interval> *  m_current;
	  Contains m_contains;
} DOWN;

typedef class View : public None<Interval>
{
	public:
		View() {}
	  View(std::set<std::string> const& f, std::stack<Interval> & matches) : m_features(f), m_count(0), m_cover(true), m_matches(&matches), m_coverage(0) {}

	  uint32_t getCount() const {return m_count;}

	  //uint32_t getNumberOfCovers() const {return m_covers;}

	  std::set<std::string> getFeatures() const {return m_features;}

	  virtual bool operator()(Node<Interval> const& node)
	  {
	  	/*
	  	m_count++;
	  	if(m_count > 10)
	  		exit(1);


	  	std::cout << "if(includes(" << node.label() << ", {";

	  	std::set<std::string>::iterator feature = m_features.begin(), last = m_features.end();
	  	while(feature != last)
	  	{
	  		std::cout << *feature << " ";
	  		++feature;
	  	}
	  	std::cout << "}";

	  	if(std::includes(node.label().getTypes().begin(), node.label().getTypes().end(), m_features.begin(), m_features.end()))
	  	{
	  		std::cout << "TRUE" << std::endl;
	  		*/
	  	  std::cout << "find covering for ";

	  	  m_current = &node;


	  		std::cout << m_current->label() << std::endl;

	  		bool covering = false;
	  		Node<Interval>::iterator edge = m_current->begin(), last = m_current->end();
	  		while(edge != last)
	  		{
	  			std::cout << "child ";
	  			std::cout << (*edge).target().label() << std::endl;
	  			if(std::includes((*edge).target().label().getTypes().begin(), (*edge).target().label().getTypes().end(), m_features.begin(), m_features.end()) && m_contains(m_current->label(), (*edge).target().label()))
	  			{
	  				std::cout << "found cover" << std::endl;
	  				covering = true;
	  				break;
	  			}

	  			++edge;
	  		}

	  		if(!covering)
	  		{
	  		  m_matches->push(m_current->label());
	  		}
	  		// m_count += pow(-1, abs(node.label().getTypes().size() - m_features.size()));

	  		m_count ++;
	  		std::cout << "count = " << m_count << std::endl;



		  	//std::cout << "} cover = " << (m_cover ? "TRUE" : "FALSE") << std::endl;
		  	m_cover = false;

	  		return false;
	  		/*
	  	}
	  	std::cout << "FALSE" << std::endl;
	  	return true;
	  	*/
	  }

	  virtual bool operator()(Node<Interval>::iterator edge)
	  {
	  	std::cout << "if(includes(" << (*edge).target().label() << ", {";

	    std::set<std::string>::iterator feature = m_features.begin(), last = m_features.end();
	  	while(feature != last)
	  	{
	  	  std::cout << *feature << " ";
	  		++feature;
	  	}
	  	std::cout << "})" << std::endl;

	  	if(std::includes((*edge).target().label().getTypes().begin(), (*edge).target().label().getTypes().end(), m_features.begin(), m_features.end()))
	  	//if(m_current->label().getSize() && (*edge).target().label().getTypes() == m_features)
	  	{
	  		m_cover = true;
	  		std::cout << "COVERS";
	  		//(*m_covers)++;
	  		std::cout << "covers = " << m_covers << std::endl;


	  	  //std::cout << "INCLUDES count: " << m_count << std::endl;
	    }
	  	else
	  	{
	  		std::cout << "DOES NOT COVER";
	  		//++m_count;
	  	  //std::cout << "DOES NOT INCLUDE count: = " << m_count << std::endl;

	  	}



	  	if(m_contains((*edge).target().label(), m_current->label()) && ((*edge).target().label().getTypes() == m_features || !m_cover))
  		{
  			//++m_count;
  			//std::cout << "count: " << m_count << std::endl;
  			// m_intervals.push_back(node.label());
  		}

	  	return m_contains((*edge).target().label(), m_current->label()) || !std::includes((*edge).target().label().getTypes().begin(), (*edge).target().label().getTypes().end(), m_features.begin(), m_features.end());
	  }

	private:
	  bool m_cover;
	  uint32_t * m_covers;
	  uint32_t m_count;
	  uint32_t m_coverage;
	  const Node<Interval> * m_current;
	  std::set<std::string> m_features;
	  std::stack<Interval> * m_matches;
	  // std::vector<Interval> m_intervals;
	  Contains m_contains;
} VIEW;

typedef class Annotate : public Down
{
	public:
	Annotate(std::string const& att="") : m_att(att) {}
	Annotate(Interval const& interval) : m_interval(interval), m_coverage(0)
	{}

	//using Down::operator();


	virtual bool operator()(Node<Interval> const& node)
	{
		const_cast< Interval& >(node.label()).addTypes(m_interval.getTypes());
		// const_cast< Interval& >(node.label()).statistics.add(m_interval);
		//const_cast< Interval& >(node.label()).statistics.m_coverage++;
		//const_cast< Interval& >(node.label()).statistics.m_size_sum += m_interval.getSize();
		//const_cast< Interval& >(node.label()).statistics.m_size_sum_sq += pow(m_interval.getSize(), 2);
		// m_atts.insert(node.label().getAttribute());
		return Down::operator()(node);
	}




	virtual bool operator()(Node<Interval>::iterator edge)
	{
		if(m_contains((*edge).target().label(), m_current->label()))
		{
			const_cast< Interval&>((*edge).target().label()).statistics.addCoverage(m_current->label().statistics.getCoverage());
			const_cast< Interval&>((*edge).target().label()).statistics.addSize(m_interval.getSize());
		}

		return Down::operator()(edge);
	}


	std::set<std::string> const& getAttributes() const
	{
		return m_atts;
	}

	private:
	int32_t m_coverage;
	Interval m_interval;
	std::string m_att;
	std::set<std::string> m_atts;



} ANNOTATE;

/*
 * Hack for testing !!!
 */

typedef class InDelCnv : public Down
{
	public:
	using Down::operator();

	virtual bool operator()(Node<Interval>::iterator edge)
	{
		std::set<std::string> types = (*edge).target().label().getTypes();
		if(types.find("cnv") != types.end() && types.find("deletion") != types.end())
		{
			std::cout << m_current << ";p=" << ((*edge).target().label().getSize() / m_current->label().getSize()) << std::endl;
		}
	}

	private:

} INDELCNV;

/*
 * End hack for testing !!!
 */

const static std::string RED = "\033[0;31m";
const static std::string WHITE = "\033[0m";

static int usage(po::options_description const& description)
{
  std::cerr << RED
   << "\nAuthor: Caleb J Kennedy (caleb.kennedy@lifetech.com)" << std::endl << std::endl
   << description << WHITE << std::endl;
  return 1;
}

static bool getRegion(std::string const& str, std::vector<std::string> & region)
{
	string_util::tokenize(str, ":-", region);

	if(region.size() != 3)
	{
		printf("\033[0;31mbad region: %s\033[0m\n", str.c_str());
		return 0;
	}

  return 1;
}

int main(int argc, char *argv[])
{
	po::options_description description("lattice [options] <region>\n[options]");
	description.add_options()
    ("help,?", "produce help message")
  ;
	po::variables_map vm;

	try
	{
		po::store(po::parse_command_line(argc, argv, description), vm);
	}

	catch(po::unknown_option exception)
	{
		return usage(description);
	}

	po::notify(vm);

	if(vm.count("help"))
	{
		return usage(description);
	}

	std::vector<std::string> top;

	if(!getRegion(argv[argc - 1], top))
	{
		return usage(description);
	}

	IntervalLattice lattice(Interval(top.front(), atoi(top[1].c_str()), atoi(top.back().c_str())));
	std::vector<std::string> tokens;

	std::ofstream fp("test.out");

	do
	{
		std::cout << ">> ";

		tokens.clear();
		std::string directive;
		getline(std::cin, directive);

		if(directive.empty())
		{
			continue;
		}

		string_util::tokenize(directive, " ", tokens);

		//std::cout << "tokens[0] = " << tokens[0] << std::endl;

		char** c_tok = new char*[tokens.size()];
		string_util::charify(c_tok, tokens);

		po::options_description openDescription("open [options] <file>\n[options]");
		openDescription.add_options()
		  ("help,?", "print this help message")
      ("comment,m", po::value<char>()->default_value('#'), "lines beginning with specified character are ignored")
      ("delim,d", po::value<std::string>()->default_value("\t"), "specifies deliminator for flat text file inputs")
      ("column,c", po::value<int32_t>()->default_value(1), "specifies interval column for flat text file inputs")
      ("bam,b", "<file> is in BAM format")
      ("gff,g", "<file> file is in GFF format")
		  ("bed,d", "<file> file is in BED format")
		;

		po::options_description exportDescription("export [options] <file>\n[options]");
		exportDescription.add_options()
		  ("features,f", po::value<int32_t>()->default_value(1))
		;

		po::options_description insertDescription("insert");
		insertDescription.add_options()
		  ("coverage,c", po::value<int32_t>(), "insert interval at specified coverage")
		;

		po::options_description viewDescription("view");
		viewDescription.add_options()
			("region,r", po::value<std::string>(), "view region")
			("verbatim,v", "only view if region has exact match")
			("score,s", "output score")
			("count,c", "output count")
		  ("file,f", po::value<std::string>(), "output filename, std::out if unspecified")
		  ("depth,d", po::value<int32_t>(), "only view intervals above the specified depth threshold")
		  ("graph,v", "output in dot format for Graphviz visualization")
		  ("gff,g", "output in GFF format")
		  ("dual,u", "output intervals in dual (upside-down) orientation")
		  ("color,c", "view node colors")
		;

		if(tokens[0] == "help")
		{
			// TODO: print directives
		}
		else if(tokens[0] == "open")
		{
			po::variables_map cvm;

  		try {
	  		po::store(po::parse_command_line(tokens.size(), c_tok, openDescription), cvm);
		  }

		  catch(po::unknown_option exception)
		  {
			  printf("\033[0;31m%s\033[0m\n", exception.what());
			  usage(openDescription);
			  continue;
		  }

  		po::notify(cvm);

	  	if(cvm.count("help"))
		  {
			  usage(openDescription);
			  continue;
		  }

	  	if(cvm.count("bam"))
	  	{
	  		Samita<> samita(tokens.back().c_str());
	  		Samita<>::iterator alignment = samita.begin(), last = samita.end();

	  		while(alignment != last)
	  		{
	  			Interval interval((*alignment).getSeq(), (*alignment).getStart(), (*alignment).getEnd());
	  			// lattice.insert<UNFILTERED>(interval);
	  			std::cout << interval << std::endl;
	  			++alignment;
	  		}

	  		continue;
	  	}

		  std::string fileName = tokens.back();
			std::ifstream file(fileName.c_str());

			if(file.is_open())
			{
				int32_t lineNumber = 0;

				while(!file.eof())
				{
					++lineNumber;

					if(file.peek() < 0)
						continue;

					if(cvm.count("gff"))
					{
						std::cout << "GFF Format" << std::endl;
					  GffFeature gffFeature;
					  file >> gffFeature;

					  std::cout << "Gff feature size = ";
					  std::cout << gffFeature.getSize();
					  std::cout << std::endl;

					  if(!gffFeature.intersects((*lattice.getBottom<UNFILTERED>()).label()))
					  {
					  	continue;
					  }

					  std::cout << "Feature = ";
					  std::cout << gffFeature << std::endl;
					  std::cout << std::endl;

					  gffFeature.Interval::setAttribute(gffFeature.getType());

					  IntervalLattice::iterator<ANNOTATE> newInterval = lattice.insert<ANNOTATE>(gffFeature);

						ANNOTATE annotate((*newInterval).label());
						lattice.traverse(newInterval, lattice.end<ANNOTATE>(), annotate);
					}
					else if(cvm.count("bed"))
					{
						std::cout << "BED FILE FORMAT" << std::endl;
					}
					else
					{
						std::string line;
						getline(file, line);

						if(line[0] == cvm["comment"].as<char>())
							continue;

						std::vector<std::string> fields;
						string_util::tokenize(line, cvm["delim"].as<std::string>(), fields);

						int32_t column = cvm["column"].as<int32_t>();
						if(column <= 0 || column > fields.size())
						{
							std::cerr << RED << "out of range: column " << column << " line " << lineNumber << WHITE << std::endl;
						}
						else
						{
							Interval interval;
							std::vector<std::string> region;

							if(fields.size() >= 2)
							{
								interval.setAttribute(fields[1]);
							}

							if(getRegion(fields[column - 1], region))
							{
								interval.setSequence(region[0]);
								interval.setInterval(atoi(region[1].c_str()), atoi(region[2].c_str()));


							  if(!interval.intersects((*lattice.getBottom<UNFILTERED>()).label()))
							  {
							  	continue;
							  }

								IntervalLattice::iterator<ANNOTATE> newInterval = lattice.insert<ANNOTATE>(interval);




								if(fields.size() >= 2)
								{
									ANNOTATE annotate(interval);
									lattice.traverse(newInterval, lattice.end<ANNOTATE>(), annotate);
								}
							}
						}
					}
				}
			}
			else
			{
				std::cerr << RED << "cannot open file: "  << fileName << WHITE << std::endl;
			}

			file.close();
		}
		else if(tokens[0] == "load")
		{
			// TODO: serialize interval
			/*
			std::ifstream ifs(tokens.back());

			if(ifs) {
			   boost::archive::binary_iarchive ia(ifs);
			   ia >> lattice;
			} else {
				std::cerr << "\033[0;31m" << "cannot open file" << "\033[0;31m" << std::endl;
			}
			*/
		}
		else if(tokens[0] == "save")
		{
			/*
			std::ofstream ofs(tokens.back());
			boost::archive::binary_oarchive oa(ofs);
			oa << lattice;
			*/
		}
		else if(tokens[0] == "insert")
		{
			Interval interval;
			std::vector<std::string> region;

			if(getRegion(tokens.back(), region))
			{
				interval.setSequence(region[0]);
				interval.setInterval(atoi(region[1].c_str()), atoi(region[2].c_str()));

				po::variables_map cvm;
				po::store(po::parse_command_line(tokens.size(), c_tok, insertDescription), cvm);
				po::notify(cvm);

				int32_t coverage = 1;
				if(cvm.count("coverage")) {
					coverage = cvm["coverage"].as<int32_t>();
				}

				lattice.insert<UNFILTERED>(interval, coverage);
			}
		}
		else if(tokens[0] == "view")
		{
			po::variables_map cvm;
			try
			{
		    po::store(po::parse_command_line(tokens.size(), c_tok, viewDescription), cvm);
			}

			catch(po::unknown_option exception)
			{
			  std::cerr << RED << exception.what() << WHITE << std::endl;
			  usage(viewDescription);
			}

			po::notify(cvm);

		  if(cvm.count("help"))
		  {
			  usage(viewDescription);
			  continue;
		  }

		  std::set<std::string> features;
		  while(tokens.size() >= 2 && *tokens.back().data() != '-')
		  {
		  	features.insert(tokens.back());
		  	tokens.pop_back();
		  }

		  std::set<std::string>::iterator feature = features.begin(), last = features.end();
		  std::cout << "features {";
		  while(feature != last)
		  {
		  	std::cout << *feature << ", ";
		  	++feature;
		  }
		  std::cout << "}" << std::endl;

	    std::ostream * fp;
	    std::ofstream fout;
	    fp = &std::cout;

	    if(cvm.count("file"))
	    {
	    	fout.open(cvm["file"].as<std::string>().c_str());

	    	if(fout.is_open())
	    	{
	    		fp = &fout;
	    	}
	    	else
	    	{
	    		std::cerr << RED << "cannot open file " << cvm["file"].as<std::string>() << WHITE << std::endl;
	    		continue;
	    	}
	    }

		  std::vector<std::string> region;
		  IntervalLattice::iterator<DOWN> start = lattice.begin<DOWN>();

		  if(cvm.count("region"))
		  {
		  	std::vector<std::string> region;

		  	if(getRegion(cvm["region"].as<std::string>(), region))
		  	{
					Interval interval(region[0], atoi(region[1].c_str()), atoi(region[2].c_str()));
				  start = lattice.find<DOWN>(interval);

				  if(cvm.count("verbatim") && (*start).label() != interval)
				  {
				  	std::cerr << RED << "region " << interval.toString() << " not found" << WHITE << std::endl;
				  	continue;
				  }
		  	}
		  }

			//std::cout << "---TRAVERSE LATTICE---" << std::endl;
		  uint32_t count = 0;

		  if(!features.empty())
		  {
		  	uint32_t covers = 0;
		  	uint32_t coverage = 0;

		  	std::stack<Interval> matches;
		    VIEW view(features, matches);
		  	IntervalLattice::iterator<VIEW> start = lattice.getTop(true, view);
		  	std::stack< IntervalLattice::iterator<VIEW> >path = lattice.traverse(start, lattice.end<VIEW>(), view);

		  	std::cout << "path size = " << path.size() << std::endl;

		  	if(cvm.count("verbatim"))
		  	{
		  		while(!matches.empty())
		  		{
		  			++count;

		  			Interval interval = matches.top();
		  			coverage += interval.getSize();

	  		    std::stringstream sst;

	  		    std::set<std::string>::iterator type = interval.getTypes().begin(), last = interval.getTypes().end();
	  		    while(type != last)
	  		    {
	  		     sst << *type << ",";
	  		     ++type;
	  		    }



	  		    if(cvm.count("gff"))
	  		    {
	  		    	GffFeature gffFeature(interval.getSequence(), "SOLID", sst.str(), interval.getStart(), interval.getEnd());
	  		      *fp << gffFeature << std::endl;
	  		    }

	  		    matches.pop();
		  		}

		  		if(cvm.count("count"))
		  		{
		  			std::cout << std::endl << count << " (" << coverage << ")" << std::endl;
		  		}

		  		continue;
		  	}

		  	/*
		  	std::cout << "after traverse feature size = " << view.getFeatures().size() << std::endl;

		  	if(cvm.count("count"))
		  	{
		  		if(cvm.count("verbatim"))
		  		{
		  			std::cout << "verbatim = " << path.size() << " - " << covers << " - 1" << std::endl;
		  			std::cout << std::endl << path.size() - covers - 1 << std::endl;
		  		}
		  		else
		  		{
		  		  std::cout << std::endl << path.size() - 1 << std::endl;
		  		}
		  		continue;
		  	}
		  	*/



		  	Contains contains;
		  				//std::ofstream out("lattice2.dot"); // option -o <file> (output) to file, else std::cout;
		  				//*fp << "digraph G {" << std::endl; // option -a format as simple (adjacency list), else format as dot (graph)


		  				while(!path.empty()) {
		  				  Node<Interval>::iterator edge = (*path.top()).begin(), last = (*path.top()).end();
		  				  while(edge != last) {
		  				    bool show = true;

		  				  Interval left = (*edge).target().label(),
		  				  		     right = (*path.top()).label();

		  				  if(cvm.count("dual"))
		  				  {
		  				  	std::swap(left, right);
		  				  }

		  					// option -d (dual) swap (*edge).target().label().intent() and (*path.top()).label().intent() in subset function.
		  					//Interval left = cvm.count("dual") ? (*path.top()).label() : (*edge).target().label();
		  					//Interval right = left == (*path.top()).label() ? (*edge).target().label() : (*path.top()).label();

		  				  if(contains(left, right))
		  				  {
		  				  	// option -b (binary) output binary set inclusion instead of attribute lists.

		  				  		if(cvm.count("verbatim"))
		  				  		{


		  				  			show = false;
		  				  		}

		  				  		if(show)
		  				  		{
		  				  			if(cvm.count("gff"))
		  				  			{
		  				  		    Interval interval = (*path.top()).label();

		  				  		    std::stringstream sst;

		  				  		    std::set<std::string>::iterator type = interval.getTypes().begin(), last = interval.getTypes().end();
		  				  		    while(type != last)
		  				  		    {
		  				  		     sst << *type << ",";
		  				  		     ++type;
		  				  		    }


   		  				  		  GffFeature gffFeature(interval.getSequence(), "SOLID", sst.str(), interval.getStart(), interval.getEnd());
		    				  		  *fp << gffFeature << std::endl;
		  				  			}
		  				  			else
		  				  			{
				  				  	  *fp << "\"" << (*path.top()).label().toString() << "(" << (*path.top()).label().getSize() << ")\" -> \"" << (*edge).target().label().toString() << "(" << (*edge).target().label().getSize() << ")\"" << std::endl;
		  				  			}
		  				  			++count;
		  				  		}

		  				  }
		  				  	++edge;
		  					}
		  					path.pop();
		  				}

		  				if(cvm.count("count"))
		  				{
		  					std::cout << count << std::endl;
		  				}
		  				//*fp << "}" << std::endl;
		  				continue;
		  }

		  // TODO: make up/down pruner(s)
		  std::stack< IntervalLattice::iterator<DOWN> > path = lattice.traverse(start, lattice.end<DOWN>());
			//std::cout << "---TRAVERSE LATTICE---" << std::endl;


			Contains contains;
			// std::ofstream out("lattice1.dot"); // option -o <file> (output) to file, else std::cout;
			*fp << "digraph G {" << std::endl; // option -a format as simple (adjacency list), else format as dot (graph)
			while(!path.empty()) {
			  Node<Interval>::iterator edge = (*path.top()).begin(), last = (*path.top()).end();
			  while(edge != last) {

			  Interval left = (*edge).target().label(),
			  		     right = (*path.top()).label();

			  if(cvm.count("dual"))
			  {
			  	std::swap(left, right);
			  }

				// option -d (dual) swap (*edge).target().label().intent() and (*path.top()).label().intent() in subset function.
				//Interval left = cvm.count("dual") ? (*path.top()).label() : (*edge).target().label();
				//Interval right = left == (*path.top()).label() ? (*edge).target().label() : (*path.top()).label();

			  if(contains(left, right))
			  {
			  	// option -b (binary) output binary set inclusion instead of attribute lists.
			  	if(cvm.count("gff"))
			  	{
			  		std::cout << dynamic_cast<const GffFeature&>((*edge).target().label()) << std::endl;
			  	}
			  	else
			  	{
			  	  *fp << "\"" << (*path.top()).label().toString() << "(" << (*path.top()).label().getSize() << ")\" -> \"" << (*edge).target().label().toString() << "(" << (*edge).target().label().getSize() << ")\"" << std::endl;
			  	}
			  }
			  	++edge;
				}
				path.pop();
			}
			*fp << "}" << std::endl;

		}
		else
		{
			if(tokens.front() != "exit")
			{
			  std::cerr << RED << "unknown directive: " << tokens.front() << WHITE << std::endl;
			}
		}

		for (unsigned long i=0; i < tokens.size(); i++)
			delete[] c_tok[i];
		delete[] c_tok;

	} while(tokens.empty() || tokens[0] != "exit");
}
