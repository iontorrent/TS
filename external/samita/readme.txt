/*!
  \mainpage Samita C++ Library

	\section sdd Software Design Description
	- \ref introduction 
	- \ref history  
	- \ref system_overview 
	- \ref design_considerations 
		- \ref assumptions  
		- \ref constraints  
		- \ref goals  
	- \ref architecture 
	- \ref policies 
		- \ref code_stye  
		- \ref code_building
		- \ref code_layout 
		- \ref code_docs 
		- \ref code_tests 
		- \ref code_debug 
	- \ref examples 

	\section introduction Introduction
	
	Samita is a C++ library for creating programs and algorithms for next generation sequencing data. 
	Samita is designed to be efficient, flexible, high quality, and to reduce the time to market for programs.  
	Samita provides many valuable classes for processing sequence alignments, genome references, as well as
	many other utilities.  
	
	The Samita library builds upon the <a href="http://samtools.sourceforge.net/"> SAMtools C library</a> by 
	providing a complementary collection of software for C++ programmers.  Where appropriate, Samita provides more efficient 
	and easier to use functionality which gives consumers of the library a competitive advantage over programs built using SAMtools.      

	\section history Release History
	Version 0.5 of the Samita C++ library was originally released in March, 2010 as part of the Bioscope 1.2 release.  
	This was not a fully functional implementation of the library but it was used for the bam2mates and largeindel applications.      

	\section system_overview System Overview 
	
	The Samital library provides classes for many different functions including:
	
	<ul style="LIST-STYLE-TYPE: square">
	<li>Common file format reading & writing</li>
	<li>Efficient data structures</li>
	<li>Iterators</li>
	<li>Filtering</li>
	<li>Multi-threading</li>
	</ul>

	<p>Below is just a partial list of the classes in the Samita library.</p>

	<ul style="LIST-STYLE-TYPE: square">
	<li><strong style="font-weight: bold;">Core cata types</strong> <em style="font-style: italic;">Align, Cigar</em></li>
	<li><strong style="font-weight: bold;">BAM files</strong> <em style="font-style: italic;">BamReader, BamReader::iterator, BamHeader, BamStats</em></li>
	<li><strong style="font-weight: bold;">Advanced alignment files</strong> <em style="font-style: italic;">AlignReader, AlignReader::iterator</em></li>
	<li><strong style="font-weight: bold;">Reference files</strong> <em style="font-style: italic;">ReferenceSequenceReader, ReferenceSequenceReader::iterator</em></li>
	<li><strong style="font-weight: bold;">FASTQ &amp; CSFASTQ</strong> <em style="font-style: italic;">FastqRecord, FastqReader, FastqReader::iterator, FastqWriter</em></li>
	<li><strong style="font-weight: bold;">GFF</strong> <em style="font-style: italic;">GffRecord, GffReader, GffReader::iterator, GffWriter</em></li>
	<li><strong style="font-weight: bold;">Miscellaneous</strong> <em style="font-style: italic;">SequenceInterval, QualityValueArray</em></li>
	</ul>
	
	<p>Below is a partial list of the classes that are scheduled to be included in the Samita library.</p>
	<ul style="LIST-STYLE-TYPE: square">
	<li><strong style="font-weight: bold;">BAM files</strong> <em style="font-style: italic;">BamWriter</em></li>
	<li><strong style="font-weight: bold;">Filtering</strong></li>
	<li><strong style="font-weight: bold;">Parameter specification & validation</strong></li>
	<li><strong style="font-weight: bold;">Logging message formatting & utilities</strong></li>
	</ul>
	

	<p>Feature Wish List.</p>
	<ul style="LIST-STYLE-TYPE: square">
	<li>Remove duplicates when iterating multiple bam files and merging at runtime</li>
	<li>Add support to merge qname sorted bam files at runtime.  Right now only coordinate sorted files can be merged.
  This would likely involve promoting the comparator function to allow the client to control how the order is 
  preserved during runtime merging.</li>
	<li>Remap read group ids at runtime when merging multiple bam files.  Right now read groups must be unique across bam files in order to 
  iterate over multiple bam files.  Should we reassign read group ids using the same schema as Picard?  Probably.</li>
	</ul>

	\section design_considerations Design Considerations

	\subsection assumptions Assumptions and Dependencies 
	
	The Samita library depends on the following third party libraries
	<ul style="LIST-STYLE-TYPE: square">
	<li><a href="http://www.boost.org/">Boost</a></li>
	<li><a href="http://logging.apache.org/log4cxx/index.html">Apache log4cxx</a></li>
	<li><a href="http://samtools.sourceforge.net/">SAMtools</a></li>
	</ul>
	
	As mentioned in the introduction, Samita can be thought of as a C++ wrapper for the <a href="http://samtools.sourceforge.net/">SAMtools library</a>. 
	However, more efficient and easier to use functionality has been implemented in some areas in order to 
	give consumers of the library a competitive advantage over programs build using SAMtools.      

	The Samita library unit tests depend on the <a href="http://sourceforge.net/apps/mediawiki/cppunit/index.php?title=Main_Page">cppunit library</a>
	
	\subsection constraints General Constraints
	
	<strong style="BACKGROUND-COLOR:red">TODO:</strong> Need to pin down the system reqs more
	
	The Samita library is designed to be portable across any 64 bit version of Linux, MacOS, or Windows.  
	However, it is only tested on Cento OS 4.0 and 5.0.      
	
	\subsection goals Goals and Guidelines 

	Below is a list of design goals:	

	<ul style="LIST-STYLE-TYPE: square">
	<li>Keep it simple</li>
	<li>Emphasize speed</li>
	<li>Be sensitive to memory, I/O, and other shared resources</li>
	<li>Promote code consistency and code reuse</li>
	<li>Remain compatible and consistent with STL and Boost libraries</li>
	</ul>
	
	\section architecture System Architecture
	
	The library is organized into the following categories:
	
	<ul style="LIST-STYLE-TYPE: square">
	<li><strong style="font-weight: bold;">common</strong>  Header only classes need by many other Samita libraries
		<ul style="LIST-STYLE-TYPE: square">
		<li>\link lifetechnologies::Interval Interval\endlink</li>
		<li>\link lifetechnologies::Feature Feature\endlink</li>
		<li>\link lifetechnologies::SequenceInterval SequenceInterval\endlink</li>
		<li>\link lifetechnologies::RGStats RGStats\endlink</li>
		<li>\link lifetechnologies::RG RG\endlink</li>
		<li>\link lifetechnologies::SQ SQ\endlink</li>
		<li>\link lifetechnologies::PG PG\endlink</li>
		<li>\link lifetechnologies::Align Align\endlink</li>
		<li>\link lifetechnologies::RecordReader RecordReader\endlink</li>
			<ul style="LIST-STYLE-TYPE: square">
				<li>\link lifetechnologies::RecordReader::record_stream_iterator RecordReader::iterator\endlink</li>
				<li>\link lifetechnologies::RecordReader::record_stream_iterator RecordReader::const_iterator\endlink</li>
			</ul>
		<li>\link lifetechnologies::RecordWriter RecordWriter\endlink</li>
		</ul>
	</li>
	<li><strong style="font-weight: bold;">exception</strong>  Exceptions thrown by the library
		<ul style="LIST-STYLE-TYPE: square">
		<li>\link lifetechnologies::index_creation_exception index_creation_exception\endlink</li>
		<li>\link lifetechnologies::invalid_cigar_operation invalid_cigar_operation\endlink</li>
		<li>\link lifetechnologies::invalid_input_record invalid_input_record\endlink</li>
		<li>\link lifetechnologies::read_group_not_found read_group_not_found\endlink</li>
		<li>\link lifetechnologies::reference_sequence_not_found reference_sequence_not_found\endlink</li>
		<li>\link lifetechnologies::reference_sequence_index_out_of_bounds reference_sequence_index_out_of_bounds\endlink</li>
		</ul>
	</li>
	<li><strong style="font-weight: bold;">align</strong>  Advanced alignment reader and writer</li>
		<ul style="LIST-STYLE-TYPE: square">
		<li>\link lifetechnologies::Cigar Cigar\endlink</li>
		<li>\link lifetechnologies::AlignReader AlignReader\endlink</li>
			<ul style="LIST-STYLE-TYPE: square">
				<li>\link lifetechnologies::AlignReader::align_iterator AlignReader::iterator\endlink</li>
				<li>\link lifetechnologies::AlignReader::align_iterator AlignReader::filter_iterator\endlink</li>
			</ul>
		<li>AlignWriter</li>
		</ul>
	</li>
	<li><strong style="font-weight: bold;">sam</strong>  BAM file reader and writer
		<ul style="LIST-STYLE-TYPE: square">
		<li>\link lifetechnologies::BamHeaderTag BamHeaderTag\endlink</li>
		<li>\link lifetechnologies::BamHeader BamHeader\endlink</li>
		<li>\link lifetechnologies::BamReader BamReader\endlink</li>
			<ul style="LIST-STYLE-TYPE: square">
				<li>BamReader::iterator</li>
				<li>BamReader::filter_iterator</li>
			</ul>
		<li>AlignWriter</li>
		<li>\link lifetechnologies::BasRecord BasRecord\endlink</li>
		<li>BamWriter</li>
		</ul>
	</li>
	<li><strong style="font-weight: bold;">filter</strong>  Useful filters
		<ul style="LIST-STYLE-TYPE: square">
		<li>\link lifetechnologies::FilterPair FilterPair\endlink</li>
		<li>\link lifetechnologies::FilterTriple FilterTriple\endlink</li>
		<li>\link lifetechnologies::FilterChain FilterChain\endlink</li>
		<li>\link lifetechnologies::RequiredFlagFilter RequiredFlagFilter\endlink</li>
		<li>\link lifetechnologies::FlagFilter FlagFilter\endlink</li>
		<li>\link lifetechnologies::MapQualFilter MapQualFilter\endlink</li>
		<li>\link lifetechnologies::StandardFilter StandardFilter\endlink</li>
		<li>\link lifetechnologies::AlignMates AlignMates\endlink</li>
		<li>\link lifetechnologies::MateFilter MateFilter\endlink</li>
		</ul>
	</li>
	<li><strong style="font-weight: bold;">reference</strong>  Reference file reader and writer
		<ul style="LIST-STYLE-TYPE: square">
		<li>\link lifetechnologies::ReferenceSequence ReferenceSequence\endlink</li>
		<li>\link lifetechnologies::ReferenceSequenceReader ReferenceSequenceReader\endlink</li>
			<ul style="LIST-STYLE-TYPE: square">
				<li>ReferenceSequenceReader::iterator</li>
				<li>ReferenceSequenceReader::const_iterator</li>
			</ul>
		</ul>
	</li>
	<li><strong style="font-weight: bold;">gff</strong>  GFF file reader and writer
		<ul style="LIST-STYLE-TYPE: square">
		<li>\link lifetechnologies::GFFReader GFFReader\endlink</li>
			<ul style="LIST-STYLE-TYPE: square">
				<li>GFFReader::iterator</li>
				<li>GFFReader::const_iterator</li>
			</ul>
		<li>\link lifetechnologies::GFFWriter GFFWriter\endlink</li>
		</ul>
	</li>
	<li><strong style="font-weight: bold;">fastq</strong>  FASTQ file reader and writer
		<ul style="LIST-STYLE-TYPE: square">
		<li>\link lifetechnologies::FastqRecord FastqRecord\endlink</li>
		<li>\link lifetechnologies::FastqReader FastqReader\endlink</li>
			<ul style="LIST-STYLE-TYPE: square">
				<li>FastqReader::iterator</li>
				<li>FastqReader::const_iterator</li>
			</ul>
		<li>\link lifetechnologies::FastqWriter FastqWriter\endlink</li>
		</ul>
	</li>
	</ul>
	
	<strong style="BACKGROUND-COLOR:red">TODO:</strong> need enumerate what parts of SAMtools were re-implemented and why \n 
	<strong style="BACKGROUND-COLOR:red">TODO:</strong> need enumerate what Boost libraries are used and for what \n 
    <strong style="BACKGROUND-COLOR:red">TODO:</strong> need to describe the error detection & recovery \n
    <strong style="BACKGROUND-COLOR:red">TODO:</strong> need to describe policies for memory management \n
    <strong style="BACKGROUND-COLOR:red">TODO:</strong> need to describe policies for I/O management \n
    <strong style="BACKGROUND-COLOR:red">TODO:</strong> need to describe policies for concurrency & synchronization \n

	\section policies Policies and Tactics 
	
	\subsection code_stye Coding Style 
	Every possible effort has been made to comply with the Life Technologies C++ coding standard.  See
	<a href="http://bioscope.apg.per.na.ab.applera.net/mwiki/index.php/Standards:Home/C%2B%2B/Coding_Standards"> 
	http://bioscope.apg.per.na.ab.applera.net/mwiki/index.php/Standards:Home/C%2B%2B/Coding_Standards</a>

	\subsection code_building Obtaining and Building the Software 
	
	First you must fetch and build the bioscope source code.  
	
	\verbatim
    $ svn co https://scm.appliedbio.sfee-hosted.com/svn/repos/corona/bioscope/trunk
	\endverbatim
	
	Then, try to build the source. 
	
	\verbatim
    $ make
	\endverbatim
	
	In order to run samita based apps, you must set the path for the shared libraries.  You can do that by,

	\verbatim
    $ LIFE_LIBS={path_to_your_trunk}/common/c++/lib:{path_to_your_trunk}/samita/lib:
    $ LD_LIBRARY_PATH=${LIFE_LIBS}:${LD_LIBRARY_PATH}
    $ export LD_LIBRARY_PATH
	\endverbatim

    \subsection code_layout Code Layout

	\verbatim
    samita          Contains the public *.hpp files for the library
    src             Contains the *.cpp and private *.hpp files for the library
    test            Contains the main test file (testRunner.cpp), unit test classes, and the unit test input files
    examples        Contains the usage examples and the example input files
    lib             Destination for shared libraries
    doc             Destination for doxygen documentation
    build 			Contains build related files
	\endverbatim

	\section code_docs Creating Documentation

	Run

	\verbatim
    $ make docs
	\endverbatim

	Then open doc/html/index.html in your favorite browser.

	\subsection code_tests Running the Tests

	The test executable just reads from the example files and
	prints to stdout.  To build and run it (from the "c++" directory):

	\verbatim
    $ make test
	\endverbatim

	\subsection code_debug Debugging

	To build debug versions of the library and/or tests, specify the debug target.

	\verbatim
    $ make debug
	\endverbatim

	\subsection code_assertions Debug Assertions

	Samita++ makes use of asserts in the code.  Those asserts do not get compiled out by default.
	So, unless you want them in your release versions you have to either:

	\li Add -DNDEBUG to your gcc flags

		-or-

	\li Add "#define NDEBUG" to any source file or header that includes a header from the Samita library.
		   If you chose this option make sure you put the #define before the #include for those headers.

    Either method is fine.  In fact, leaving the assertions in for release is also fine but typically they are not.
    
	\section examples Examples
	
	<strong style="BACKGROUND-COLOR:red">TODO:</strong> need to describe the different example applications
	
	Below is a wish list for examples.
	<ul style="LIST-STYLE-TYPE: square">
	<li>Add more examples using STL algorithms</li>
		<ul style="LIST-STYLE-TYPE: square">
		<li>ptr_fun and other functional adaptor functions</li>
		<li>function composition</li>
		</ul>
	</ul>
    
*/
