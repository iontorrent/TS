/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */

#include <fstream>
#include <gtest/gtest.h>
#include "OptionArgs.h"

using namespace std;
using namespace ION;

// Exception Catch Macro
#define IONTEST_CATCH                                   \
    catch( std::runtime_error& x )                      \
    {                                                   \
        cerr << "Exception: " << x.what() << endl;      \
        ASSERT_TRUE( false );                           \
    }                                                   \
    catch( ... )                                        \
    {                                                   \
        cerr << "Caught Unknown Exception!"  << endl;   \
        ASSERT_TRUE( false );                           \
    }                                                   \


// Does the file exist?
bool DoesFileExist( const std::string& strFileName )
{
    if( strFileName.empty() )
        return false;
    
    ifstream file( strFileName.c_str() );
    return file.is_open();
}


// Delete the file?
bool DeleteFile( const std::string& strFileName )
{
    bool bRet = false;
    
    if( strFileName.empty() )
        return false;
    
    bool bExists = DoesFileExist( strFileName );
    if( bExists )
    {
        if( 0 == remove( strFileName.c_str() ) )
        {
            bRet = true;
        }
    }
    
    return bRet;
}


// Typical Test Case
TEST(CommandLineTest, TypicalCase)
{
    // Simulated command line argument
    char *argv[] = {
        (char*) "test-prog",
        (char*) "--mult-double", (char*) "0.0,1.0",
        (char*) "--mult-int", (char*) "-1,-10,0",
        (char*) "--hello", (char*) "world",
        (char*) "-b", (char*) "true",
        (char*) "-d", (char*) "2.0",
        (char*) "-i", (char*) "5",
        (char*) "--unchecked", (char*) "fun",
        (char*) "trailing1",
        (char*) "trailing2"
    };

    // The argument count
    int argc = ( sizeof( argv ) / sizeof( argv[0] ) );
    ASSERT_GE( argc, 1 );

    try
    {
        // Create an instance of the OptionsArg class
        // and process the command line arguments.
        OptionArgs options;
        options.Process( argc, argv );

        // Double type test.
        // Getting "mult-double" should return true and have 2 double values.
        // No default value used.
        {
            vector<double> v;
            EXPECT_TRUE( GetOption<double>( options, "mult-double", v ) );
            EXPECT_EQ( v.size(), 2 );
            EXPECT_EQ( v[0], 0.0 );
            EXPECT_EQ( v[1], 1.0 );
        }

        // Int type test.
        // Getting "mult-int" should return true and have 3 int values.
        // No default value used.
        {
            vector<int> v;
            EXPECT_TRUE( GetOption<int>( options, "mult-int", v ) );
            ASSERT_EQ( v.size(), 3 );
            EXPECT_EQ( v[0], -1 );
            EXPECT_EQ( v[1], -10 );
            EXPECT_EQ( v[2], 0 );
        }

        // String type long name test.
        // Getting "hello" should return true and have one value of "world".
        // If "hello" doesn't exist and fails, then use the value of "there".
        {
            vector<string> v;
            EXPECT_TRUE( GetOption<string>( options, "hello", v, "there" ) );
            ASSERT_EQ( v.size(), 1 );
            EXPECT_EQ( v[0], "world" );
        }

        // String type short name test.
        // Getting "b" should return true and return one value of "true".
        // If getting "b" fails, then use "false" as the default value.
        {
            vector<string> v;
            EXPECT_TRUE( GetOption<string>( options, "", "b", v, "false" ) );
            ASSERT_EQ( v.size(), 1 );
            EXPECT_EQ( v[0], "true" );
        }

        // Double type short name test.
        // Getting "d" should return true and return one double value of 2.0.
        // No default value used.
        {
            vector<double> v;
            EXPECT_TRUE( GetOption<double>( options, "", "d", v ) );
            ASSERT_EQ( v.size(), 1 );
            EXPECT_EQ( v[0], 2.0 );
        }

        // Integer type short name test.
        // Getting "i" should return true and return one int value of 5.
        // No default value used.
        {
            vector<int> v;
            EXPECT_TRUE( GetOption<int>( options, "", "i", v ) );
            ASSERT_EQ( v.size(), 1 );
            EXPECT_EQ( v[0], 5 );
        }

        // String type long name test.
        // Getting "unchecked" should return true with one value of "fun".
        // If getting "unchecked" fails, then "nofun" is the default value.
        {
            vector<string> v;        
            EXPECT_TRUE( GetOption<string>( options, "unchecked", v, "nofun" ) );
            ASSERT_EQ( v.size(), 1 );
            EXPECT_EQ( v[0], "fun" );
        }

        // Float type long name test.
        // Getting "PI" should return false and the default value of "3.141" is used.
        {
            vector<float> v;
            EXPECT_FALSE( GetOption<float>( options, "PI", v, "3.141" ) );
            ASSERT_EQ( v.size(), 1 );
            EXPECT_EQ( v[0], 3.141f );
        }

        // Double type short name test.
        // Getting "e" should return false and the default value of "2.168" is used.
        {
            vector<double> v;
            EXPECT_FALSE( GetOption<double>( options, "", "e", v, "2.168" ) );
            EXPECT_EQ( v.size(), 1 );
            EXPECT_EQ( v[0], 2.168 );
        }

        // Get the vector of non-options and there should be two non-option values.
        {
            vector<string> v;
            OptionArgsBase::STRINGVEC::iterator iterNonOpts;
            iterNonOpts = options.GetNonOptions().begin();
            
            for( ; iterNonOpts != options.GetNonOptions().end(); iterNonOpts++ )
            {
                v.push_back( *iterNonOpts );		
            }

            EXPECT_EQ( v.size(), 2 );
            EXPECT_EQ( v[0], "trailing1" );
            EXPECT_EQ( v[1], "trailing2" );
        }
    }
    IONTEST_CATCH
}


TEST(CommandLineTest, DoubleDashOption)
{
    // Simulated command line argument
    char *argv[] = {
        (char*) "test-prog",
        (char*) "--mult-double", (char*) "0.0,1.1,2.2",
        (char*) "--",
        (char*) "--mult-int", (char*) "-1,-10,0",
        (char*) "--hello", (char*) "world",
        (char*) "-b", (char*) "true",
        (char*) "-d", (char*) "2.0",
        (char*) "-i", (char*) "5",
        (char*) "--unchecked", (char*) "fun",
        (char*) "trailing1",
        (char*) "trailing2"
    };

    // The argument count
    int argc = ( sizeof( argv ) / sizeof( argv[0] ) );
    ASSERT_GE( argc, 1 );

    try
    {
        // Create an instance of the OptionsArg class
        // and process the command line arguments.
        OptionArgs options;
        options.Process( argc, argv );

        // Double type test.
        // Getting "mult-double" should return true and have 2 double values.
        // No default value used.
        {
            vector<double> v;
            EXPECT_TRUE( GetOption<double>( options, "mult-double", v ) );
            EXPECT_EQ( v.size(), 3 );
            EXPECT_EQ( v[0], 0.0 );
            EXPECT_EQ( v[1], 1.1 );
            EXPECT_EQ( v[2], 2.2 );
        }

        // Int type test.
        // Getting "mult-int" should fail since this option follows the "--"
        {
            vector<int> v;
            EXPECT_FALSE( GetOption<int>( options, "mult-int", v, "123" ) );
            EXPECT_EQ( v[0], 123 );
        }

        // Verify the count in the vector of non-options
        {
            vector<string> vecStrArgs;
            OptionArgsBase::STRINGVEC::iterator iterNonOpts;
            iterNonOpts = options.GetNonOptions().begin();
            for( ; iterNonOpts != options.GetNonOptions().end(); iterNonOpts++ )
            {
                vecStrArgs.push_back( *iterNonOpts );
            }
            ASSERT_EQ( vecStrArgs.size(), 15 );
        }
    }
    IONTEST_CATCH
}


TEST(CommandLineJSONTest, WritingJSON)
{
    // The JSON file name.
    const std::string strJSONFileName( "options.json" );
    
    // Simulated command line argument
    char *argv[] = {
        (char*) "test-prog",
        (char*) "--mult-double", (char*) "0.0,1.1,2.2",
        (char*) "--mult-int", (char*) "-1,-10,0",
        (char*) "--hello", (char*) "world",
        (char*) "-b", (char*) "true",
        (char*) "-d", (char*) "2.0",
        (char*) "-i", (char*) "5",
        (char*) "--",
        (char*) "--unchecked", (char*) "fun",
        (char*) "trailing1",
        (char*) "trailing2"
    };

    // The argument count
    int argc = ( sizeof( argv ) / sizeof( argv[0] ) );
    ASSERT_GE( argc, 1 );
    
    try
    {
        // Delete the file if it exists. Return result is irrelevant.
        DeleteFile( strJSONFileName );

        // Verify that the JSON file does not exist.
        EXPECT_FALSE( DoesFileExist( strJSONFileName ) );

        // Create an instance of the OptionsArg class
        // and process the command line arguments.
        OptionArgs options;
        options.Process( argc, argv );

        // Write the current options to the JSON file.
        options.WriteOptions( strJSONFileName );

        // Verify that the new JSON file does exist.
        EXPECT_TRUE( DoesFileExist( strJSONFileName ) );
    }
    IONTEST_CATCH
}


TEST(CommandLineJSONTest, ReadingJSON)
{
    // The JSON file name.
    const std::string strJSONFileName( "options.json" );
    
    // Simulated empty command line argument
    char *argv[] = { (char*) "test-prog" };

    // The argument count
    int argc = ( sizeof( argv ) / sizeof( argv[0] ) );
    ASSERT_GE( argc, 1 );

    try
    {
        // Verify that the JSON file exists.
        EXPECT_TRUE( DoesFileExist( strJSONFileName ) );

        // Create an instance of the OptionsArg class
        // and process the command line arguments.
        OptionArgs options;
        options.Process( argc, argv );

        // Read the current options from the JSON file.
        options.ReadOptions( strJSONFileName );

        // Delete the JSON file we just read.
        EXPECT_TRUE( DeleteFile( strJSONFileName ) );        

        // Verify that the JSON file does not exist.
        EXPECT_FALSE( DoesFileExist( strJSONFileName ) );
    }
    IONTEST_CATCH
}


TEST(CommandLineOptionDefinitionTest, AddWriteNewDefinitions)
{
    // The JSON file name.
    const std::string strJSONFileName( "predefopts.json" );
    
    // Simulated empty command line argument
    char *argv[] = { (char*) "test-prog" };

    // The argument count
    int argc = ( sizeof( argv ) / sizeof( argv[0] ) );
    ASSERT_GE( argc, 1 );

    try
    {
        // Delete the file if it exists. Return result is irrelevant.
        DeleteFile( strJSONFileName );

        // Verify that the JSON file does not exist.
        EXPECT_FALSE( DoesFileExist( strJSONFileName ) );
        
        // Create an instance of the OptionsArg class
        // and process the command line arguments.
        OptionArgs options;
        options.Process( argc, argv );
        
        // Predefine some command line options.
        options.DefineOption( "pi_value", "3.141" );
        options.DefineOption( "e_value", "2.718" );
        options.DefineOption( "golden_ratio", "1.618" );
        options.DefineOption( "c", "299792458" );
        options.DefineOption( "G", "6.67384e-11" );
        options.DefineOption( "plank_length", "" );
        options.DefineOption( "first3", "1,2,3" );

        // Write the current options to the JSON file.
        options.WriteOptions( strJSONFileName );
        
        // Verify that the new JSON file does exist.
        EXPECT_TRUE( DoesFileExist( strJSONFileName ) );
        
        // Getting "PI" should return a value of "3.141".
        {
            vector<float> v;
            EXPECT_TRUE( GetOption<float>( options, "pi_value", v ) );
            EXPECT_EQ( v.size(), 1 );
            EXPECT_EQ( v[0], 3.141f );
        }
        
        // Getting "e" should a value of "2.718".
        {
            vector<double> v;
            EXPECT_TRUE( GetOption<double>( options, "e_value", v ) );
            EXPECT_EQ( v.size(), 1 );
            EXPECT_EQ( v[0], 2.718 );
        }
        
        // Getting "golden_ratio" should return a value of "1.618".
        {
            vector<float> v;
            EXPECT_TRUE( GetOption<float>( options, "golden_ratio", v ) );
            EXPECT_EQ( v.size(), 1 );
            EXPECT_EQ( v[0], 1.618f );
        }
        
        // Getting "c" should return a value of "299792458".
        {
            vector<long> v;
            EXPECT_TRUE( GetOption<long>( options, "c", v ) );
            EXPECT_EQ( v.size(), 1 );
            EXPECT_EQ( v[0], 299792458 );
        }
        
        // Getting "G" should return a value of "6.67384e-11".
        {
            vector<std::string> v;
            EXPECT_TRUE( GetOption<std::string>( options, "G", v ) );
            EXPECT_EQ( v.size(), 1 );
            EXPECT_EQ( v[0], "6.67384e-11" );
        }
        
        // Getting "Boltzmann" should return failure since it does not exist.
        {
            vector<std::string> v;
            EXPECT_FALSE( GetOption<std::string>( options, "Boltzmann", v ) );
        }
        
        // Getting "plank_length" should return true with no arguments.
        {
            vector<std::string> v;
            EXPECT_TRUE( GetOption<std::string>( options, "plank_length", v ) );
            EXPECT_EQ( v.size(), 0 );
        }
        
        // Getting "renormalization" should return false since it does not exist.
        {
            vector<std::string> v;
            EXPECT_FALSE( GetOption<std::string>( options, "renormalization", v ) );
        }

        // Getting "first3" should return true with three parameter values.
        {
            vector<int> v;
            EXPECT_TRUE( GetOption<int>( options, "first3", v ) );
            EXPECT_EQ( v.size(), 3 );
            EXPECT_EQ( v[0], 1 );
            EXPECT_EQ( v[1], 2 );
            EXPECT_EQ( v[2], 3 );
        }
    }
    IONTEST_CATCH
}


TEST(CommandLineOptionDefinitionTest, ReadNewDefinitions)
{
    // The JSON file name.
    const std::string strJSONFileName( "predefopts.json" );
    
    // Simulated empty command line argument
    char *argv[] = { (char*) "test-prog" };

    // The argument count
    int argc = ( sizeof( argv ) / sizeof( argv[0] ) );
    ASSERT_GE( argc, 1 );

    try
    {
        // Verify that the JSON file exists.
        EXPECT_TRUE( DoesFileExist( strJSONFileName ) );

        // Create an instance of the OptionsArg class
        // and process the command line arguments.
        OptionArgs options;
        options.Process( argc, argv );

        // Read the current options from the JSON file.
        options.ReadOptions( strJSONFileName );

        // Getting "PI" should return a value of "3.141".
        {
            vector<float> v;
            EXPECT_TRUE( GetOption<float>( options, "pi_value", v ) );
            EXPECT_EQ( v.size(), 1 );
            EXPECT_EQ( v[0], 3.141f );
        }
        
        // Getting "e" should a value of "2.718".
        {
            vector<double> v;
            EXPECT_TRUE( GetOption<double>( options, "e_value", v ) );
            EXPECT_EQ( v.size(), 1 );
            EXPECT_EQ( v[0], 2.718 );
        }
        
        // Getting "golden_ratio" should return a value of "1.618".
        {
            vector<float> v;
            EXPECT_TRUE( GetOption<float>( options, "golden_ratio", v ) );
            EXPECT_EQ( v.size(), 1 );
            EXPECT_EQ( v[0], 1.618f );
        }
        
        // Getting "c" should return a value of "299792458".
        {
            vector<long> v;
            EXPECT_TRUE( GetOption<long>( options, "c", v ) );
            EXPECT_EQ( v.size(), 1 );
            EXPECT_EQ( v[0], 299792458 );
        }
        
        // Getting "G" should return a value of "6.67384e-11".
        {
            vector<std::string> v;
            EXPECT_TRUE( GetOption<std::string>( options, "G", v ) );
            EXPECT_EQ( v.size(), 1 );
            EXPECT_EQ( v[0], "6.67384e-11" );
        }
        
        // Getting "Boltzmann" should return failure since it does not exist.
        {
            vector<std::string> v;
            EXPECT_FALSE( GetOption<std::string>( options, "Boltzmann", v ) );
        }
        
        // Getting "plank_length" should return true with no arguments.
        {
            vector<std::string> v;
            EXPECT_TRUE( GetOption<std::string>( options, "plank_length", v ) );
            EXPECT_EQ( v.size(), 0 );
        }
        
        // Getting "renormalization" should return false since it does not exist.
        {
            vector<std::string> v;
            EXPECT_FALSE( GetOption<std::string>( options, "renormalization", v ) );
        }
        
        // Getting "first3" should return true with three parameter values.
        {
            vector<int> v;
            EXPECT_TRUE( GetOption<int>( options, "first3", v ) );
            EXPECT_EQ( v.size(), 3 );
            EXPECT_EQ( v[0], 1 );
            EXPECT_EQ( v[1], 2 );
            EXPECT_EQ( v[2], 3 );
        }

        // Delete the JSON file we just read.
        EXPECT_TRUE( DeleteFile( strJSONFileName ) );        

        // Verify that the JSON file does not exist.
        EXPECT_FALSE( DoesFileExist( strJSONFileName ) );
    }
    IONTEST_CATCH
}


TEST(CommandLineOptionDefinitionTest, MergeCommandLinePredefinitions)
{    
    // Simulated command line argument
    char *argv[] = {
        (char*) "test-prog",
        (char*) "--mult-double", (char*) "0.0,1.0",
        (char*) "--mult-int", (char*) "-1,-10,0",
        (char*) "--hello", (char*) "world",
        (char*) "-b", (char*) "true",
        (char*) "-d", (char*) "2.0",
        (char*) "-i", (char*) "5",
        (char*) "--unchecked", (char*) "fun",
        (char*) "trailing1",
        (char*) "trailing2"
    };
    
    // The argument count
    int argc = ( sizeof( argv ) / sizeof( argv[0] ) );
    ASSERT_GE( argc, 1 );

    try
    {
        // Create an instance of the OptionsArg class
        // and process the command line arguments.
        OptionArgs options;
        options.Process( argc, argv );
        
        // Predefine some command line options.
        options.DefineOption( "pi_value", "3.141" );
        options.DefineOption( "e_value", "2.718" );
        options.DefineOption( "golden_ratio", "1.618" );
        options.DefineOption( "c", "299792458" );
        options.DefineOption( "G", "6.67384e-11" );
        options.DefineOption( "plank_length", "" );
        options.DefineOption( "first3", "1,2,3" );        
        options.DefineOption( "mult-double", "100.01,201.201" );
        options.DefineOption( "mult-int", "-1000,-100,0" );

        // Double type test.
        // Getting "mult-double" should return true and have 2 double values.
        // No default value used.
        {
            vector<double> v;
            EXPECT_TRUE( GetOption<double>( options, "mult-double", v ) );
            EXPECT_EQ( v.size(), 2 );
            EXPECT_EQ( v[0], 100.01 );
            EXPECT_EQ( v[1], 201.201 );
        }

        // Int type test.
        // Getting "mult-int" should return true and have 3 int values.
        // No default value used.
        {
            vector<int> v;
            EXPECT_TRUE( GetOption<int>( options, "mult-int", v ) );
            ASSERT_EQ( v.size(), 3 );
            EXPECT_EQ( v[0], -1000 );
            EXPECT_EQ( v[1], -100 );
            EXPECT_EQ( v[2], 0 );
        }
        
        // Getting "PI" should return a value of "3.141".
        {
            vector<float> v;
            EXPECT_TRUE( GetOption<float>( options, "pi_value", v ) );
            EXPECT_EQ( v.size(), 1 );
            EXPECT_EQ( v[0], 3.141f );
        }
        
        // Getting "e" should a value of "2.718".
        {
            vector<double> v;
            EXPECT_TRUE( GetOption<double>( options, "e_value", v ) );
            EXPECT_EQ( v.size(), 1 );
            EXPECT_EQ( v[0], 2.718 );
        }
        
        // Getting "golden_ratio" should return a value of "1.618".
        {
            vector<float> v;
            EXPECT_TRUE( GetOption<float>( options, "golden_ratio", v ) );
            EXPECT_EQ( v.size(), 1 );
            EXPECT_EQ( v[0], 1.618f );
        }
        
        // Getting "c" should return a value of "299792458".
        {
            vector<long> v;
            EXPECT_TRUE( GetOption<long>( options, "c", v ) );
            EXPECT_EQ( v.size(), 1 );
            EXPECT_EQ( v[0], 299792458 );
        }
        
        // Getting "G" should return a value of "6.67384e-11".
        {
            vector<std::string> v;
            EXPECT_TRUE( GetOption<std::string>( options, "G", v ) );
            EXPECT_EQ( v.size(), 1 );
            EXPECT_EQ( v[0], "6.67384e-11" );
        }
        
        // Getting "Boltzmann" should return failure since it does not exist.
        {
            vector<std::string> v;
            EXPECT_FALSE( GetOption<std::string>( options, "Boltzmann", v ) );
        }
        
        // Getting "plank_length" should return true with no arguments.
        {
            vector<std::string> v;
            EXPECT_TRUE( GetOption<std::string>( options, "plank_length", v ) );
            EXPECT_EQ( v.size(), 0 );
        }
        
        // Getting "renormalization" should return false since it does not exist.
        {
            vector<std::string> v;
            EXPECT_FALSE( GetOption<std::string>( options, "renormalization", v ) );
        }

        // Getting "first3" should return true with three parameter values.
        {
            vector<int> v;
            EXPECT_TRUE( GetOption<int>( options, "first3", v ) );
            EXPECT_EQ( v.size(), 3 );
            EXPECT_EQ( v[0], 1 );
            EXPECT_EQ( v[1], 2 );
            EXPECT_EQ( v[2], 3 );
        }
        
        // Getting "d" should return a value of "2.0".
        {
            vector<double> v;
            EXPECT_TRUE( GetOption<double>( options, "d", v ) );
            EXPECT_EQ( v.size(), 1 );
            EXPECT_EQ( v[0], 2.0 );
        }
        
        // Getting "i" should return a value of "5".
        {
            vector<int> v;
            EXPECT_TRUE( GetOption<int>( options, "i", v ) );
            EXPECT_EQ( v.size(), 1 );
            EXPECT_EQ( v[0], 5 );
        }
    }
    IONTEST_CATCH
}
