# Microsoft Developer Studio Project File - Name="AeDemo3NT" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version 6.00
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Console Application" 0x0103

CFG=AeDemo3NT - Win32 Debug SSL
!MESSAGE This is not a valid makefile. To build this project using NMAKE,
!MESSAGE use the Export Makefile command and run
!MESSAGE 
!MESSAGE NMAKE /f "AeDemo3NT.mak".
!MESSAGE 
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "AeDemo3NT.mak" CFG="AeDemo3NT - Win32 Debug SSL"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "AeDemo3NT - Win32 Release" (based on "Win32 (x86) Console Application")
!MESSAGE "AeDemo3NT - Win32 Debug" (based on "Win32 (x86) Console Application")
!MESSAGE "AeDemo3NT - Win32 Release SSL" (based on "Win32 (x86) Console Application")
!MESSAGE "AeDemo3NT - Win32 Debug SSL" (based on "Win32 (x86) Console Application")
!MESSAGE 

# Begin Project
# PROP AllowPerConfigDependencies 0
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""
CPP=cl.exe
RSC=rc.exe

!IF  "$(CFG)" == "AeDemo3NT - Win32 Release"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "Release"
# PROP BASE Intermediate_Dir "Release"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "Release"
# PROP Intermediate_Dir "Release"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_CONSOLE" /D "_MBCS" /YX /FD /c
# ADD CPP /nologo /MT /W3 /GX /O2 /I "Include" /I "Sysdeps\Win32" /D "NDEBUG" /D "WIN32" /D "_CONSOLE" /D "_MBCS" /D "ENABLE_LARGEFILE64" /YX /FD /c
# SUBTRACT CPP /X
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /machine:I386
# ADD LINK32 AgentEmbedded.lib ws2_32.lib expat.lib libdes.lib zlib.lib /nologo /subsystem:console /machine:I386 /out:"Release/AeDemo3.exe" /libpath:"Release" /libpath:"Libsrc\expat\Release" /libpath:"Libsrc\libdes\Release" /libpath:"Libsrc\zlib\Release"

!ELSEIF  "$(CFG)" == "AeDemo3NT - Win32 Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "AeDemo3NT___Win32_Debug"
# PROP BASE Intermediate_Dir "AeDemo3NT___Win32_Debug"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "Debug"
# PROP Intermediate_Dir "Debug"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_CONSOLE" /D "_MBCS" /YX /FD /GZ /c
# ADD CPP /nologo /MTd /W3 /Gm /GX /Zi /Od /I "Include" /I "Sysdeps\Win32" /D "_DEBUG" /D "WIN32" /D "_CONSOLE" /D "_MBCS" /D "ENABLE_LARGEFILE64" /YX /FD /GZ /c
# ADD BASE RSC /l 0x409 /d "_DEBUG"
# ADD RSC /l 0x409 /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /debug /machine:I386 /pdbtype:sept
# ADD LINK32 AgentEmbedded.lib ws2_32.lib expat.lib libdes.lib zlib.lib /nologo /subsystem:console /debug /machine:I386 /out:"Debug/AeDemo3.exe" /pdbtype:sept /libpath:"Debug" /libpath:"Libsrc\expat\Debug" /libpath:"Libsrc\libdes\Debug" /libpath:"Libsrc\zlib\Debug"
# SUBTRACT LINK32 /incremental:no /nodefaultlib

!ELSEIF  "$(CFG)" == "AeDemo3NT - Win32 Release SSL"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "AeDemo3NT___Win32_Release_SSL"
# PROP BASE Intermediate_Dir "AeDemo3NT___Win32_Release_SSL"
# PROP BASE Ignore_Export_Lib 0
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "Release_SSL"
# PROP Intermediate_Dir "Release_SSL"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /MT /W3 /GX /O2 /I "Include" /I "Sysdeps\Win32" /D "NDEBUG" /D "WIN32" /D "_CONSOLE" /D "_MBCS" /YX /FD /c
# SUBTRACT BASE CPP /X
# ADD CPP /nologo /MT /W3 /GX /O2 /I "Include" /I "Sysdeps\Win32" /D "NDEBUG" /D "WIN32" /D "_CONSOLE" /D "_MBCS" /D "ENABLE_LARGEFILE64" /YX /FD /c
# SUBTRACT CPP /X
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 AgentEmbedded.lib ws2_32.lib expat.lib libdes.lib zlib.lib /nologo /subsystem:console /machine:I386 /out:"Release/AeDemo3.exe" /libpath:"Release" /libpath:"Libsrc\expat\Release" /libpath:"Libsrc\libdes\Release" /libpath:"Libsrc\zlib\Release"
# ADD LINK32 AgentEmbedded.lib ws2_32.lib expat.lib libdes.lib zlib.lib ssleay32.lib libeay32.lib gdi32.lib /nologo /subsystem:console /machine:I386 /out:"Release_SSL/AeDemo3.exe" /libpath:"Release_SSL" /libpath:"Libsrc\expat\Release" /libpath:"Libsrc\libdes\Release" /libpath:"Libsrc\zlib\Release" /libpath:"openssl\lib"

!ELSEIF  "$(CFG)" == "AeDemo3NT - Win32 Debug SSL"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "AeDemo3NT___Win32_Debug_SSL"
# PROP BASE Intermediate_Dir "AeDemo3NT___Win32_Debug_SSL"
# PROP BASE Ignore_Export_Lib 0
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "Debug_SSL"
# PROP Intermediate_Dir "Debug_SSL"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /MTd /W3 /Gm /GX /Zi /Od /I "Include" /I "Sysdeps\Win32" /D "_DEBUG" /D "WIN32" /D "_CONSOLE" /D "_MBCS" /YX /FD /GZ /c
# ADD CPP /nologo /MTd /W3 /Gm /GX /Zi /Od /I "Include" /I "Sysdeps\Win32" /D "_DEBUG" /D "WIN32" /D "_CONSOLE" /D "_MBCS" /D "ENABLE_LARGEFILE64" /YX /FD /GZ /c
# ADD BASE RSC /l 0x409 /d "_DEBUG"
# ADD RSC /l 0x409 /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 AgentEmbedded.lib ws2_32.lib expat.lib libdes.lib zlib.lib /nologo /subsystem:console /debug /machine:I386 /nodefaultlib:"MSVCRT" /out:"Debug/AeDemo3.exe" /pdbtype:sept /libpath:"Debug" /libpath:"Libsrc\expat\Debug" /libpath:"Libsrc\libdes\Debug" /libpath:"Libsrc\zlib\Debug"
# SUBTRACT BASE LINK32 /incremental:no /nodefaultlib
# ADD LINK32 AgentEmbedded.lib ws2_32.lib expat.lib libdes.lib zlib.lib ssleay32.lib libeay32.lib gdi32.lib /nologo /subsystem:console /debug /machine:I386 /nodefaultlib:"LIBCMT" /out:"Debug_SSL/AeDemo3.exe" /pdbtype:sept /libpath:"Debug_SSL" /libpath:"Libsrc\expat\Debug" /libpath:"Libsrc\libdes\Debug" /libpath:"Libsrc\zlib\Debug" /libpath:"openssl\lib"
# SUBTRACT LINK32 /incremental:no /nodefaultlib

!ENDIF 

# Begin Target

# Name "AeDemo3NT - Win32 Release"
# Name "AeDemo3NT - Win32 Debug"
# Name "AeDemo3NT - Win32 Release SSL"
# Name "AeDemo3NT - Win32 Debug SSL"
# Begin Group "Source Files"

# PROP Default_Filter "cpp;c;cxx;rc;def;r;odl;idl;hpj;bat"
# Begin Source File

SOURCE=.\Demo\AeDemo3.c
# End Source File
# Begin Source File

SOURCE=.\Demo\AeDemoCommon.c
# End Source File
# End Group
# Begin Group "Header Files"

# PROP Default_Filter "h;hpp;hxx;hm;inl"
# Begin Source File

SOURCE=.\Demo\AeDemoCommon.h
# End Source File
# End Group
# Begin Group "Resource Files"

# PROP Default_Filter "ico;cur;bmp;dlg;rc2;rct;bin;rgs;gif;jpg;jpeg;jpe"
# End Group
# End Target
# End Project
