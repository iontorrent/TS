# Microsoft Developer Studio Project File - Name="AgentEmbeddedNT" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version 6.00
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Static Library" 0x0104

CFG=AgentEmbeddedNT - Win32 Debug SSL
!MESSAGE This is not a valid makefile. To build this project using NMAKE,
!MESSAGE use the Export Makefile command and run
!MESSAGE 
!MESSAGE NMAKE /f "AgentEmbeddedNT.mak".
!MESSAGE 
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "AgentEmbeddedNT.mak" CFG="AgentEmbeddedNT - Win32 Debug SSL"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "AgentEmbeddedNT - Win32 Release" (based on "Win32 (x86) Static Library")
!MESSAGE "AgentEmbeddedNT - Win32 Debug" (based on "Win32 (x86) Static Library")
!MESSAGE "AgentEmbeddedNT - Win32 Release SSL" (based on "Win32 (x86) Static Library")
!MESSAGE "AgentEmbeddedNT - Win32 Debug SSL" (based on "Win32 (x86) Static Library")
!MESSAGE 

# Begin Project
# PROP AllowPerConfigDependencies 0
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""
CPP=cl.exe
RSC=rc.exe

!IF  "$(CFG)" == "AgentEmbeddedNT - Win32 Release"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "Release"
# PROP BASE Intermediate_Dir "Release"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "Release"
# PROP Intermediate_Dir "Release"
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_MBCS" /D "_LIB" /YX /FD /c
# ADD CPP /nologo /MT /W3 /GX /O2 /I ".\\" /I "Cobf" /I "Include" /I "Sysdeps\Win32" /I "Compat" /I "Libsrc\expat\xmlparse" /I "Libsrc\libdes" /I "Libsrc\zlib" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "ENABLE_REMOTE_SESSION" /D "ENABLE_FILE_TRANSFER" /D "ENABLE_LARGEFILE64" /YX /FD /c
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LIB32=link.exe -lib
# ADD BASE LIB32 /nologo
# ADD LIB32 /nologo /out:"Release\AgentEmbedded.lib"

!ELSEIF  "$(CFG)" == "AgentEmbeddedNT - Win32 Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "Debug"
# PROP BASE Intermediate_Dir "Debug"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "Debug"
# PROP Intermediate_Dir "Debug"
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_MBCS" /D "_LIB" /YX /FD /GZ /c
# ADD CPP /nologo /MTd /W3 /Gm /GX /Zi /Od /I ".\\" /I "Cobf" /I "Include" /I "Sysdeps\Win32" /I "Compat" /I "Libsrc\expat\xmlparse" /I "Libsrc\libdes" /I "Libsrc\zlib" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "ENABLE_REMOTE_SESSION" /D "ENABLE_FILE_TRANSFER" /D "ENABLE_LARGEFILE64" /YX /FD /GZ /c
# ADD BASE RSC /l 0x409 /d "_DEBUG"
# ADD RSC /l 0x409 /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LIB32=link.exe -lib
# ADD BASE LIB32 /nologo
# ADD LIB32 /nologo /out:"Debug\AgentEmbedded.lib"

!ELSEIF  "$(CFG)" == "AgentEmbeddedNT - Win32 Release SSL"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "AgentEmbeddedNT___Win32_Release_SSL"
# PROP BASE Intermediate_Dir "AgentEmbeddedNT___Win32_Release_SSL"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "Release_SSL"
# PROP Intermediate_Dir "Release_SSL"
# PROP Target_Dir ""
# ADD BASE CPP /nologo /MT /W3 /GX /O2 /I ".\\" /I "Cobf" /I "Include" /I "Sysdeps\Win32" /I "Compat" /I "Libsrc\expat\xmlparse" /I "Libsrc\libdes" /I "Libsrc\zlib" /D "NDEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "ENABLE_REMOTE_SESSION" /D "ENABLE_FILE_TRANSFER" /YX /FD /c
# ADD CPP /nologo /MT /W3 /GX /O2 /I ".\\" /I "Cobf" /I "Include" /I "Sysdeps\Win32" /I "Compat" /I "Libsrc\expat\xmlparse" /I "Libsrc\libdes" /I "Libsrc\zlib" /I "openssl\include" /D "NDEBUG" /D "HAVE_OPENSSL" /D "ENABLE_SSL" /D "WIN32" /D "_MBCS" /D "_LIB" /D "ENABLE_REMOTE_SESSION" /D "ENABLE_FILE_TRANSFER" /D "ENABLE_LARGEFILE64" /YX /FD /c
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LIB32=link.exe -lib
# ADD BASE LIB32 /nologo /out:"Release\AgentEmbedded.lib"
# ADD LIB32 /nologo /out:"Release_SSL\AgentEmbedded.lib"

!ELSEIF  "$(CFG)" == "AgentEmbeddedNT - Win32 Debug SSL"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "AgentEmbeddedNT___Win32_Debug_SSL"
# PROP BASE Intermediate_Dir "AgentEmbeddedNT___Win32_Debug_SSL"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "Debug_SSL"
# PROP Intermediate_Dir "Debug_SSL"
# PROP Target_Dir ""
# ADD BASE CPP /nologo /MTd /W3 /Gm /GX /Zi /Od /I ".\\" /I "Cobf" /I "Include" /I "Sysdeps\Win32" /I "Compat" /I "Libsrc\expat\xmlparse" /I "Libsrc\libdes" /I "Libsrc\zlib" /D "_DEBUG" /D "WIN32" /D "_MBCS" /D "_LIB" /D "ENABLE_REMOTE_SESSION" /D "ENABLE_FILE_TRANSFER" /YX /FD /GZ /c
# ADD CPP /nologo /MTd /W3 /Gm /GX /Zi /Od /I ".\\" /I "Cobf" /I "Include" /I "Sysdeps\Win32" /I "Compat" /I "Libsrc\expat\xmlparse" /I "Libsrc\libdes" /I "Libsrc\zlib" /I "openssl\include" /D "_DEBUG" /D "HAVE_OPENSSL" /D "ENABLE_SSL" /D "WIN32" /D "_MBCS" /D "_LIB" /D "ENABLE_REMOTE_SESSION" /D "ENABLE_FILE_TRANSFER" /D "ENABLE_LARGEFILE64" /YX /FD /GZ /c
# ADD BASE RSC /l 0x409 /d "_DEBUG"
# ADD RSC /l 0x409 /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LIB32=link.exe -lib
# ADD BASE LIB32 /nologo /out:"Debug\AgentEmbedded.lib"
# ADD LIB32 /nologo /out:"Debug_SSL\AgentEmbedded.lib"

!ENDIF 

# Begin Target

# Name "AgentEmbeddedNT - Win32 Release"
# Name "AgentEmbeddedNT - Win32 Debug"
# Name "AgentEmbeddedNT - Win32 Release SSL"
# Name "AgentEmbeddedNT - Win32 Debug SSL"
# Begin Group "Source Files"

# PROP Default_Filter "cpp;c;cxx;rc;def;r;odl;idl;hpj;bat"
# Begin Source File

SOURCE=.\Cobf\a0.c
# End Source File
# Begin Source File

SOURCE=.\Cobf\a1.c
# End Source File
# Begin Source File

SOURCE=.\Cobf\a10.c
# End Source File
# Begin Source File

SOURCE=.\Cobf\a11.c
# End Source File
# Begin Source File

SOURCE=.\Cobf\a12.c
# End Source File
# Begin Source File

SOURCE=.\Cobf\a13.c
# End Source File
# Begin Source File

SOURCE=.\Cobf\a14.c
# End Source File
# Begin Source File

SOURCE=.\Cobf\a15.c
# End Source File
# Begin Source File

SOURCE=.\Cobf\a16.c
# End Source File
# Begin Source File

SOURCE=.\Cobf\a17.c
# End Source File
# Begin Source File

SOURCE=.\Cobf\a18.c
# End Source File
# Begin Source File

SOURCE=.\Cobf\a19.c
# End Source File
# Begin Source File

SOURCE=.\Cobf\a2.c
# End Source File
# Begin Source File

SOURCE=.\Cobf\a20.c
# End Source File
# Begin Source File

SOURCE=.\Cobf\a21.c
# End Source File
# Begin Source File

SOURCE=.\Cobf\a22.c
# End Source File
# Begin Source File

SOURCE=.\Cobf\a23.c
# End Source File
# Begin Source File

SOURCE=.\Cobf\a24.c
# End Source File
# Begin Source File

SOURCE=.\Cobf\a25.c
# End Source File
# Begin Source File

SOURCE=.\Cobf\a26.c
# End Source File
# Begin Source File

SOURCE=.\Cobf\a3.c
# End Source File
# Begin Source File

SOURCE=.\Cobf\a4.c
# End Source File
# Begin Source File

SOURCE=.\Cobf\a5.c
# End Source File
# Begin Source File

SOURCE=.\Cobf\a6.c
# End Source File
# Begin Source File

SOURCE=.\Cobf\a7.c
# End Source File
# Begin Source File

SOURCE=.\Cobf\a8.c
# End Source File
# Begin Source File

SOURCE=.\Cobf\a9.c
# End Source File
# Begin Source File

SOURCE=.\Sysdeps\Win32\AeOS.c
# End Source File
# Begin Source File

SOURCE=.\Compat\md4c.c
# End Source File
# Begin Source File

SOURCE=.\Compat\md5c.c
# End Source File
# End Group
# Begin Group "Header Files"

# PROP Default_Filter "h;hpp;hxx;hm;inl"
# Begin Source File

SOURCE=.\Compat\AeCompat.h
# End Source File
# Begin Source File

SOURCE=.\Cobf\cobf.h
# End Source File
# End Group
# End Target
# End Project
