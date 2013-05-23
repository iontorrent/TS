/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef PROGRAMSTATE_H
#define PROGRAMSTATE_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include "json/json.h"

#include "CommandLineOpts.h"
#include "ImageSpecClass.h"

class ProgramState {
  public:
    ProgramState(std::string &path);
    void Save(CommandLineOpts &clo, SeqListClass &seq, ImageSpecClass &my_image_spec);
    void WriteState();
    void LoadState(CommandLineOpts &clo, SeqListClass &seq, ImageSpecClass &my_image_spec);
        
  private:  
    std::string outFile;
    Json::Value state_json;
    // CommandLineOpts:
    void AddBkgControl(Json::Value &json, BkgModelControlOpts &bgk_control);
    void SetBkgControl(Json::Value &json, BkgModelControlOpts &bgk_control);
    void AddModControl(Json::Value &json, ModuleControlOpts &mod_control);
    void SetModControl(Json::Value &json, ModuleControlOpts &mod_control);
    void AddKeyContext(Json::Value &json, KeyContext &key_context);
    void SetKeyContext(Json::Value &json, KeyContext &key_context);
    void AddFlowContext(Json::Value &json, FlowContext &flow_context);
    void SetFlowContext(Json::Value &json, FlowContext &flow_context);
    void AddSysContext(Json::Value &json, SystemContext &sys_context);
    void SetSysContext(Json::Value &json, SystemContext &sys_context);
    void AddBfdControl(Json::Value &json, BeadfindControlOpts &bfd_control); //BF options - save but don't need to reload
    void AddLocContext(Json::Value &json, SpatialContext &loc_context);
    void SetLocContext(Json::Value &json, SpatialContext &loc_context);
    void AddImgControl(Json::Value &json, ImageControlOpts &img_control);
    void SetImgControl(Json::Value &json, ImageControlOpts &img_control);
    // SeqList:
    void AddSeqList(Json::Value &json, SeqListClass &seq);
    void SetSeqList(Json::Value &json, SeqListClass &seq);
    // ImageSpecClass:
    void AddImgSpec(Json::Value &json, ImageSpecClass &my_image_spec); 
    void SetImgSpec(Json::Value &json, ImageSpecClass &my_image_spec);
    // ChipID:
    void AddChipID(Json::Value &json);
    void SetChipID(Json::Value &json);
    

    void WriteJson(const Json::Value &json, const std::string& filename_json);
    void LoadJson(Json::Value &json, const std::string& filename_json);
    
    // classes saved:
    std::string clo_name;    
    std::string mod_control_name;
    std::string key_context_name;
    std::string flow_context_name;
    std::string sys_context_name;
    std::string bfd_control_name;
    std::string bkg_control_name;
    std::string loc_context_name;
    std::string img_control_name;
    std::string seq_list_name;
    std::string img_spec_name;
    std::string chip_id_name;
};

#endif // PROGRAMSTATE_H
